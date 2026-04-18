"""3D Gaussian Splatting builder using gsplat (nerfstudio-project/gsplat).

Trains a Gaussian Splat scene from video frames and exports to standard
3DGS PLY format viewable in SuperSplat, Luma AI, Polycam, etc.

Two initialization modes
------------------------
sfm   — Uses pycolmap camera poses + 3D point cloud (best quality; requires
        pycolmap and sufficient overlap between frames).
free  — Estimates forward-facing camera poses from frame timestamps when
        pycolmap is unavailable or SfM failed.  Produces lower-quality
        splats but always runs when CUDA + gsplat are available.

Requirements
------------
    pip install gsplat          # CUDA JIT kernels (compiled on first use)
    GPU with CUDA required

Returns
-------
dict:
    splat_ply   : str | None — absolute path to 3DGS PLY (None if skipped)
    viewer_html : str | None — absolute path to standalone HTML viewer
    point_count : int
    train_sec   : float
    method      : "gsplat_sfm" | "gsplat_free" | "skipped"
    skipped     : bool
    reason      : str | None
"""

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from selfsuvis.pipeline.core.logging import get_logger

logger = get_logger(__name__)

# ── Training hyper-parameters ─────────────────────────────────────────────────
_MIN_FRAMES   = 3       # Minimum frames to attempt gsplat
_TRAIN_STEPS  = 3000    # Iterations for small scenes (4–100 frames)
_LR_MEANS     = 1.6e-4
_LR_REST      = 1.0e-3
_MAX_SIDE     = 720     # Resize long side to this for training efficiency
_N_FREE_INIT  = 2000    # Initial Gaussians in pose-free mode
_N_SFM_CAP   = 50_000  # Max Gaussians; prevents OOM on small GPU


def _check_gsplat() -> Tuple[bool, str]:
    """Return (ok, reason). ok=True when gsplat + CUDA are available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        import gsplat  # noqa: F401
        return True, ""
    except ImportError:
        return False, "gsplat not installed (pip install gsplat)"


# ── Public API ────────────────────────────────────────────────────────────────


def build_gaussian_splat(
    frame_list: List[Tuple[str, float]],
    map_dir: Path,
    sfm_frames: Optional[List[Dict[str, Any]]] = None,
    device: str = "cuda",
    max_steps: int = _TRAIN_STEPS,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train 3D Gaussians from video frames and export standard PLY.

    Parameters
    ----------
    frame_list  : List of (frame_path, t_sec) from frame extraction — used
                  for images in pose-free mode.
    map_dir     : Output directory; also contains colmap/ subdir (if present).
    sfm_frames  : Optional list of SfM frame dicts (from run_sfm) each with
                  ``pose_json`` (R, t) and ``pose_status``.  When provided
                  and at least _MIN_FRAMES have status="success", uses real
                  SfM poses.  Falls back to pose-free otherwise.
    device      : Must be "cuda".
    max_steps   : Training iterations.
    seed        : Random seed.
    """
    _skip = lambda r: {
        "splat_ply": None, "viewer_html": None, "point_count": 0,
        "train_sec": 0.0, "method": "skipped", "skipped": True, "reason": r,
    }

    ok, reason = _check_gsplat()
    if not ok:
        return _skip(reason)
    if not frame_list:
        return _skip("no frames")

    try:
        # Decide pose mode
        posed = [f for f in (sfm_frames or []) if f.get("pose_status") == "success"]
        use_sfm = len(posed) >= _MIN_FRAMES

        if use_sfm:
            logger.info("gsplat: using SfM poses (%d frames)", len(posed))
            return _train(posed, map_dir, device, max_steps, seed, mode="sfm")
        else:
            logger.info("gsplat: using pose-free mode (%d keyframes)", len(frame_list))
            synthetic = _make_free_poses(frame_list)
            return _train(synthetic, map_dir, device, max_steps, seed, mode="free")
    except Exception as exc:
        logger.warning("gsplat training failed: %s", exc, exc_info=True)
        return _skip(str(exc))


# ── Core training ─────────────────────────────────────────────────────────────


def _train(
    frames: List[Dict[str, Any]],
    map_dir: Path,
    device: str,
    max_steps: int,
    seed: int,
    mode: str,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy

    torch.manual_seed(seed)
    t_start = time.time()

    # Load data
    images, viewmats, Ks, (H, W) = _load_data(frames, map_dir, device)
    N_views = len(images)
    logger.info("gsplat: %d training views  H=%d W=%d", N_views, H, W)

    # Init Gaussians
    means, scales, quats, opacities, sh0, shN = _init_gaussians(
        frames, map_dir, mode, device
    )
    params = dict(means=means, scales=scales, quats=quats,
                  opacities=opacities, sh0=sh0, shN=shN)

    # Optimisers — one entry per params key (DefaultStrategy requires exact key match)
    optimizers = {
        "means":     torch.optim.Adam([means],     lr=_LR_MEANS),
        "scales":    torch.optim.Adam([scales],    lr=_LR_REST),
        "quats":     torch.optim.Adam([quats],     lr=_LR_REST),
        "opacities": torch.optim.Adam([opacities], lr=_LR_REST),
        "sh0":       torch.optim.Adam([sh0],       lr=_LR_REST),
        "shN":       torch.optim.Adam([shN],       lr=_LR_REST),
    }

    # DefaultStrategy: best for small bounded scenes.
    # refine_stop_iter = half of steps to avoid over-densification on small data.
    refine_stop = max(max_steps // 2, 1000)
    strategy = DefaultStrategy(
        refine_start_iter=300,
        refine_stop_iter=refine_stop,
        refine_every=100,
        reset_every=max_steps + 1,   # disable opacity reset for short runs
        verbose=False,
    )
    state = strategy.initialize_state(scene_scale=1.0)

    for step in range(max_steps):
        idx  = step % N_views
        gt   = images[idx]          # (H, W, 3)
        vmat = viewmats[idx:idx+1]  # (1, 4, 4)
        K    = Ks[idx:idx+1]        # (1, 3, 3)

        # 1. Forward rasterization — info dict is populated with means2d
        colors_sh = torch.cat([sh0, shN], dim=1)   # (N, K, 3)
        render_colors, _alphas, info = rasterization(
            means=means,
            quats=F.normalize(quats, dim=-1),
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities),
            colors=colors_sh.unsqueeze(0),          # (1, N, K, 3)
            viewmats=vmat,
            Ks=K,
            width=W, height=H,
            sh_degree=0,
            packed=True,
            absgrad=True,
            rasterize_mode="antialiased",
        )

        # 2. Hook into means2d gradient BEFORE backward (requires info["means2d"])
        strategy.step_pre_backward(params=params, optimizers=optimizers,
                                   state=state, step=step, info=info)

        # 3. Loss + backward
        loss = F.l1_loss(render_colors[0], gt)
        loss.backward()

        # 4. Post-backward: densification / pruning decisions
        strategy.step_post_backward(params=params, optimizers=optimizers,
                                    state=state, step=step, info=info, packed=True)
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        if (step + 1) % 1000 == 0:
            logger.info("  gsplat %d/%d  loss=%.4f  N=%d",
                        step + 1, max_steps, loss.item(), len(means))

    train_sec = time.time() - t_start

    # Export
    splat_path   = map_dir / "gaussian_splat.ply"
    viewer_html  = map_dir / "view_splat.html"
    with torch.no_grad():
        _export_ply(means.detach(), scales.detach(), quats.detach(),
                    opacities.detach(), sh0.detach(), shN.detach(), splat_path)
    _write_viewer_html(viewer_html, splat_path.name)

    logger.info("gsplat: %d Gaussians → %s  (%.1fs)",
                len(means), splat_path.name, train_sec)
    return {
        "splat_ply":   str(splat_path),
        "viewer_html": str(viewer_html),
        "point_count": int(len(means)),
        "train_sec":   train_sec,
        "method":      f"gsplat_{mode}",
        "skipped":     False,
        "reason":      None,
    }


# ── Data loading ──────────────────────────────────────────────────────────────


def _load_data(
    frames: List[Dict[str, Any]],
    map_dir: Path,
    device: str,
) -> Tuple[List, Any, Any, Tuple[int, int]]:
    """Load images + camera matrices from frame dicts."""
    import torch
    from torchvision import transforms as T

    recon = _load_pycolmap_recon(map_dir / "colmap")
    images, vmats, Ks_list = [], [], []
    H_ref = W_ref = None

    for f in frames:
        img = Image.open(f["frame_path"]).convert("RGB")
        W, H = img.size

        # Resize large frames
        if max(W, H) > _MAX_SIDE:
            scale = _MAX_SIDE / max(W, H)
            W_new, H_new = int(W * scale), int(H * scale)
            img = img.resize((W_new, H_new), Image.BICUBIC)
            W, H = W_new, H_new
        if H_ref is None:
            H_ref, W_ref = H, W

        # World-to-camera 4×4
        pose  = f["pose_json"]
        R     = np.array(pose["R"], dtype=np.float32)
        t_vec = np.array(pose["t"], dtype=np.float32)
        w2c   = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3]  = t_vec
        vmats.append(torch.tensor(w2c))

        # Intrinsics
        cam_id = pose.get("camera_id", 1)
        fx, fy, cx, cy = _get_intrinsics(recon, cam_id, W, H, pose)
        Ks_list.append(torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32
        ))

        img_t = T.ToTensor()(img).permute(1, 2, 0).to(device)
        images.append(img_t)

    viewmats = torch.stack(vmats).to(device)
    Ks = torch.stack(Ks_list).to(device)
    return images, viewmats, Ks, (H_ref, W_ref)


def _get_intrinsics(
    recon, camera_id: int, W: int, H: int, pose: Dict
) -> Tuple[float, float, float, float]:
    """Extract (fx, fy, cx, cy), falling back to estimated pinhole."""
    # Prefer stored intrinsics from pose_json (added by pose-free init)
    if "fx" in pose:
        return pose["fx"], pose["fy"], pose["cx"], pose["cy"]
    # Try pycolmap reconstruction
    if recon is not None:
        try:
            cam    = recon.cameras[camera_id]
            p      = cam.params
            cW, cH = cam.width, cam.height
            model  = (cam.model.name if hasattr(cam.model, "name") else str(cam.model)).upper()
            if "SIMPLE_RADIAL" in model or "SIMPLE_PINHOLE" in model:
                fx = fy = float(p[0]); cx = float(p[1]); cy = float(p[2])
            elif "PINHOLE" in model:
                fx = float(p[0]); fy = float(p[1]); cx = float(p[2]); cy = float(p[3])
            elif "RADIAL" in model:
                fx = fy = float(p[0]); cx = float(p[1]); cy = float(p[2])
            else:
                fx = fy = cW / (2.0 * math.tan(math.radians(35))); cx = cW/2; cy = cH/2
            # Scale for our resized image
            sx = W / cW; sy = H / cH
            return fx * sx, fy * sy, cx * sx, cy * sy
        except Exception:
            pass
    # Estimate: assume ~70° HFOV
    fx = fy = W / (2.0 * math.tan(math.radians(35)))
    return fx, fy, W / 2.0, H / 2.0


# ── Gaussian initialisation ───────────────────────────────────────────────────


def _init_gaussians(
    frames: List[Dict[str, Any]],
    map_dir: Path,
    mode: str,
    device: str,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    pts, cols = _get_init_points(frames, map_dir, mode)

    N     = len(pts)
    pts_t = torch.tensor(pts, device=device)
    col_t = torch.tensor(cols, device=device).clamp(0, 1)

    means     = nn.Parameter(pts_t.clone())
    scales    = nn.Parameter(torch.log(torch.full((N, 3), 0.01, device=device)))
    quats     = nn.Parameter(F.normalize(torch.randn(N, 4, device=device), dim=-1))
    opacities = nn.Parameter(torch.logit(torch.full((N,), 0.1, device=device)))
    sh0       = nn.Parameter(((col_t - 0.5) / 0.28209).unsqueeze(1))  # (N,1,3)
    shN       = nn.Parameter(torch.zeros(N, 0, 3, device=device))

    logger.info("  gsplat init: %d Gaussians (mode=%s)", N, mode)
    return means, scales, quats, opacities, sh0, shN


def _get_init_points(
    frames: List[Dict[str, Any]],
    map_dir: Path,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pts [N,3], cols [N,3]) for Gaussian initialisation."""
    # SfM mode: try pycolmap 3D points first
    if mode == "sfm":
        pts, cols = _load_sfm_points(map_dir / "colmap")
        if pts is not None and len(pts) >= 8:
            # Cap point count to avoid OOM
            if len(pts) > _N_SFM_CAP:
                idx = np.random.choice(len(pts), _N_SFM_CAP, replace=False)
                pts, cols = pts[idx], cols[idx]
            return pts, cols

    # Fallback / free mode: scatter Gaussians in FRONT of camera centres.
    # The free-pose cameras look along world +Z (identity rotation from
    # _make_free_poses), so Gaussians must have z_world > 0.  Initialising at
    # z≈0 places all splats at the camera plane — they are behind or exactly at
    # the near clip, never rendered, and produce zero gradient.
    centres = []
    for f in frames:
        R = np.array(f["pose_json"]["R"], dtype=np.float32)
        t = np.array(f["pose_json"]["t"], dtype=np.float32)
        centres.append(-R.T @ t)
    centres = np.array(centres, dtype=np.float32)
    centroid = centres.mean(axis=0)
    xy_spread = max(float(np.std(centres[:, :2])), 0.3)

    rng = np.random.default_rng(42)
    # Place Gaussians in a frustum volume: z in [0.5, 3.0], x/y spread around centroid
    z_vals = rng.uniform(0.5, 3.0, _N_FREE_INIT).astype(np.float32)
    x_vals = centroid[0] + rng.standard_normal(_N_FREE_INIT).astype(np.float32) * xy_spread
    y_vals = centroid[1] + rng.standard_normal(_N_FREE_INIT).astype(np.float32) * xy_spread * 0.6
    pts  = np.stack([x_vals, y_vals, z_vals], axis=1)
    cols = np.full((len(pts), 3), 0.5, dtype=np.float32)
    return pts, cols


def _load_sfm_points(colmap_dir: Path):
    """Load 3D points from pycolmap Reconstruction. Returns (pts, cols) or (None, None)."""
    try:
        import pycolmap
        recon = _load_pycolmap_recon(colmap_dir)
        if recon is None or len(recon.points3D) == 0:
            return None, None
        pts  = np.array([p.xyz for p in recon.points3D.values()], dtype=np.float32)
        cols = np.array([p.color for p in recon.points3D.values()], dtype=np.float32) / 255.0
        return pts, cols
    except Exception:
        return None, None


def _load_pycolmap_recon(colmap_dir: Path):
    """Load pycolmap Reconstruction; returns None on any failure."""
    try:
        import pycolmap
        for candidate in [colmap_dir / "0", colmap_dir]:
            if candidate.exists():
                r = pycolmap.Reconstruction(str(candidate))
                if r.num_reg_images() > 0:
                    return r
        return None
    except Exception:
        return None


# ── Pose-free camera estimation ───────────────────────────────────────────────


def _make_free_poses(
    frame_list: List[Tuple[str, float]],
) -> List[Dict[str, Any]]:
    """Assign synthetic forward-facing poses to keyframes.

    Places cameras at positions [i * 0.1, 0, 0] (lateral motion along X),
    all looking toward +Z.  Intrinsics are estimated per frame from image size.
    """
    result = []
    n = len(frame_list)
    for i, (fp, t_sec) in enumerate(frame_list):
        # Camera position: (i * baseline, 0, 0) in world space
        baseline = 0.1
        pos_world = np.array([i * baseline, 0.0, 0.0], dtype=np.float32)

        # Rotation: identity — world +X right, +Y down, +Z forward
        R = np.eye(3, dtype=np.float32)
        t = (-R @ pos_world).tolist()  # world-to-camera translation

        # Estimate intrinsics from image size
        try:
            img = Image.open(fp)
            W, H = img.size
        except Exception:
            W, H = 1920, 1080
        if max(W, H) > _MAX_SIDE:
            scale = _MAX_SIDE / max(W, H)
            W = int(W * scale); H = int(H * scale)
        fx = fy = W / (2.0 * math.tan(math.radians(35)))
        cx, cy = W / 2.0, H / 2.0

        result.append({
            "frame_path": fp,
            "t_sec":      t_sec,
            "pose_status": "success",
            "pose_json": {
                "R":  R.tolist(),
                "t":  t,
                "camera_id": 1,
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            },
        })
    return result


# ── PLY export ────────────────────────────────────────────────────────────────


def _export_ply(means, scales, quats, opacities, sh0, shN, path: Path) -> None:
    """Export Gaussians to standard 3DGS PLY format.

    Tries gsplat.export_splats first; falls back to manual plyfile write.
    """
    try:
        from gsplat import export_splats
        export_splats(
            means=means, scales=scales, quats=quats,
            opacities=opacities, sh0=sh0, shN=shN,
            format="ply", save_to=str(path),
        )
        logger.info("  PLY written via gsplat.export_splats → %s", path)
        return
    except Exception as exc:
        logger.debug("gsplat.export_splats failed (%s), using plyfile fallback", exc)

    # Manual plyfile write in 3DGS property format
    import torch
    import torch.nn.functional as F

    N = len(means)
    means_np  = means.cpu().float().numpy()
    opac_np   = opacities.cpu().float().numpy()        # logit space
    scales_np = scales.cpu().float().numpy()            # log space
    # Quaternion: 3DGS convention is (w, x, y, z)
    quats_n   = F.normalize(quats, dim=-1).cpu().float().numpy()
    sh0_np    = sh0.squeeze(1).cpu().float().numpy()    # (N, 3)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    # Add higher-order SH if present
    if shN.shape[1] > 0:
        sh_rest = shN.reshape(N, -1).cpu().float().numpy()
        for k in range(sh_rest.shape[1]):
            dtype.append((f"f_rest_{k}", "f4"))

    from plyfile import PlyData, PlyElement
    vertex = np.empty(N, dtype=dtype)
    vertex["x"] = means_np[:, 0]; vertex["y"] = means_np[:, 1]; vertex["z"] = means_np[:, 2]
    vertex["nx"] = 0; vertex["ny"] = 0; vertex["nz"] = 0
    vertex["f_dc_0"] = sh0_np[:, 0]
    vertex["f_dc_1"] = sh0_np[:, 1]
    vertex["f_dc_2"] = sh0_np[:, 2]
    vertex["opacity"] = opac_np
    vertex["scale_0"] = scales_np[:, 0]
    vertex["scale_1"] = scales_np[:, 1]
    vertex["scale_2"] = scales_np[:, 2]
    # 3DGS stores quats as (w, x, y, z)
    vertex["rot_0"] = quats_n[:, 0]; vertex["rot_1"] = quats_n[:, 1]
    vertex["rot_2"] = quats_n[:, 2]; vertex["rot_3"] = quats_n[:, 3]
    if shN.shape[1] > 0:
        for k in range(sh_rest.shape[1]):
            vertex[f"f_rest_{k}"] = sh_rest[:, k]

    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))
    logger.info("  PLY written via plyfile fallback → %s", path)


# ── HTML viewer ───────────────────────────────────────────────────────────────


def _write_viewer_html(html_path: Path, ply_filename: str) -> None:
    """Generate a standalone HTML viewer using GaussianSplats3D (CDN, no-worker mode)."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>3D Gaussian Splat Viewer</title>
<style>
  body {{ margin:0; background:#0d1117; color:#e6edf3; font-family:sans-serif; }}
  canvas {{ display:block; width:100vw; height:100vh; }}
  #info {{ position:fixed; top:12px; left:12px; background:rgba(0,0,0,.65);
           padding:8px 14px; border-radius:6px; font-size:13px; line-height:1.8; z-index:10; }}
  #info a {{ color:#58a6ff; }}
  #err  {{ color:#f85149; }}
</style>
</head>
<body>
<div id="info">
  <strong>3D Gaussian Splat</strong><br>
  Left-drag rotate &nbsp;·&nbsp; Right-drag pan &nbsp;·&nbsp; Scroll zoom<br>
  Or drag <code>{ply_filename}</code> onto
  <a href="https://playcanvas.com/supersplat/editor" target="_blank">SuperSplat</a>.<br>
  <span id="status">Loading…</span>
  <span id="err"></span>
</div>
<script type="module">
// Uses @mkkellogg/gaussian-splats-3d UMD build — no workers, no SharedArrayBuffer needed.
// Compatible with plain `python -m http.server`.
import GaussianSplats3D from 'https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.1/build/gaussian-splats-3d.module.js';

const {{ Viewer, SceneFormat }} = GaussianSplats3D;

const viewer = new Viewer({{
  cameraUp: [0, -1, 0],
  initialCameraPosition: [0, -1, 3],
  initialCameraLookAt:   [0,  0, 0],
  selfDrivenMode: true,
  useWorkers: false,
  sharedMemoryForWorkers: false,
}});

viewer
  .addSplatScene('./{ply_filename}', {{
    format: SceneFormat.Ply,
    splatAlphaRemovalThreshold: 5,
    showLoadingUI: false,
  }})
  .then(() => {{
    document.getElementById('status').textContent = 'Loaded ✓';
    viewer.start();
  }})
  .catch(e => {{
    document.getElementById('status').textContent = '';
    document.getElementById('err').textContent =
      'Load error: ' + e.message +
      ' — ensure you are serving with python -m http.server';
    console.error(e);
  }});
</script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    logger.info("  Viewer HTML → %s", html_path)
