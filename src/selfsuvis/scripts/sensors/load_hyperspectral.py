"""Load Indian Pines / Salinas hyperspectral .mat and show band statistics."""

import pathlib

import scipy.io

for mat_file in pathlib.Path(__file__).parent.glob("*.mat"):
    data = scipy.io.loadmat(mat_file)
    cube = next(v for k, v in data.items() if not k.startswith("_"))
    print(
        f"{mat_file.name}: shape={cube.shape} dtype={cube.dtype} "
        f"min={cube.min():.1f} max={cube.max():.1f}"
    )
    # Compute NDVI if at least 4 bands (assume band ordering: B, G, R, NIR, ...)
    if cube.shape[-1] >= 4:
        nir = cube[..., -1].astype(float)
        red = cube[..., -2].astype(float)
        ndvi = (nir - red) / (nir + red + 1e-8)
        print(f"  NDVI mean={ndvi.mean():.3f} std={ndvi.std():.3f}")
