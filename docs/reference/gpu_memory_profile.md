# GPU Memory Budget Profile

**Device:** NVIDIA GeForce RTX 4060 Ti
**Total VRAM:** 15916 MiB (15.5 GiB)
**Profiled:** FP16 (USE_FP16=true, matches production default)

## Individual Model VRAM

| Model                               | Load (MiB) | Infer peak (MiB) |
|-------------------------------------|-----------|----------------|
| CLIP ViT-B-16 (FP16)                |       304 |           472 |
| DINOv3 ViT-B/14 (FP16)              |       174 |           245 |
| Florence-2-large (FP16)             |      1484 |          2722 |
| Baseline (CUDA ctx + desktop)       |         0 |               — |

## Simultaneous Budget

| Combination                         | Est. total (MiB) | Fits in 15916 MiB? |
|-------------------------------------|-----------------|---------------|
| CLIP + DINOv3 (embedding pass)      |             717 | ✓ Yes |
| Florence-2 + CLIP (caption+embed)   |            3194 | ✓ Yes |
| Florence-2 + CLIP + DINOv3 (all)    |            3439 | ✓ Yes |

**Verdict:** CAN coexist

## Recommendation for `pipeline/indexer.py`

All three models fit simultaneously — no lifecycle management required. Monitor headroom if nerfstudio shares the GPU via docker-compose.override.yml.

## Notes

- RTX 4060 Ti has 15916 MiB. RTX 4090 (24 GB) and A100 (40/80 GB) have more headroom.
- nerfstudio splatfacto peak VRAM is scene-dependent (~8–16 GB for typical outdoor missions).
  On a 16 GB GPU with all worker models loaded, a simultaneous nerfstudio run WILL OOM.
  The v1 architecture already serialises them: indexer completes before mapper starts.
- FP32 doubles all model sizes. Keep USE_FP16=true (default) on the worker.
