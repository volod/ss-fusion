# CLIP + DINOv3 Embeddings Runbook

> Covers: model selection, VRAM management, FP16 vs BF16, DINOv3 vs openclip
> switching, embedding drift, and collection reset.

---

## 1. Architecture overview

```
VideoIndexer
  └─ Pass 0 (per frame, main loop)
       ├─ OpenCLIPEmbedder.encode_image()   → clip vector (512-dim ViT-B/16)
       │    → Qdrant upsert (named vector "clip")
       └─ DINOEmbedder.encode_image()       → dino vector (768-dim ViT-B/14)
             → Qdrant upsert (named vector "dino")   [MODEL_NAME=dinov3 only]
```

Both models stay loaded in the worker for the duration of the run. They are not
offloaded between frames.

---

## 2. Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `openclip` | Embedding backend: `openclip` \| `dinov2` \| `dinov3` |
| `OPENCLIP_MODEL` | `ViT-B-16` | OpenCLIP model architecture |
| `OPENCLIP_PRETRAINED` | `openai` | OpenCLIP pretrained weights tag |
| `DEVICE` | `auto` | Device: `auto` \| `cpu` \| `cuda` |
| `USE_FP16` | `true` | FP16 inference on CUDA (disable for MPS or old GPUs) |

---

## 3. Model selection

| Use case | Recommended setting |
|---|---|
| Default / general search | `MODEL_NAME=openclip` (CLIP ViT-B/16 openai) |
| Better outdoor/drone recall | `MODEL_NAME=dinov3` |
| Max retrieval quality | `OPENCLIP_MODEL=ViT-L-14 OPENCLIP_PRETRAINED=openai` |
| Low VRAM (< 4 GB) | `MODEL_NAME=openclip` (smallest) |

**Switching from openclip to dinov3** adds the `dino` named vector to Qdrant — you
must wipe and re-index or Qdrant will have partial dino coverage. See §6.

**Changing `OPENCLIP_MODEL` or `OPENCLIP_PRETRAINED`** changes the embedding space.
All prior indexed frames become incompatible — full re-index required.

---

## 4. Quick start

```bash
# Default (CLIP only, ViT-B/16 openai)
ssv --mode local

# CLIP + DINOv3 dual embedding
MODEL_NAME=dinov3 ssv --mode local

# Larger CLIP for better retrieval quality
OPENCLIP_MODEL=ViT-L-14 OPENCLIP_PRETRAINED=openai ssv --mode local

# Download / warm model cache
python -m selfsuvis.scripts.prepare_models --clip --dino
```

---

## 5. Health check

```bash
# Verify CLIP and DINOv3 embedder loads and produces correct-shaped output
python -c "
from models.openclip_model import OpenCLIPEmbedder
from PIL import Image
import numpy as np
m = OpenCLIPEmbedder()
img = Image.fromarray(np.zeros((224,224,3), dtype='uint8'))
v = m.encode_image(img)
print('CLIP shape:', v.shape)
"

python -c "
from models.dino_model import DINOEmbedder
from PIL import Image
import numpy as np
m = DINOEmbedder()
img = Image.fromarray(np.zeros((224,224,3), dtype='uint8'))
v = m.encode_image(img)
print('DINO shape:', v.shape)
"
```

Expected: `CLIP shape: (512,)`, `DINO shape: (768,)`.

---

## 6. Qdrant reset after model change

Switching `MODEL_NAME`, `OPENCLIP_MODEL`, or `OPENCLIP_PRETRAINED` invalidates all
existing vector data. Reset before re-indexing:

```bash
scripts/ssv/ssv-reset-qdrant.sh
```

The script drops and recreates the collection. All frames must be re-indexed from
source video files; processed-file dedup registry is **not** cleared (only Qdrant
vectors are dropped).

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: open_clip` | openclip not installed | `pip install open-clip-torch` |
| `CUDA out of memory` | Both CLIP and DINO loaded at once | Switch to CPU: `DEVICE=cpu` or free other models first |
| `cosine similarity always near 0` | FP16 underflow on very old GPU | `USE_FP16=false` |
| Search returns wrong results after model change | Old vectors still in Qdrant | Run `scripts/ssv/ssv-reset-qdrant.sh` and re-index |
| DINOv3 not found | Model not downloaded | `python -m selfsuvis.scripts.prepare_models --dino` |
