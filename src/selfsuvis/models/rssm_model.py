"""RSSM (Recurrent State Space Model) for temporal embedding analysis.

Inspired by the DreamerV3 world model architecture described in:
  Romero et al., "Dream to Fly: Model-Based Reinforcement Learning for
  Vision-Based Drone Flight", ICRA 2026.
  https://rpg.ifi.uzh.ch/docs/ICRA26_Romero.pdf

The paper trains DreamerV3 end-to-end from raw pixels to drone commands.
This module adapts the RSSM concept to work on pre-computed CLIP embedding
sequences — operating in semantic latent space rather than pixel space.

Key outputs:
  surprise_scores  — per-frame temporal surprise: how unexpected was this
                     frame given the RSSM's prediction from prior context?
                     surprise_k = cosine_distance(predicted_z̃_k, actual_z_k)
  recurrent_states — per-frame recurrent state h_k (temporally-aware embedding)

The surprise score directly improves the active learning signal:
frames that the world model failed to predict from history are genuinely
novel and most informative for annotation → they get elevated al_score,
which guides SSL fine-tuning and the hydration chain (Steps 28-30).

Architecture (operates in CLIP embedding space):
  Encoder:   Linear(clip_dim → 2*latent_dim)
             → [μ_k, log σ²_k] → z_k via reparameterisation
  Recurrent: GRU(input=latent_dim, hidden=hidden_dim) → h_k
  Dynamics:  Linear(hidden_dim → latent_dim) → z̃_{k+1} (predicted next latent)
  Decoder:   Linear(hidden_dim + latent_dim → clip_dim) → x̂_k (optional)

Training (online, per-mission, ~20 gradient steps):
  Loss = α·MSE(z̃_{k+1}, z_{k+1}) + β·KL(z_k || N(0,1))
  No pre-trained weights needed — adapts to each mission's visual dynamics.

Graceful degradation:
  If torch is unavailable or training fails, falls back to an EMA-based
  surprise estimate (no gradient computation required).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default RSSM hyperparameters — kept small for CPU-friendly operation.
_DEFAULT_HIDDEN_DIM = 256
_DEFAULT_LATENT_DIM = 32
_DEFAULT_TRAIN_STEPS = 20
_DEFAULT_LR = 3e-4
_DEFAULT_KL_BETA = 0.5


class RSSMEmbedder:
    """Lightweight RSSM for per-mission temporal surprise scoring.

    Typical usage::

        embedder = RSSMEmbedder()
        result = embedder.encode_sequence(clip_embeddings)  # (N, dim) float32
        surprise_scores = result["surprise_scores"]   # (N,) float32, higher = more novel
        recurrent_states = result["recurrent_states"] # (N, hidden_dim) float32

    The embedder trains itself online on the mission's CLIP embedding sequence
    before scoring, so no pre-trained checkpoint is required.
    """

    def __init__(
        self,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        latent_dim: int = _DEFAULT_LATENT_DIM,
        train_steps: int = _DEFAULT_TRAIN_STEPS,
        lr: float = _DEFAULT_LR,
        kl_beta: float = _DEFAULT_KL_BETA,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.train_steps = train_steps
        self.lr = lr
        self.kl_beta = kl_beta
        self._torch_available: Optional[bool] = None

    def _check_torch(self) -> bool:
        if self._torch_available is None:
            try:
                import torch  # noqa: F401
                self._torch_available = True
            except ImportError:
                self._torch_available = False
                logger.debug("torch not available — RSSM will use EMA fallback")
        return self._torch_available  # type: ignore[return-value]

    # ── public API ────────────────────────────────────────────────────────────

    def encode_sequence(
        self,
        clip_embeddings: np.ndarray,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Encode a sequence of CLIP embeddings with the RSSM.

        Trains the RSSM online on the provided sequence, then returns per-frame
        surprise scores and recurrent states.

        Args:
            clip_embeddings: (N, clip_dim) float32 array of L2-normalised CLIP embeds.
            device: PyTorch device string ("cpu", "cuda"). Auto-selects when None.

        Returns:
            Dict with keys:
              "surprise_scores"   — (N,) float32, values in [0, 1]. Higher = more novel.
              "recurrent_states"  — (N, hidden_dim) float32. Temporally-aware frame embed.
              "latents"           — (N, latent_dim) float32. Stochastic latent z_k.
              "train_loss"        — Final RSSM training loss (float).
              "method"            — "rssm" or "ema_fallback".
              "model"             — "RSSMEmbedder".
              "hidden_dim"        — hidden_dim used.
              "latent_dim"        — latent_dim used.
        """
        if clip_embeddings.ndim != 2 or len(clip_embeddings) == 0:
            return self._empty_result(len(clip_embeddings) if clip_embeddings.ndim == 2 else 0)

        clip_dim = clip_embeddings.shape[1]

        if self._check_torch():
            try:
                return self._encode_with_rssm(clip_embeddings, clip_dim, device)
            except Exception as exc:
                logger.warning("RSSM inference failed (%s) — using EMA fallback", exc)

        return self._encode_with_ema(clip_embeddings)

    # ── torch RSSM path ───────────────────────────────────────────────────────

    def _encode_with_rssm(
        self,
        embeddings: np.ndarray,
        clip_dim: int,
        device: Optional[str],
    ) -> Dict[str, Any]:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.optim import Adam

        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        emb_t = torch.from_numpy(embeddings.astype(np.float32)).to(dev)  # (N, D)
        N = emb_t.shape[0]

        # ── Model layers ──────────────────────────────────────────────────────
        encoder = nn.Linear(clip_dim, 2 * self.latent_dim).to(dev)
        recurrent = nn.GRUCell(self.latent_dim, self.hidden_dim).to(dev)
        dynamics = nn.Linear(self.hidden_dim, self.latent_dim).to(dev)
        decoder = nn.Linear(self.hidden_dim + self.latent_dim, clip_dim).to(dev)

        # Orthogonal init for better gradient flow
        nn.init.orthogonal_(encoder.weight)
        nn.init.orthogonal_(dynamics.weight)

        params = (
            list(encoder.parameters())
            + list(recurrent.parameters())
            + list(dynamics.parameters())
            + list(decoder.parameters())
        )
        opt = Adam(params, lr=self.lr)

        # ── Online training on mission sequence ───────────────────────────────
        final_loss = float("nan")
        for _ in range(self.train_steps):
            h = torch.zeros(1, self.hidden_dim, device=dev)
            total_loss = torch.tensor(0.0, device=dev)

            for t in range(N):
                x = emb_t[t : t + 1]  # (1, D)

                # Encode observation → stochastic latent z_t
                enc = encoder(x)
                mu, log_var = enc[:, : self.latent_dim], enc[:, self.latent_dim :]
                log_var = log_var.clamp(-10, 2)
                std = (0.5 * log_var).exp()
                eps = torch.randn_like(std)
                z = mu + std * eps  # reparameterised sample

                # Recurrent update: h_{t+1} = GRU(z_t, h_t)
                h = recurrent(z, h)

                # Reconstruction loss
                x_hat = decoder(torch.cat([h, z], dim=-1))
                rec_loss = F.mse_loss(x_hat, x)

                # KL regularisation: KL( N(μ, σ²) || N(0,1) )
                kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()

                # Dynamics prediction: predict z_{t+1} from current h
                if t < N - 1:
                    z_pred = dynamics(h)
                    x_next = emb_t[t + 1 : t + 2]
                    enc_next = encoder(x_next)
                    mu_next = enc_next[:, : self.latent_dim]
                    pred_loss = F.mse_loss(z_pred, mu_next.detach())
                    total_loss = total_loss + rec_loss + self.kl_beta * kl_loss + pred_loss
                else:
                    total_loss = total_loss + rec_loss + self.kl_beta * kl_loss

            opt.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
            final_loss = float(total_loss.item())

        # ── Inference: collect surprise scores and states ─────────────────────
        surprise_scores = np.zeros(N, dtype=np.float32)
        recurrent_states = np.zeros((N, self.hidden_dim), dtype=np.float32)
        latents = np.zeros((N, self.latent_dim), dtype=np.float32)

        with torch.no_grad():
            h = torch.zeros(1, self.hidden_dim, device=dev)
            z_prev_pred: Optional[torch.Tensor] = None

            for t in range(N):
                x = emb_t[t : t + 1]
                enc = encoder(x)
                mu = enc[:, : self.latent_dim]
                log_var = enc[:, self.latent_dim :].clamp(-10, 2)
                std = (0.5 * log_var).exp()
                z = mu + std * torch.randn_like(std)

                if z_prev_pred is not None:
                    # Cosine distance between predicted and actual latent
                    surprise = 1.0 - float(
                        F.cosine_similarity(z_prev_pred, mu, dim=-1).item()
                    )
                    surprise_scores[t] = float(np.clip(surprise, 0.0, 1.0))

                h = recurrent(z, h)
                z_prev_pred = dynamics(h)

                recurrent_states[t] = h.squeeze(0).cpu().numpy()
                latents[t] = z.squeeze(0).cpu().numpy()

        # Normalise surprise scores to [0, 1]
        s_min, s_max = surprise_scores.min(), surprise_scores.max()
        if s_max > s_min:
            surprise_scores = (surprise_scores - s_min) / (s_max - s_min)

        return {
            "surprise_scores": surprise_scores,
            "recurrent_states": recurrent_states,
            "latents": latents,
            "train_loss": final_loss,
            "method": "rssm",
            "model": "RSSMEmbedder",
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
        }

    # ── EMA fallback ──────────────────────────────────────────────────────────

    def _encode_with_ema(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Exponential moving average baseline — no gradient computation required."""
        N, D = embeddings.shape
        alpha = 0.1  # slow EMA → captures longer-range temporal context
        surprise_scores = np.zeros(N, dtype=np.float32)
        ema = embeddings[0].copy()

        for t in range(N):
            if t > 0:
                cos_sim = float(np.dot(ema, embeddings[t]) / (
                    np.linalg.norm(ema) * np.linalg.norm(embeddings[t]) + 1e-9
                ))
                surprise_scores[t] = float(np.clip(1.0 - cos_sim, 0.0, 1.0))
            ema = alpha * embeddings[t] + (1.0 - alpha) * ema

        s_min, s_max = surprise_scores.min(), surprise_scores.max()
        if s_max > s_min:
            surprise_scores = (surprise_scores - s_min) / (s_max - s_min)

        recurrent_states = np.zeros((N, self.hidden_dim), dtype=np.float32)

        return {
            "surprise_scores": surprise_scores,
            "recurrent_states": recurrent_states,
            "latents": np.zeros((N, self.latent_dim), dtype=np.float32),
            "train_loss": float("nan"),
            "method": "ema_fallback",
            "model": "RSSMEmbedder",
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
        }

    def _empty_result(self, n: int) -> Dict[str, Any]:
        return {
            "surprise_scores": np.zeros(n, dtype=np.float32),
            "recurrent_states": np.zeros((n, self.hidden_dim), dtype=np.float32),
            "latents": np.zeros((n, self.latent_dim), dtype=np.float32),
            "train_loss": float("nan"),
            "method": "empty",
            "model": "RSSMEmbedder",
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
        }
