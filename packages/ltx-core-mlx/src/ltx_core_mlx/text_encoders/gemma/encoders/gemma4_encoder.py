"""Gemma 4 unified (LTX-2.5) language-model wrapper via mlx-lm.

The LTX-2.5 text encoder is ``gemma4-12b-ltx-v1`` — a Lightricks-trained
``gemma4_unified`` checkpoint. Same tap contract as Gemma 3 (all 49 hidden
states of dim 3840, LEFT-padded to 1024), with three deltas mirrored from the
v1.2.0 reference (receipts: LTX_TESTING/LTX25-REFERENCE-NOTES.md §B):

1. **BOS is prepended manually** — the Gemma 4 tokenizer emits none
   (``tokenizer.encode`` adds nothing; verified on the converted checkpoint).
2. **The last tapped state is post-final-norm** — the reference taps HF
   ``output_hidden_states``, whose convention is
   ``[embed, layer_1, …, layer_47, norm(layer_48)]``.
3. The layer stack is dual-regime (sliding 8KV×256 vs global MQA 1KV×512,
   ``attention_k_eq_v`` on global layers) — all handled inside mlx-lm's
   ``gemma4_text`` layers; this wrapper only drives the loop.

The uniform causal+padding mask passed to every layer is exact because
``max_length (1024) <= sliding_window (1024)`` — a 1024-token window over a
1024-token sequence IS full causal, so per-layer sliding masks are equivalent
(the same argument the Gemma 3 wrapper relies on).
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx

from .base_encoder import GemmaLanguageModel


class Gemma4LanguageModel(GemmaLanguageModel):
    """LTX-2.5 Gemma-4-unified wrapper (encode-only).

    Loads via ``mlx_lm.load``. The unified checkpoint cannot generate usefully —
    prompt enhancement needs a separate generative Gemma checkpoint, matching
    upstream's ``--prompt-enhancer-gemma-root``.

    ⚠️ **The checkpoint declares ``model_type: gemma4_unified``; released mlx-lm registers
    it as ``gemma4``.** Upstream added a ``MODEL_REMAPPING`` entry for this, but it is NOT
    in any release (absent from v0.31.2 and v0.31.3, present only on ``main``). Rather than
    require an unreleased dependency, ``load`` remaps it here via ``model_config`` — so 2.5
    works on plain ``pip install mlx-lm``. Remove this once a release carries the mapping.
    """

    #: HF convention: the last hidden state in ``output_hidden_states`` is
    #: post-final-norm. Overridable for golden-diff experiments.
    final_norm_last_state: bool = True

    def load(self, model_path: str | None = None) -> None:
        """Load with ``gemma4_unified`` remapped to mlx-lm's ``gemma4``.

        See the class docstring: the remapping exists upstream but is unreleased, so we
        supply it ourselves instead of pinning a git dependency.
        """
        from mlx_lm import load as mlx_lm_load

        path = model_path or self._model_path
        if path is None:
            raise ValueError("model_path must be provided")
        self._model, self._tokenizer = mlx_lm_load(path, model_config={"model_type": "gemma4"})

    def tokenize(self, text: str, max_length: int = 1024) -> tuple[mx.array, mx.array]:
        """Tokenize with a manual BOS and LEFT padding to ``max_length``.

        Mirrors the reference tokenizer (v1.2.0 ``gemma/tokenizer.py:31-63``):
        strip, prepend BOS (Gemma 4 emits none), no EOS, left-pad, truncate to
        the LAST ``max_length`` tokens.
        """
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        bos_id = getattr(self._tokenizer, "bos_token_id", None) or 2
        tokens = self._tokenizer.encode(text.strip())
        # ⚠️ Truncate from the FRONT (keep the head), then prepend BOS, then re-clip —
        # the same order the vendor uses and the same order `base_encoder` uses.
        #
        # 🔑 This override is WHY the earlier fix missed 2.5. `base_encoder.tokenize` was
        # corrected to `tokens[:max_length]`, but this subclass overrides `tokenize`
        # wholesale and kept `tokens[-max_length:]` — so the fix landed on 2.3's path and
        # left 2.5's, the generation it was actually written for. A base-class fix does not
        # reach a subclass that overrides the method.
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        if not tokens or tokens[0] != bos_id:
            tokens = [bos_id, *tokens][:max_length]

        pad_id = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else 0
        pad_length = max_length - len(tokens)
        padded = [pad_id] * pad_length + tokens
        attention_mask = [0] * pad_length + [1] * len(tokens)
        return mx.array([padded]), mx.array([attention_mask])

    def get_all_hidden_states(
        self,
        token_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> list[mx.array]:
        """All 49 hidden states from the Gemma-4 text stack.

        Manual loop over ``language_model.model`` (the mlx-lm
        ``Gemma4TextModel``): embed × embed_scale, then each layer with the
        combined causal+padding mask, collecting every output; the final
        state gets the model's final RMSNorm (HF tap convention).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        self._ensure_metal_headroom()

        # mlx-lm gemma4.Model -> .language_model (gemma4_text.Model) -> .model
        # (Gemma4TextModel with embed_tokens / layers / norm / embed_scale)
        inner = self._model
        for attr in ("language_model", "model"):
            if hasattr(inner, attr):
                inner = getattr(inner, attr)
            if hasattr(inner, "embed_tokens"):
                break
        if not hasattr(inner, "embed_tokens"):
            raise RuntimeError("Cannot find embed_tokens in the Gemma-4 model hierarchy")

        all_hidden_states: list[mx.array] = []

        h = inner.embed_tokens(token_ids)
        # mlx-lm gemma4_text: h = h * self.embed_scale (python float
        # hidden_size**0.5) — replicate exactly, no bf16 pre-round.
        h = h * inner.embed_scale
        all_hidden_states.append(h)

        # One combined causal+padding mask for every layer. Exact for
        # T <= sliding_window (see module docstring).
        T = token_ids.shape[1]
        causal_mask = mx.triu(mx.full((T, T), -1e9, dtype=mx.bfloat16), k=1)
        if attention_mask is not None:
            pad_mask = (1 - attention_mask[:, None, None, :].astype(mx.bfloat16)) * -1e9
            combined_mask = causal_mask[None, None, :, :] + pad_mask
        else:
            combined_mask = causal_mask[None, None, :, :]

        eval_every = int(os.environ.get("LTX2_GEMMA_EVAL_EVERY", "1"))
        n_layers = len(inner.layers)
        for i, layer in enumerate(inner.layers):
            out = layer(h, combined_mask, None, per_layer_input=None, shared_kv=None, offset=None)
            # gemma4_text blocks return (h, kvs, offset)
            h = out[0] if isinstance(out, tuple) else out
            if i == n_layers - 1 and self.final_norm_last_state:
                all_hidden_states.append(inner.norm(h))
            else:
                all_hidden_states.append(h)
            if eval_every and (i + 1) % eval_every == 0:
                mx.eval(all_hidden_states[-1])

        return all_hidden_states

    # --- Prompt enhancement: not supported on the unified checkpoint ---

    def enhance_t2v(self, *args, **kwargs) -> str:
        raise NotImplementedError(
            "gemma4_unified is encode-only; prompt enhancement needs a separate "
            "generative Gemma checkpoint (upstream --prompt-enhancer-gemma-root)."
        )

    def enhance_i2v(self, *args, **kwargs) -> str:
        raise NotImplementedError(
            "gemma4_unified is encode-only; prompt enhancement needs a separate "
            "generative Gemma checkpoint (upstream --prompt-enhancer-gemma-root)."
        )


def resolve_text_encoder(model_dir: str | Path, gemma_model_id: str | None = None) -> GemmaLanguageModel:
    """Pick the right encoder for a converted LTX model directory.

    LTX-2.5 conversions ship the tuned TE in-dir (``gemma4-12b-ltx-v1/``);
    2.3 uses an external Gemma-3 checkpoint (``gemma_model_id``).
    """
    model_dir = Path(model_dir)
    gemma4_dir = model_dir / "gemma4-12b-ltx-v1"
    if gemma4_dir.is_dir():
        return Gemma4LanguageModel(model_path=gemma4_dir)
    return GemmaLanguageModel(model_path=gemma_model_id) if gemma_model_id else GemmaLanguageModel()
