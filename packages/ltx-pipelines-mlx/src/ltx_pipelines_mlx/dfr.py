"""DFR (Diffusion Fidelity Rendering) — keyframe-slot base, spatial detailing, tiled temporal rounds.

Ported from ltx-pipelines ``dfr_pipeline.py`` (LTX-2.5). This is the pipeline the
"native multishot generation" and "diffusion fidelity rendering" claims describe.

Shape of the run:

  Stage 1  half-res base + generated keyframe SLOTS on an x8-border segment grid.
           The half-res result is reserved as the detailing reference.
  Stage 2  full res. Video and slots are latent-upsampled; slots are re-appended
           SEEDED with their upsampled stage-1 content (not re-rolled from noise)
           so the detailing pass refines them instead of inventing new ones.
  Rounds   optional temporal x2 densification, ``2**round`` keyframe-seam tiles per
           round, each tile inventing mid-segment slots and denoising with ancestral
           Euler, then stitched on the seam latents. Tiles denoise VIDEO-ONLY
           (``audio_state=None``), matching upstream — audio comes from stage 2 and
           nothing after refines it.

The canvas geometry lives in :mod:`dfr_layout` and is gated bit-exactly against
upstream; this module is the orchestration over it.

⚠️ Conditioning fps is capped at ``_MAX_CONDITIONING_FPS`` INDEPENDENTLY of playback
fps. RoPE time is ``pixel_frame / fps``, so a 120 fps time base halves every token's
temporal span versus the trained distribution and the model can no longer lay out 8
pixel frames inside one latent token — it decodes as a motion spike at each latent
border followed by a stall. Playback fps is used for decoding only.
"""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape, snap_output_dimensions
from ltx_core_mlx.conditioning.mask_utils import first_latent_frame_keyframes_mask
from ltx_core_mlx.conditioning.types.keyframe_cond import VideoConditionByKeyframeIndex
from ltx_core_mlx.conditioning.types.keyframe_slots import (
    VideoGeneratedKeyframeSlots,
    extract_keyframes,
)
from ltx_core_mlx.conditioning.types.reference_video_cond import VideoConditionByReferenceLatent
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import (
    compute_audio_positions,
    compute_audio_token_count,
    compute_video_positions,
)

from .dfr_layout import (
    TEMPORAL_SCALE,
    remap_positions_to_local,
    resolve_canvas,
    stitch_tile_latents,
    tile_ranges,
)
from .distilled import DistilledPipeline
from .scheduler import DISTILLED_SIGMAS, STAGE_2_SIGMAS
from .utils.helpers import create_noised_state
from .utils.progress import phase
from .utils.samplers import denoise_loop

# Anchor keyframes carried between temporal rounds are ours, pinned just short of fully
# clean so a tile can still settle its seam frame.
_ANCHOR_KEYFRAME_STRENGTH = 0.95
_TEMPORAL_ANCESTRAL_ETA = 0.5
_MAX_CONDITIONING_FPS = 60.0


def _slot_initials_from_video(video_latent: mx.array, positions, temporal_scale: int) -> mx.array:
    """Stack the nearest video latent frames as ``(B, C, K, H, W)`` slot seeds."""
    frames = []
    for position in positions:
        index = min(max(round(int(position) / temporal_scale), 0), video_latent.shape[2] - 1)
        frames.append(video_latent[:, :, index : index + 1])
    return mx.concatenate(frames, axis=2)


def _merge_carry_forward_keyframes(anchor_positions, anchor_latents, slot_positions, slot_latents):
    """Next round's anchor bag: carried keyframe stills plus this round's denoised slots.

    Positions must already be on the CURRENT round's pixel grid; the caller remaps (x2)
    for the next round.
    """
    by_position: dict[int, mx.array] = {}
    for positions, latents, label in (
        (anchor_positions, anchor_latents, "anchor"),
        (slot_positions, slot_latents, "slot"),
    ):
        if not positions:
            continue
        if latents is None:
            raise RuntimeError(f"Missing {label} keyframe latents for carry-forward merge")
        if latents.shape[2] != len(positions):
            raise ValueError(f"{label} latents K={latents.shape[2]} != {len(positions)} positions")
        for index, position in enumerate(positions):
            by_position[int(position)] = latents[:, :, index : index + 1]
    if not by_position:
        raise RuntimeError("Carry-forward keyframe bag is empty")
    ordered = sorted(by_position)
    return ordered, mx.concatenate([by_position[p] for p in ordered], axis=2)


class DFRPipeline(DistilledPipeline):
    """Diffusion Fidelity Rendering on a keyframe-slot-capable LTX-2.5 checkpoint.

    Reuses :class:`DistilledPipeline`'s loading, upsampler, and decode paths; replaces
    the generation with the DFR stage/round structure.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_upsampler = None
        self.generated_keyframe_latents: list[mx.array] = []
        self.generated_keyframe_positions: list[int] = []

    # --- weights -----------------------------------------------------------------

    def _load_temporal_upsampler(self) -> None:
        """Load the temporal x2 latent upsampler (mirrors ``_load_upsampler``, other file)."""
        import json

        from ltx_core_mlx.model.upsampler.model import LatentUpsampler
        from ltx_core_mlx.utils.weights import load_split_safetensors

        weights_path = self.model_dir / "temporal_upscaler_x2_v1_0.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Temporal upsampler weights not found at {weights_path}. "
                "temporal_upsample_rounds > 0 requires them; without the trained module the "
                "densified latent decodes into a periodic temporal artefact with no other error."
            )
        config_path = self.model_dir / f"{weights_path.stem}_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text()).get("config", {})
            self.temporal_upsampler = LatentUpsampler.from_config(config)
        else:
            self.temporal_upsampler = LatentUpsampler()
        raw = load_split_safetensors(weights_path)
        stem_prefix = weights_path.stem + "."
        if raw and all(k.startswith(stem_prefix) for k in raw):
            raw = {k[len(stem_prefix) :]: v for k, v in raw.items()}
        self.temporal_upsampler.load_weights(list(raw.items()))
        aggressive_cleanup()

    # --- helpers -----------------------------------------------------------------

    def _upsample_latent(self, latent_bcfhw: mx.array, upsampler) -> mx.array:
        """Denormalize -> neural upsample -> re-normalize, the invariant this repo learned the hard way.

        The upsamplers operate in UN-normalized latent space. Skipping the denorm/renorm
        round-trip produces grid artefacts, not an error.
        """
        x = latent_bcfhw.transpose(0, 2, 3, 4, 1)
        x = self.vae_encoder.denormalize_latent(x)
        x = upsampler(x.transpose(0, 4, 1, 2, 3))
        return self.vae_encoder.normalize_latent(x.transpose(0, 2, 3, 4, 1)).transpose(0, 4, 1, 2, 3)

    def _keyframes_mask(self, state, tokens_per_frame: int):
        """Frame-0 marker UNION the state's own slot marks.

        The frame-0 helper builds a fresh mask; assigning it would unmark every slot and
        silently disable the embedding that makes a slot a keyframe.
        """
        if not getattr(self.dit.config, "use_keyframes_abs_pos_embedding", False):
            return None
        mask = first_latent_frame_keyframes_mask(
            state.latent.shape[1], tokens_per_frame, batch=state.latent.shape[0]
        )
        if state.keyframes_mask is not None:
            mask = mx.maximum(mask, state.keyframes_mask.astype(mask.dtype))
        return mask

    # --- generation ---------------------------------------------------------------

    def generate_dfr(
        self,
        prompt: str,
        height: int = 512,
        width: int = 768,
        num_frames: int = 97,
        *,
        frame_rate: float,
        seed: int = 42,
        temporal_upsample_rounds: int = 0,
        detailing_reference: bool = False,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        **_unused_kwargs,
    ) -> tuple[mx.array, mx.array]:
        """Run DFR. Returns ``(video_latent, audio_latent)`` like the other pipelines.

        ``detailing_reference`` appends the reserved half-res latent as an IC reference in
        stage 2. **Defaults to False, matching upstream**: upstream appends it only when a
        detailing IC-LoRA is loaded, and that LoRA has no default (``--detailing-lora`` is
        opt-in). We do not host one, so the reference would be out of distribution — the
        adapter that teaches the model to read those tokens is exactly what is missing.
        Kept as a flag so it can be exercised once a detailing LoRA is available.
        """
        if temporal_upsample_rounds not in (0, 1, 2):
            raise ValueError(f"temporal_upsample_rounds must be 0, 1, or 2, got {temporal_upsample_rounds}")
        # ⚠️ ORDER IS LOAD-BEARING: encode the prompt and FREE Gemma before loading the
        # DiT. The 2.5 text encoder is a bf16 12B (~24 GB); loading it on top of the
        # 38 GB transformer nearly doubles peak. An earlier revision of this method
        # called load() first and measured 71.8 GB peak vs 40.8 GB for the plain
        # distilled pipeline — that entire "DFR costs +31 GB" result was this ordering,
        # not the feature. It is also the exact mistake commit 1a30f74 fixed for the
        # other pipelines; see CLAUDE.md "Pipeline-load ordering".
        self._load_text_encoder()
        with phase("Encoding prompt", verbose=self.verbose):
            video_embeds, audio_embeds = self._encode_text(prompt)
            mx.eval(video_embeds, audio_embeds)
        if self.low_memory:
            self.prompt_encoder.free()
            aggressive_cleanup()

        self.load()
        if not getattr(self.dit.config, "use_keyframes_abs_pos_embedding", False):
            raise ValueError(
                "DFR requires a generated-keyframe checkpoint (transformer config "
                "use_keyframes_abs_pos_embedding); this model has none, so the slots would be "
                "denoised as unmarked tokens and the extra compute would be wasted."
            )
        if temporal_upsample_rounds > 0 and self.temporal_upsampler is None:
            self._load_temporal_upsampler()

        requested_frames = num_frames
        num_frames, segment, positions = resolve_canvas(num_frames)

        height, width = snap_output_dimensions(height, width, two_stage=True)
        s1_h, s1_w = (height // 2 // 32) * 32, (width // 2 // 32) * 32

        sigmas_1 = DISTILLED_SIGMAS[: stage1_steps + 1] if stage1_steps else DISTILLED_SIGMAS
        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps else STAGE_2_SIGMAS

        F, H1, W1 = compute_video_latent_shape(num_frames, s1_h, s1_w)
        n_audio = compute_audio_token_count(num_frames, frame_rate)
        x0_model = X0Model(self.dit)

        # --- Stage 1: half-res base + keyframe slots ---------------------------------
        slot_item_1 = VideoGeneratedKeyframeSlots(
            positions, frame_rate=frame_rate, sigma=1.0, noise_seed=seed + 20000
        )
        video_state = create_noised_state(
            base_shape=(1, F * H1 * W1, 128),
            conditionings=[slot_item_1],
            spatial_dims=(F, H1, W1),
            positions=compute_video_positions(F, H1, W1, frame_rate),
            seed=seed,
            sigma=1.0,
            legacy_scalar_blend=True,
        )
        audio_state = create_noised_state(
            base_shape=(1, n_audio, 128),
            conditionings=[],
            spatial_dims=(F, H1, W1),
            positions=compute_audio_positions(n_audio),
            seed=seed + 1,
            sigma=1.0,
            legacy_scalar_blend=True,
        )
        with phase(f"DFR stage 1 (half-res + {len(positions)} slots)"):
            out_1 = denoise_loop(
                model=x0_model,
                video_state=video_state,
                audio_state=audio_state,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                sigmas=sigmas_1,
                keyframes_mask=self._keyframes_mask(video_state, H1 * W1),
                ancestral_eta=1.0,
                noise_seed=seed + 10000,
            )
        aggressive_cleanup()

        n_base_1 = F * H1 * W1
        slots_half = extract_keyframes(out_1.video_latent, slot_item_1.layout, H1, W1)
        reserved_half = self.video_patchifier.unpatchify(out_1.video_latent[:, :n_base_1, :], (F, H1, W1))

        # --- Stage 2: spatial detailing ----------------------------------------------
        H2, W2 = H1 * 2, W1 * 2
        upscaled_video = self._upsample_latent(reserved_half, self.upsampler)
        upsampled_slots = self._upsample_latent(mx.concatenate(slots_half, axis=2), self.upsampler)

        slot_item_2 = VideoGeneratedKeyframeSlots(
            positions,
            frame_rate=frame_rate,
            sigma=float(sigmas_2[0]),
            noise_seed=seed + 30000,
            initial_keyframes=upsampled_slots,
        )
        conditionings_2: list = [slot_item_2]
        if detailing_reference:
            conditionings_2.append(
                VideoConditionByReferenceLatent(
                    reference_latent=self.video_patchifier.patchify(reserved_half)[0],
                    # Half-res grid positions; downscale_factor scales the SPATIAL axes
                    # into the full-res frame, leaving time alone.
                    reference_positions=compute_video_positions(F, H1, W1, frame_rate),
                    downscale_factor=2,
                    strength=1.0,
                )
            )
        video_state_2 = create_noised_state(
            base_shape=(1, F * H2 * W2, 128),
            conditionings=conditionings_2,
            spatial_dims=(F, H2, W2),
            positions=compute_video_positions(F, H2, W2, frame_rate),
            seed=seed + 2,
            sigma=float(sigmas_2[0]),
            initial_latent=self.video_patchifier.patchify(upscaled_video)[0],
            legacy_scalar_blend=True,
        )
        audio_state_2 = create_noised_state(
            base_shape=(1, n_audio, 128),
            conditionings=[],
            spatial_dims=(F, H2, W2),
            positions=compute_audio_positions(n_audio),
            seed=seed + 3,
            sigma=float(sigmas_2[0]),
            initial_latent=out_1.audio_latent,
            legacy_scalar_blend=True,
        )
        with phase("DFR stage 2 (full-res detailing)"):
            out_2 = denoise_loop(
                model=x0_model,
                video_state=video_state_2,
                audio_state=audio_state_2,
                video_text_embeds=video_embeds,
                audio_text_embeds=audio_embeds,
                sigmas=sigmas_2,
                keyframes_mask=self._keyframes_mask(video_state_2, H2 * W2),
            )
        aggressive_cleanup()

        n_base_2 = F * H2 * W2
        carry_positions = list(positions)
        carry_keyframes = mx.concatenate(
            extract_keyframes(out_2.video_latent, slot_item_2.layout, H2, W2), axis=2
        )
        video_latent = self.video_patchifier.unpatchify(out_2.video_latent[:, :n_base_2, :], (F, H2, W2))
        # Return the audio in latent form (B, 8, T, 16), like every other pipeline —
        # the decoder unpacks that shape, not the (B, T, 128) token form.
        audio_latent = self.audio_patchifier.unpatchify(out_2.audio_latent)
        current_fps = frame_rate

        # --- Temporal rounds ----------------------------------------------------------
        for round_idx in range(1, temporal_upsample_rounds + 1):
            video_latent = self._upsample_latent(video_latent, self.temporal_upsampler)
            num_frames = 2 * (num_frames - 1) + 1
            current_fps *= 2
            cond_fps = min(current_fps, _MAX_CONDITIONING_FPS)
            seam_positions = [2 * p for p in carry_positions]
            anchor_keyframes = carry_keyframes
            seam_to_index = {s: i for i, s in enumerate(seam_positions)}
            tiles = tile_ranges(seam_positions, num_frames, 2**round_idx, temporal_scale=TEMPORAL_SCALE)

            tile_latents: list[mx.array] = []
            slot_positions_r: list[int] = []
            slot_slices: list[mx.array] = []
            temporal_sigmas = DISTILLED_SIGMAS[4:]

            for tile_index, tile in enumerate(tiles):
                local_frames = (tile.latent_end_exclusive - tile.latent_start - 1) * TEMPORAL_SCALE + 1
                tile_video = video_latent[:, :, tile.latent_start : tile.latent_end_exclusive]
                Ft = tile.latent_end_exclusive - tile.latent_start
                round_conditionings: list = []

                # Every seam in the window is a hard keyframe, including the one at local frame 0.
                anchor_global = list(tile.anchor_kf_global)
                if anchor_global:
                    missing = [p for p in anchor_global if p not in seam_to_index]
                    if missing:
                        raise RuntimeError(f"Anchor seams {missing} missing from the carry-forward bag")
                    for p, local in zip(
                        anchor_global, remap_positions_to_local(anchor_global, tile.pixel_start), strict=True
                    ):
                        i = seam_to_index[p]
                        round_conditionings.append(
                            VideoConditionByKeyframeIndex(
                                frame_idx=local,
                                keyframe_latent=self.video_patchifier.patchify(
                                    anchor_keyframes[:, :, i : i + 1]
                                )[0],
                                spatial_dims=(Ft, H2, W2),
                                frame_rate=cond_fps,
                                strength=_ANCHOR_KEYFRAME_STRENGTH,
                            )
                        )

                slot_global = list(tile.slot_kf_global)
                slot_item_r = None
                if slot_global:
                    slot_local = remap_positions_to_local(slot_global, tile.pixel_start)
                    slot_item_r = VideoGeneratedKeyframeSlots(
                        slot_local,
                        frame_rate=cond_fps,
                        sigma=float(temporal_sigmas[0]),
                        noise_seed=seed + 40000 + 100 * round_idx + tile_index,
                        initial_keyframes=_slot_initials_from_video(tile_video, slot_local, TEMPORAL_SCALE),
                    )
                    round_conditionings.append(slot_item_r)

                tile_state = create_noised_state(
                    base_shape=(1, Ft * H2 * W2, 128),
                    conditionings=round_conditionings,
                    spatial_dims=(Ft, H2, W2),
                    positions=compute_video_positions(Ft, H2, W2, cond_fps),
                    seed=seed + 5000 + 100 * round_idx + tile_index,
                    sigma=float(temporal_sigmas[0]),
                    initial_latent=self.video_patchifier.patchify(tile_video)[0],
                    legacy_scalar_blend=True,
                )
                with phase(f"DFR round {round_idx} tile {tile_index + 1}/{len(tiles)} ({local_frames}f)"):
                    tile_out = denoise_loop(
                        model=x0_model,
                        video_state=tile_state,
                        audio_state=None,
                        video_text_embeds=video_embeds,
                        audio_text_embeds=audio_embeds,
                        sigmas=temporal_sigmas,
                        keyframes_mask=self._keyframes_mask(tile_state, H2 * W2),
                        ancestral_eta=_TEMPORAL_ANCESTRAL_ETA,
                        # Tiles are positionally identical, so a shared ancestral seed
                        # would inject byte-identical noise into every one of them.
                        noise_seed=seed + 1000 * round_idx + tile_index,
                    )
                tile_latents.append(
                    self.video_patchifier.unpatchify(
                        tile_out.video_latent[:, : Ft * H2 * W2, :], (Ft, H2, W2)
                    )
                )
                if slot_item_r is not None:
                    slot_positions_r.extend(slot_global)
                    slot_slices.append(
                        mx.concatenate(
                            extract_keyframes(tile_out.video_latent, slot_item_r.layout, H2, W2), axis=2
                        )
                    )
                aggressive_cleanup()

            video_latent = stitch_tile_latents(tile_latents, tiles)
            expected_t = (num_frames - 1) // TEMPORAL_SCALE + 1
            if video_latent.shape[2] != expected_t:
                raise RuntimeError(f"Stitched latent T={video_latent.shape[2]} != expected {expected_t}")

            slot_latents = mx.concatenate(slot_slices, axis=2) if slot_slices else None
            if slot_positions_r and slot_latents is not None:
                # Lead-in segments repeat the previous tile's slots; the earlier tile's wins.
                first_index: dict[int, int] = {}
                for index, position in enumerate(slot_positions_r):
                    first_index.setdefault(position, index)
                slot_positions_r = sorted(first_index)
                slot_latents = mx.concatenate(
                    [slot_latents[:, :, first_index[p] : first_index[p] + 1] for p in slot_positions_r], axis=2
                )
            carry_positions, carry_keyframes = _merge_carry_forward_keyframes(
                seam_positions, anchor_keyframes, slot_positions_r, slot_latents
            )

        # The canvas may have padded its tail, and each round maps N -> 2(N-1)+1, so the
        # caller's contract is ``(requested - 1) * 2**rounds + 1``.
        target_frames = (requested_frames - 1) * 2**temporal_upsample_rounds + 1
        if target_frames > num_frames:
            raise RuntimeError(f"Target {target_frames} frames exceeds the generated canvas {num_frames}")
        if target_frames != num_frames:
            keep = (target_frames - 1) // TEMPORAL_SCALE + 1
            video_latent = video_latent[:, :, :keep]

        self.generated_keyframe_latents = [
            carry_keyframes[:, :, i : i + 1] for i in range(carry_keyframes.shape[2])
        ]
        self.generated_keyframe_positions = list(carry_positions)
        self.dfr_canvas = {"segment": segment, "canvas_frames": num_frames,
                           "requested_frames": requested_frames, "slot_positions": list(positions)}
        return video_latent, audio_latent


    def generate_and_save(self, prompt: str, output_path: str, **kwargs) -> str:
        """Generate with DFR and save, freeing the DiT before the decoders load.

        ⚠️ The free is load-bearing, not tidiness. Decoding with the transformer still
        resident stacks ~38 GB of weights under the decoder and roughly doubles peak —
        a first C2 measurement that skipped this reported +31 GB for DFR and the number
        was entirely this, not the feature.
        """
        video_latent, audio_latent = self.generate_dfr(
            prompt=prompt, frame_rate=kwargs.pop("frame_rate"), **kwargs
        )
        if self.low_memory:
            self.dit = None
            self.prompt_encoder.free()
            self.image_conditioner.free()
            self.upsampler = None
            self.temporal_upsampler = None
            self._loaded = False
            aggressive_cleanup()
        self._load_decoders()
        result = self._decode_and_save_video(
            video_latent, audio_latent, output_path, frame_rate=kwargs.get("frame_rate", 24.0)
        )
        aggressive_cleanup()
        return result


__all__ = ["DFRPipeline"]
