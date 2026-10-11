import lightning.pytorch as pl
import torch

from inference.backend import InferenceBackend, SpectrogramContext
from lib import logging
from lib.path_traversal import materialize_paths


class ForcedAlignmentInferenceModule(pl.LightningModule):
    """Forced alignment inference. Works with any InferenceBackend."""

    def __init__(
            self, backend: InferenceBackend,
            *,
            score_unit: str = "levenshtein",
            skip_penalty: float = 0.5,
    ):
        super().__init__()
        self.backend = backend
        self.score_unit = score_unit
        self.skip_penalty = skip_penalty

    def predict_step(self, batch, batch_idx):
        for msg in batch.get("warning", []):
            logging.warning(msg, callback=self.trainer.progress_bar_callback.print)

        if "waveform" not in batch:
            return []

        waveform = batch["waveform"]  # [B, L]
        duration = batch["duration"]  # [B] seconds
        paths = batch["paths"]

        # ---- Spectrogram ----
        spec = self.backend.spectrogram(waveform, duration)

        # ---- Score ----
        scored = self.backend.score(
            spec, paths=paths, words=batch["words"], candidates=batch["candidates"], unit=self.score_unit,
        )
        tokens, words, groups = materialize_paths(paths, batch["words"], batch["groups"], scored.choices)

        # ---- Align ----
        result = self.backend.align(
            spec, tokens=tokens, groups=groups, unit="frame", skip_penalty=self.skip_penalty,
        )

        # ---- Assemble per-item results ----
        results = []
        for i, identifier in enumerate(batch["identifier"]):
            N_i = int((tokens[i] != 0).sum().item())
            T_i = int(spec.mask[i].sum().item())
            W_i = len(batch["lexicon"][i])

            spec_i = spec.features[i, :T_i]
            tokens_i = tokens[i, :N_i]
            groups_i = groups[i, :N_i]
            spans_i = result.spans[i, :N_i]
            sim_i = result.similarity[i, :T_i, :N_i]
            agreement_i = result.agreement[i].item()

            phonemes = [
                label
                for candidates, choice in zip(batch["lexicon"][i], scored.choices[i, :W_i].tolist())
                if choice > 0
                for label in candidates[choice - 1]["phonemes"]
            ]
            results.append({
                "identifier": identifier,
                "spectrogram": spec_i,
                "tokens": tokens_i,
                "groups": groups_i,
                "words": words[i, :N_i],
                "choices": scored.choices[i, :W_i],
                "scores": scored.scores[i, :W_i] if scored.scores is not None else None,
                "spans": spans_i,
                "similarity": sim_i,
                "agreement": agreement_i,
                "phonemes": phonemes,
                "texts": batch["texts"][i],
                "lexicon": batch["lexicon"][i],
            })

        return results

    def test_step(self, batch, batch_idx) -> dict:
        """Online evaluation on PhonemeTimingDataset batches.

        Returns predicted spans in frames so they share the same unit
        as ground truth (batch["spans"]).  The callback then applies a
        single unit_size_ms to convert both to ms.
        """
        spectrogram = batch["spectrogram"]  # [B, T, C]
        tokens = batch["tokens"]  # [B, N]
        T = spectrogram.shape[1]
        device = spectrogram.device

        mask = torch.arange(T, device=device).unsqueeze(0) < batch["T"].unsqueeze(1)
        spec = SpectrogramContext(features=spectrogram, mask=mask)
        result = self.backend.align(spec, tokens=tokens, unit="frame", skip_penalty=self.skip_penalty)
        return {"spans": result.spans, "similarity": result.similarity}


class OfflineEvaluationModule(pl.LightningModule):
    """Passes paired TextGrid spans to callbacks.

    Batch (from PairedDataset) contains both spans_pred and spans_gt.
    Returns batch["spans_pred"] so the callback sees outputs["spans"].
    """

    def test_step(self, batch, batch_idx) -> dict:
        return {"spans": batch["spans_pred"]}
