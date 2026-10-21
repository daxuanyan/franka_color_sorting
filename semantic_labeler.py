"""Assign open-vocabulary labels to RGB-D instance crops with a local VLM."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


def parse_label_response(text: str, allowed: dict[str, str]) -> str:
    """Resolve exactly one allowed ID from a short model response."""
    normalized = text.strip().lower()
    numeric = re.fullmatch(r"(?:option\s*)?(\d+)[.)]?", normalized)
    if numeric:
        index = int(numeric.group(1)) - 1
        labels = list(allowed)
        if 0 <= index < len(labels):
            return labels[index]
    matches = []
    for label in allowed:
        if re.search(rf"(?<![a-z0-9_]){re.escape(label.lower())}(?![a-z0-9_])", normalized):
            matches.append(label)
    if len(matches) != 1:
        raise ValueError(f"expected one allowed label ID, got {matches}: {text!r}")
    return matches[0]


def prototype_label(image: Image.Image, allowed: dict[str, str]) -> tuple[str, float]:
    """Fast colour-prototype baseline used only for ablation and CI."""
    prototypes = {
        "red": np.array([220.0, 50.0, 50.0]),
        "blue": np.array([50.0, 50.0, 220.0]),
        "green": np.array([50.0, 200.0, 50.0]),
        "yellow": np.array([220.0, 210.0, 50.0]),
    }
    pixels = np.asarray(image.convert("RGB"), dtype=float).reshape(-1, 3)
    saturation = np.max(pixels, axis=1) - np.min(pixels, axis=1)
    foreground = pixels[saturation > 30.0]
    mean = np.mean(foreground if len(foreground) else pixels, axis=0)
    candidates = {label: prototypes[label] for label in allowed if label in prototypes}
    if not candidates:
        raise ValueError("prototype backend supports only red/blue/green/yellow IDs")
    distances = {label: float(np.linalg.norm(mean - value)) for label, value in candidates.items()}
    label = min(distances, key=distances.get)
    confidence = max(0.0, 1.0 - distances[label] / 255.0)
    return label, confidence


class SmolVLMLabeler:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        # Avoid network probes when the model is already cached. Users can
        # unset this and pre-download another model explicitly if desired.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_name, local_files_only=True, torch_dtype=self._dtype
        ).to(self._device)
        self._model.eval()

    def classify(self, image: Image.Image, allowed: dict[str, str]) -> tuple[str, float, str]:
        choices = "\n".join(
            f"{index}. {label}: {description}"
            for index, (label, description) in enumerate(allowed.items(), start=1)
        )
        prompt = (
            "Classify the main tabletop object in this crop. Choose exactly one label ID from the list.\n"
            f"{choices}\nReturn only the label ID, with no explanation."
        )
        # Render this model's small chat template directly. This avoids making
        # runtime inference depend on the host application's Jinja version.
        formatted = f"<|im_start|>User:<image>{prompt}<end_of_utterance>\nAssistant:"
        inputs = self._processor(
            text=formatted,
            images=[image.convert("RGB")],
            return_tensors="pt",
        ).to(self._device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self._dtype)
        with self._torch.inference_mode():
            generated = self._model.generate(**inputs, do_sample=False, max_new_tokens=12)
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        raw = self._processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        label = parse_label_response(raw, allowed)
        # Generative VLMs do not provide a calibrated detection confidence.
        # Mark exact constrained outputs conservatively; geometry confidence is
        # combined downstream and remains the final gate authority.
        return label, 0.75, raw


class CLIPLabeler:
    """Batch zero-shot classifier with globally optimal one-to-one matching."""

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        self._model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(self._device)
        self._model.eval()

    def classify_all(
        self, images: list[Image.Image], allowed: dict[str, str]
    ) -> tuple[list[tuple[str, float]], np.ndarray]:
        from scipy.optimize import linear_sum_assignment

        labels = list(allowed)
        prompts = [f"a clear photo of {allowed[label]} on a table" for label in labels]
        inputs = self._processor(
            text=prompts,
            images=[image.convert("RGB") for image in images],
            return_tensors="pt",
            padding=True,
        ).to(self._device)
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
            scores = outputs.logits_per_image.float().cpu().numpy()
        # CLIP's row softmax is a useful relative score, not a calibrated
        # probability. Geometry confidence remains a separate downstream gate.
        probabilities = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        rows, cols = linear_sum_assignment(-scores)
        chosen = [None] * len(images)
        for row, col in zip(rows, cols):
            chosen[int(row)] = (labels[int(col)], float(probabilities[row, col]))
        return chosen, scores


def label_manifest(manifest: dict[str, Any], backend: str, model: str) -> dict[str, Any]:
    labels = manifest.get("labels")
    candidates = manifest.get("candidates")
    if not isinstance(labels, dict) or not labels or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in labels.items()
    ):
        raise ValueError("manifest.labels must map label IDs to text descriptions")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("manifest.candidates must be a non-empty list")
    loaded_candidates: list[tuple[str, Image.Image]] = []
    for candidate in candidates:
        instance_id = candidate.get("instance_id")
        crop_path = candidate.get("crop_path")
        if not isinstance(instance_id, str) or not isinstance(crop_path, str):
            raise ValueError("each candidate requires instance_id and crop_path")
        loaded_candidates.append((instance_id, Image.open(crop_path).convert("RGB")))
    if backend == "clip":
        classifier = CLIPLabeler(model)
        selected, scores = classifier.classify_all([image for _, image in loaded_candidates], labels)
        assignments = {}
        matched = set()
        for (instance_id, _), choice in zip(loaded_candidates, selected):
            if choice is None:
                continue
            label, confidence = choice
            matched.add(label)
            assignments[instance_id] = {
                "label": label,
                "confidence": confidence,
                "raw": "global_clip_assignment",
            }
        return {
            "backend": backend,
            "assignments": assignments,
            "unmatched_labels": sorted(set(labels) - matched),
            "score_matrix": scores.tolist(),
        }

    remaining = dict(labels)
    assignments: dict[str, dict[str, Any]] = {}
    vlm = SmolVLMLabeler(model) if backend == "smolvlm" else None
    for instance_id, image in loaded_candidates:
        if not remaining:
            break
        if vlm is None:
            label, confidence = prototype_label(image, remaining)
            raw = label
        else:
            label, confidence, raw = vlm.classify(image, remaining)
        assignments[instance_id] = {"label": label, "confidence": confidence, "raw": raw}
        remaining.pop(label)
    return {"backend": backend, "assignments": assignments, "unmatched_labels": sorted(remaining)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=("clip", "smolvlm", "prototype"), default="clip")
    parser.add_argument("--model", help="Override the selected backend's default model")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    model = args.model or (DEFAULT_CLIP_MODEL if args.backend == "clip" else DEFAULT_MODEL)
    result = label_manifest(manifest, args.backend, model)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
