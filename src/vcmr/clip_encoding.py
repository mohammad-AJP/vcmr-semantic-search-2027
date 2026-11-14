"""Encode source-grounded parser phrases with the supplied CLIP backbone.

No retrieval metrics or timestamp labels are used here. Encoder compatibility is
checked against supplied full-query features before any phrase experiments.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch
import clip

from vcmr.data import load_records, load_split, map_clip_query_ids
from vcmr.parser import PARSER_VERSION, parse_query

MODEL_NAME = "ViT-B/32"
MODEL_IDENTIFIER = "OpenAI CLIP ViT-B/32, clip.encode_text"


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


class ClipTextEncoder:
    def __init__(self, root: str | Path, device: str = "cuda", batch_size: int = 256):
        self.root = Path(root)
        self.device = device
        self.batch_size = batch_size
        checkpoint = self.root / ".cache/clip/ViT-B-32.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Expected existing exact CLIP checkpoint: {checkpoint}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA encoder, but CUDA is unavailable")
        self.model, _ = clip.load(MODEL_NAME, device=device, jit=False,
                                  download_root=str(checkpoint.parent))
        self.model.eval()
        try:
            clip_version = importlib.metadata.version("openai-clip")
        except importlib.metadata.PackageNotFoundError:
            clip_version = "unknown"
        self.metadata = {
            "identifier": MODEL_IDENTIFIER,
            "checkpoint": str(checkpoint.relative_to(self.root)),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "clip_package_version": clip_version,
            "torch_version": torch.__version__,
            "device": device,
            "device_name": torch.cuda.get_device_name() if device.startswith("cuda") else "CPU",
            "model_dtype": str(self.model.dtype),
            "tokenization": "clip.tokenize, context_length=77, truncate=True",
            "phrase_output": "float32 L2-normalized 512-dimensional text embeddings",
            "batch_size": batch_size,
        }

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.empty((0, 512), dtype=np.float32)
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                tokens = clip.tokenize(texts[start:start + self.batch_size], truncate=True).to(self.device)
                values = self.model.encode_text(tokens).float()
                if normalize:
                    values = torch.nn.functional.normalize(values, dim=-1)
                outputs.append(values.cpu().numpy())
                if len(texts) > 2048 and (start // self.batch_size + 1) % 20 == 0:
                    print(f"encoded {min(start + self.batch_size, len(texts))}/{len(texts)} unique phrases", flush=True)
        result = np.concatenate(outputs).astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite CLIP output")
        return result


def verify_existing_queries(root: str | Path, encoder: ClipTextEncoder, n: int = 128) -> dict:
    root = Path(root)
    records = load_records(root / "data/tvr/official_train.jsonl")[:n]
    mapping = map_clip_query_ids(root, records)
    encoded = encoder.encode([r["desc"] for r in records], normalize=False)
    with h5py.File(root / "data/tvr/clip_query.h5", "r") as handle:
        supplied = np.stack([np.asarray(handle[mapping[str(r["desc_id"])]], dtype=np.float64).reshape(-1) for r in records])
    if supplied.shape != encoded.shape:
        raise ValueError(f"Supplied/encoded feature shapes differ: {supplied.shape}, {encoded.shape}")
    supplied_norm = np.linalg.norm(supplied, axis=1)
    encoded_norm = np.linalg.norm(encoded.astype(np.float64), axis=1)
    cosine = np.einsum("ij,ij->i", supplied, encoded) / (supplied_norm * encoded_norm)
    rel_norm = np.abs(encoded_norm - supplied_norm) / np.maximum(supplied_norm, 1e-12)
    normalized_distance = np.linalg.norm(encoded / encoded_norm[:, None] - supplied / supplied_norm[:, None], axis=1)
    report = {
        "status": "passed" if float(np.median(cosine)) >= 0.999 else "failed_requires_investigation",
        "encoder": encoder.metadata,
        "verification": {
            "selection": "first 128 records in official_train.jsonl (before internal splitting)",
            "number_queries": len(records), "source": "data/tvr/clip_query.h5",
            "mapping": "manifests/tvr_clip_query_ids.json",
            "cosine_median": float(np.median(cosine)), "cosine_min": float(cosine.min()),
            "cosine_mean": float(cosine.mean()),
            "relative_norm_difference_median": float(np.median(rel_norm)),
            "relative_norm_difference_max": float(rel_norm.max()),
            "supplied_norm_median": float(np.median(supplied_norm)),
            "reencoded_raw_norm_median": float(np.median(encoded_norm)),
            "normalized_l2_difference_median": float(np.median(normalized_distance)),
            "threshold_median_cosine": 0.999,
            "per_query": [{"desc_id": str(r["desc_id"]), "cosine": float(c), "relative_norm_difference": float(d)}
                          for r, c, d in zip(records, cosine, rel_norm)],
        },
        "encoding_splits": {},
    }
    _atomic_json(root / "runs/clip_encoder_verification.json", report)
    return report


def _existing_output_summary(root: Path, split: str, records: list[dict], encoder: ClipTextEncoder):
    h5_path = root / f"data/tvr/phrases_{split}.h5"
    parse_path = root / f"data/tvr/parses_{split}.jsonl"
    if not (h5_path.exists() and parse_path.exists()):
        return None
    expected_ids = {str(r["desc_id"]) for r in records}
    try:
        with h5py.File(h5_path, "r") as handle:
            if not handle.attrs.get("complete", False):
                return None
            if handle.attrs.get("parser_version") != PARSER_VERSION or handle.attrs.get("encoder_checkpoint_sha256") != encoder.metadata["checkpoint_sha256"]:
                return None
            if set(handle.keys()) != expected_ids:
                return None
            phrase_count = int(handle.attrs["number_phrases"])
            unique_count = int(handle.attrs["number_unique_phrases"])
        with parse_path.open() as stream:
            parses = [json.loads(line) for line in stream]
        if {str(p["desc_id"]) for p in parses} != expected_ids or len(parses) != len(records):
            return None
        return {"status": "existing_complete_output", "number_queries": len(records),
                "number_phrases": phrase_count, "number_unique_phrases": unique_count,
                "features": str(h5_path.relative_to(root)), "parses": str(parse_path.relative_to(root))}
    except (OSError, KeyError, ValueError):
        return None


def encode_split(root: str | Path, split: str, encoder: ClipTextEncoder) -> dict:
    root = Path(root)
    records = load_split(root, split)
    existing = _existing_output_summary(root, split, records, encoder)
    if existing:
        print(f"{split}: verified existing complete outputs", flush=True)
        return existing
    started = time.monotonic()
    h5_path = root / f"data/tvr/phrases_{split}.h5"
    parse_path = root / f"data/tvr/parses_{split}.jsonl"
    h5_temp = h5_path.with_name(h5_path.name + ".tmp")
    parse_temp = parse_path.with_name(parse_path.name + ".tmp")
    parsed_records = []
    unique_texts = []
    text_indices = {}
    query_indices = []
    for index, record in enumerate(records):
        parse = parse_query(record["desc"])
        parse["desc_id"] = str(record["desc_id"])
        parse["text_encoder"] = MODEL_IDENTIFIER
        phrases = parse["retrieval_phrases"]
        if not 1 <= len(phrases) <= 6 or phrases[0]["text"] != record["desc"]:
            raise ValueError(f"Parser phrase contract failed for {record['desc_id']}")
        indices = []
        for phrase in phrases:
            if phrase["text"] not in text_indices:
                text_indices[phrase["text"]] = len(unique_texts)
                unique_texts.append(phrase["text"])
            indices.append(text_indices[phrase["text"]])
        query_indices.append(indices)
        parsed_records.append(parse)
        if (index + 1) % 1000 == 0:
            print(f"{split}: parsed {index + 1}/{len(records)} queries", flush=True)
    print(f"{split}: encoding {len(unique_texts)} unique phrases", flush=True)
    embeddings = encoder.encode(unique_texts)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, rtol=2e-5, atol=2e-5):
        raise ValueError("Phrase embeddings are not unit-normalized")
    phrase_count = sum(len(indices) for indices in query_indices)
    with parse_temp.open("w") as stream, h5py.File(h5_temp, "w") as handle:
        handle.attrs["complete"] = False
        handle.attrs["encoder_identifier"] = MODEL_IDENTIFIER
        handle.attrs["encoder_checkpoint_sha256"] = encoder.metadata["checkpoint_sha256"]
        handle.attrs["parser_version"] = PARSER_VERSION
        handle.attrs["split"] = split
        handle.attrs["full_query_phrase_index"] = 0
        handle.attrs["number_queries"] = len(records)
        handle.attrs["number_phrases"] = phrase_count
        handle.attrs["number_unique_phrases"] = len(unique_texts)
        for parse, indices in zip(parsed_records, query_indices):
            values = embeddings[indices]
            handle.create_dataset(parse["desc_id"], data=values, dtype="float32")
            stream.write(json.dumps(parse, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        handle.attrs["complete"] = True
        handle.flush()
    os.replace(parse_temp, parse_path)
    os.replace(h5_temp, h5_path)
    summary = {
        "status": "complete", "number_queries": len(records),
        "number_phrases": phrase_count, "number_unique_phrases": len(unique_texts),
        "elapsed_seconds": time.monotonic() - started,
        "features": str(h5_path.relative_to(root)), "parses": str(parse_path.relative_to(root)),
        "max_unit_norm_error": float(np.abs(norms - 1.0).max()),
        "retrieval_metrics_computed": False,
    }
    print(json.dumps({split: summary}), flush=True)
    return summary
