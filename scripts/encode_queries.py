#!/usr/bin/env python
"""Verify CLIP feature compatibility, then encode frozen TVR split phrases."""
import argparse
import json
from pathlib import Path

from vcmr.clip_encoding import ClipTextEncoder, _atomic_json, encode_split, verify_existing_queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--splits", nargs="+", default=["dev_select", "dev_confirm", "official_val"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    encoder = ClipTextEncoder(args.root, args.device, args.batch_size)
    report = verify_existing_queries(args.root, encoder, n=128)
    printable = {k: v for k, v in report["verification"].items() if k != "per_query"}
    print(json.dumps({"verification_status": report["status"], **printable}, indent=2), flush=True)
    if report["status"] != "passed":
        raise SystemExit("Encoder mismatch: bulk encoding stopped pending preprocessing investigation")
    if args.verify_only:
        return
    for split in args.splits:
        report["encoding_splits"][split] = encode_split(args.root, split, encoder)
        _atomic_json(args.root / "runs/clip_encoder_verification.json", report)


if __name__ == "__main__":
    main()
