"""
Builds the complete synthetic evaluation dataset, indexes chunks,
maps gold chunks, splits DEV and HELD_OUT, and computes SHA-256 manifest.
"""

import os
import sys
import json
import hashlib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

from evaluation.dataset.generate_dataset import generate_documents, run_chunking_and_map_gold
from evaluation.dataset.questions_data import RAW_QUESTIONS

DATASET_DIR = os.path.dirname(__file__)


def main():
    print("Generating synthetic documents...")
    docs_created = generate_documents()
    print(f"Created {len(docs_created)} files.")

    print("Running app chunker and mapping gold chunks...")
    mapped_questions, indexed_chunks = run_chunking_and_map_gold(RAW_QUESTIONS, docs_created)

    total_chunks = sum(len(c) for c in indexed_chunks.values())
    print(f"Total chunks indexed across {len(indexed_chunks)} documents: {total_chunks}")

    dev_questions = [q for q in mapped_questions if q["split"] == "DEV"]
    held_out_questions = [q for q in mapped_questions if q["split"] == "HELD_OUT"]

    print(f"Total questions: {len(mapped_questions)}")
    print(f"  DEV split: {len(dev_questions)} ({len(dev_questions)/len(mapped_questions)*100:.1f}%)")
    print(f"  HELD_OUT split: {len(held_out_questions)} ({len(held_out_questions)/len(mapped_questions)*100:.1f}%)")

    dev_path = os.path.join(DATASET_DIR, "dev_questions.json")
    held_out_path = os.path.join(DATASET_DIR, "held_out_questions.json")
    all_path = os.path.join(DATASET_DIR, "all_questions.json")

    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(dev_questions, f, indent=2)

    # Save frozen held-out with exact formatting
    held_out_bytes = json.dumps(held_out_questions, indent=2, sort_keys=True).encode("utf-8")
    with open(held_out_path, "wb") as f:
        f.write(held_out_bytes)

    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(mapped_questions, f, indent=2)

    held_out_sha256 = hashlib.sha256(held_out_bytes).hexdigest()
    print(f"HELD_OUT SHA-256: {held_out_sha256}")

    # Write dataset manifest
    manifest = {
        "num_documents": len(docs_created),
        "documents": list(docs_created.keys()),
        "total_chunks_indexed": total_chunks,
        "num_questions_total": len(mapped_questions),
        "num_dev": len(dev_questions),
        "num_held_out": len(held_out_questions),
        "held_out_sha256": held_out_sha256,
    }

    with open(os.path.join(DATASET_DIR, "dataset_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Dataset build complete.")


if __name__ == "__main__":
    main()
