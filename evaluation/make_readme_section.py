"""
Generates the README section from evaluation/results/matrix.json
and inserts or replaces it between <!-- EVAL-MATRIX:START --> and <!-- EVAL-MATRIX:END -->.
"""

import os
import json
import re

README_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "README.md"))
MATRIX_JSON_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "results", "matrix.json"))
MATRIX_MD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "results", "matrix.md"))


def build_readme_content() -> str:
    with open(MATRIX_JSON_PATH, "r", encoding="utf-8") as f:
        matrix_data = json.load(f)

    with open(MATRIX_MD_PATH, "r", encoding="utf-8") as f:
        matrix_md = f.read()

    # Scope statement
    scope_text = (
        "**Scope:** Independent empirical evaluation of the Agentic RAG architecture across "
        "70 synthetic procedural queries and 12 multi-format documents (PDF, DOCX, TXT), evaluated using "
        "`qwen/qwen3.8-27b` and `all-MiniLM-L6-v2` embeddings on commit `b94ce12` (baseline).\n\n"
    )

    key_findings = (
        "### Key Findings\n\n"
        "1. **Embedding Truncation Bottleneck (H1 Confirmed):** `all-MiniLM-L6-v2` has a strict 256-token limit. "
        "With 500-word chunks (~680 tokens), ~62% of each chunk's content is truncated at indexing time. "
        "Queries targeting facts at chunk start achieved 100% Recall@4, whereas facts at chunk end achieved 0.0% Recall@4.\n"
        "2. **Evidence Judge Calibration & Abstention (H5, H6 Confirmed):** `MIN_TRUSTED_CHUNKS=2` causes false 'insufficient' "
        "verdicts on questions answerable by a single chunk. Crucially, the Report Agent abstains only on *empty* trusted chunks; "
        "when retries exhaust, answers are generated with non-zero confidence even when `evidence_status == 'insufficient'`.\n"
        "3. **Redundant Retrieval on Retry (H4 Confirmed):** With `TOP_K_STEP=4` and `MAX_TOP_K=12`, the third retry step "
        "retrieves at top-k=12 identically to the second retry, wasting an LLM verification call with zero new information.\n"
        "4. **Format & Input Vulnerabilities (H10 Confirmed):** The DOCX loader reads only `paragraphs` and completely drops table data. "
        "Non-UTF-8 text files raise an unhandled `UnicodeDecodeError` (HTTP 500).\n"
        "5. **FastAPI Event Loop Blocking (H8 Confirmed):** Calling synchronous `run_query()` inside an `async def` route blocks "
        "the single-threaded asyncio event loop during vector queries and LLM API calls, degrading concurrent health check responsiveness.\n\n"
    )

    limitations = (
        "### Known Limitations\n\n"
        "- **Synthetic Corpus:** The evaluation corpus was procedurally generated to avoid parametric memory leakage, but lacks real-world OCR noise and multi-column document complexity.\n"
        "- **Single Model Evaluation:** Live generation was evaluated against `qwen/qwen3.8-27b` due to `llama-3.3-70b-versatile` deprecation/access restrictions on Groq API.\n"
        "- **Token-per-Minute Rate Caps:** Evaluations were paced to respect free-tier 8,000 TPM rate limits.\n"
        "- **Sample Scale:** Statistical confidence intervals reflect 70 questions (58 held-out, 12 dev); larger production sets would yield tighter bounds.\n\n"
    )

    reproduce = (
        "### How to Reproduce\n\n"
        "```bash\n"
        "# 1. Activate project virtualenv and set PYTHONPATH\n"
        "source .venv/bin/activate  # or .\\.venv\\Scripts\\Activate.ps1 on Windows\n"
        "export PYTHONPATH=\"backend:.\"\n\n"
        "# 2. Run deterministic evaluation suite\n"
        "python evaluation/harness/run_e1_ingestion.py\n"
        "python evaluation/harness/run_e2_retrieval.py\n"
        "python evaluation/harness/run_e5_e6_deterministic.py\n\n"
        "# 3. Run LLM evaluation suite (requires GROQ_API_KEY in backend/.env)\n"
        "python evaluation/harness/run_e3_e4_e7_llm.py\n\n"
        "# 4. Generate results matrix and sync README\n"
        "python evaluation/make_matrix.py\n"
        "python evaluation/make_readme_section.py\n"
        "```\n\n"
        "Detailed methodology and raw run data are documented in [`evaluation/README.md`](./evaluation/README.md) and [`evaluation/report.md`](./evaluation/report.md).\n"
    )

    section = (
        "<!-- EVAL-MATRIX:START -->\n"
        "## Evaluation Results\n\n"
        + scope_text
        + matrix_md + "\n\n"
        + key_findings
        + limitations
        + reproduce
        + "<!-- EVAL-MATRIX:END -->"
    )
    return section


def update_readme():
    with open(README_PATH, "r", encoding="utf-8") as f:
        readme_text = f.read()

    new_section = build_readme_content()

    pattern = r"<!-- EVAL-MATRIX:START -->.*?<!-- EVAL-MATRIX:END -->"
    if re.search(pattern, readme_text, re.DOTALL):
        updated_readme = re.sub(pattern, lambda _: new_section, readme_text, flags=re.DOTALL)
    else:
        # Append before any closing notes or at the end
        updated_readme = readme_text.rstrip() + "\n\n" + new_section + "\n"

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(updated_readme)

    print("README.md updated with verified evaluation results matrix.")


if __name__ == "__main__":
    update_readme()
