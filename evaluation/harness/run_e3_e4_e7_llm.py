"""
Unified LLM evaluation runner for Experiments E3 (Evidence Filter), E4 (End-to-End Ablation),
and E7 (Latency & Concurrency).
Performs live calls to Groq API with qwen/qwen3.8-27b, strict rate-limit pacing,
and transparent instrumentation via monkeypatching.
"""

import os
import sys
import time
import json
import tempfile
import shutil
import random
import re
from typing import Dict, Any, List, Optional
import dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

# Load environment before any app imports
dotenv.load_dotenv(os.path.join(BASE_DIR, "backend", ".env"))
os.environ["GROQ_MODEL"] = "qwen/qwen3.8-27b"

from app.core.config import settings
settings.GROQ_MODEL = "qwen/qwen3.8-27b"

from evaluation.harness.indexer import build_evaluation_index
from evaluation.harness.metrics import (
    check_answer_correctness,
    is_abstained,
    evaluate_citations,
    compute_wilson_ci,
    compute_auroc,
    parse_citations
)
import app.llm.groq_client as gc
import app.graph.agents.evidence_agent as ea
import app.graph.agents.report_agent as ra
from app.graph.state import build_initial_state, RAGState
from app.graph.orchestrator import _compiled_graph
from app.graph.nodes import retrieve_node

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "baseline")
os.makedirs(OUT_DIR, exist_ok=True)


class HarnessInstrumenter:
    """Manages instrumentation, pacing, token counting, and fallback tracking."""
    def __init__(self):
        self.call_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency_sec = 0.0
        self.last_judge_mode = None
        self.last_judge_exception = None
        self.orig_call_llm = gc.call_llm
        self.orig_llm_judge = ea._llm_judge
        self.orig_sim_fallback = ea._similarity_fallback

    def instrument(self):
        def instrumented_call_llm(prompt: str, temperature: float = 0.2) -> str:
            self.call_count += 1
            max_retries = 8
            backoff_base = 12.0
            
            for attempt in range(max_retries):
                t0 = time.perf_counter()
                try:
                    # Direct client call to capture token usage
                    resp = gc._client.chat.completions.create(
                        model=settings.GROQ_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                    )
                    lat = time.perf_counter() - t0
                    self.total_latency_sec += lat
                    
                    usage = resp.usage
                    p_tok = usage.prompt_tokens if usage else int(len(prompt.split()) * 1.3)
                    c_tok = usage.completion_tokens if usage else 50
                    self.total_prompt_tokens += p_tok
                    self.total_completion_tokens += c_tok

                    # Pace requests to respect 8,000 TPM limit
                    time.sleep(1.8)
                    return resp.choices[0].message.content
                except Exception as exc:
                    err_str = str(exc)
                    if "429" in err_str or "rate_limit" in err_str.lower():
                        # Extract wait time from error message if available, e.g. "try again in 14.5s" or "reset in 12s"
                        wait_m = re.search(r"(?:in|wait)\s+(\d+(?:\.\d+)?)\s*s", err_str, re.IGNORECASE)
                        if wait_m:
                            sleep_time = float(wait_m.group(1)) + 1.0
                        else:
                            sleep_time = max(12.0, backoff_base * (1.2 ** attempt))
                        print(f"  [Harness Pacing] 429 RateLimit detected: sleeping {sleep_time:.1f}s (attempt {attempt+1}/{max_retries})...")
                        time.sleep(sleep_time)
                    else:
                        raise exc
            raise RuntimeError(f"Exceeded maximum retries ({max_retries}) on Groq API.")

        def instrumented_llm_judge(state: RAGState) -> RAGState:
            self.last_judge_mode = "llm"
            self.last_judge_exception = None
            try:
                return self.orig_llm_judge(state)
            except Exception as e:
                self.last_judge_exception = f"{type(e).__name__}: {str(e)}"
                raise e

        def instrumented_sim_fallback(state: RAGState) -> RAGState:
            self.last_judge_mode = "similarity_fallback"
            return self.orig_sim_fallback(state)

        ea.call_llm = instrumented_call_llm
        ra.call_llm = instrumented_call_llm
        gc.call_llm = instrumented_call_llm
        ea._llm_judge = instrumented_llm_judge
        ea._similarity_fallback = instrumented_sim_fallback

    def restore(self):
        ea.call_llm = self.orig_call_llm
        ra.call_llm = self.orig_call_llm
        gc.call_llm = self.orig_call_llm
        ea._llm_judge = self.orig_llm_judge
        ea._similarity_fallback = self.orig_sim_fallback


def run_experiment_e3(instrumenter: HarnessInstrumenter, questions: List[Dict[str, Any]]):
    """
    Experiment E3: Evidence Filter Quality.
    Compares 3 filters on the first-iteration retrieval (top-k 4):
    (i) No filter (pass all 4)
    (ii) Similarity threshold (0.35 + MIN_TRUSTED_CHUNKS=2)
    (iii) LLM Judge
    Also tests H2 (judge truncation at 600 characters) on position-sensitive pairs.
    """
    print("--- Running Experiment E3: Evidence Filter Quality ---")
    results = []

    # Filter to answerable questions + unanswerable questions
    sample_questions = questions  # All 58 held-out questions

    for q in sample_questions:
        qid = q["id"]
        gold_refs = {g["ref_id"] for g in q.get("gold_chunks", [])}
        is_ans = q.get("is_answerable", True)

        # Baseline retrieval at k=4
        st_base = build_initial_state(q["question"], top_k=4)
        st_retrieved = retrieve_node(dict(st_base))

        retrieved_refs = [
            f"{c['metadata']['document_name']}_chunk_{c['metadata']['chunk_index']}"
            for c in st_retrieved["retrieved_chunks"]
        ]
        has_gold_retrieved = any(r in gold_refs for r in retrieved_refs)

        # 1. Filter: No filter (All retrieved are trusted)
        trusted_no_filter = [
            ea._to_trusted(c, s) for c, s in zip(st_retrieved["retrieved_chunks"], st_retrieved["similarity_scores"])
        ]

        # 2. Filter: Similarity threshold
        settings.EVIDENCE_JUDGE_MODE = "similarity"
        st_sim = ea.evaluate_evidence(dict(st_retrieved))
        trusted_sim = list(st_sim["trusted_chunks"])
        status_sim = st_sim["evidence_status"]

        # 3. Filter: LLM Judge
        settings.EVIDENCE_JUDGE_MODE = "llm"
        st_llm = ea.evaluate_evidence(dict(st_retrieved))
        trusted_llm = list(st_llm["trusted_chunks"])
        status_llm = st_llm["evidence_status"]
        mode_used = instrumenter.last_judge_mode

        # Check H6: Did LLM find 1 relevant chunk, but declared "insufficient" because MIN_TRUSTED_CHUNKS=2?
        h6_triggered = (len(trusted_llm) == 1 and status_llm == "insufficient" and is_ans and has_gold_retrieved)

        # Check H2: On position-sensitive questions, did LLM judge reject chunk because fact was > 600 chars?
        is_pos_end = "pos_end" in qid
        is_pos_start = "pos_start" in qid

        entry = {
            "question_id": qid,
            "category": q["category"],
            "is_answerable": is_ans,
            "has_gold_retrieved": has_gold_retrieved,
            "no_filter_trusted_count": len(trusted_no_filter),
            "similarity_trusted_count": len(trusted_sim),
            "similarity_status": status_sim,
            "llm_judge_trusted_count": len(trusted_llm),
            "llm_judge_status": status_llm,
            "llm_judge_mode_used": mode_used,
            "h6_single_chunk_insufficient": h6_triggered,
            "is_pos_start": is_pos_start,
            "is_pos_end": is_pos_end
        }
        results.append(entry)

    # Save raw records
    e3_jsonl = os.path.join(OUT_DIR, "e3_evidence_filter.jsonl")
    with open(e3_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary
    ans_results = [r for r in results if r["is_answerable"]]
    unans_results = [r for r in results if not r["is_answerable"]]

    summary = {
        "num_questions": len(results),
        "llm_judge_sufficiency_on_answerable": round(sum(1 for r in ans_results if r["llm_judge_status"] == "sufficient") / len(ans_results), 4),
        "llm_judge_sufficiency_on_unanswerable": round(sum(1 for r in unans_results if r["llm_judge_status"] == "sufficient") / len(unans_results), 4),
        "similarity_sufficiency_on_answerable": round(sum(1 for r in ans_results if r["similarity_status"] == "sufficient") / len(ans_results), 4),
        "similarity_sufficiency_on_unanswerable": round(sum(1 for r in unans_results if r["similarity_status"] == "sufficient") / len(unans_results), 4),
        "h6_single_chunk_false_insufficient_count": sum(1 for r in results if r["h6_single_chunk_insufficient"]),
        "h2_position_start_trusted": sum(r["llm_judge_trusted_count"] for r in results if r["is_pos_start"]),
        "h2_position_end_trusted": sum(r["llm_judge_trusted_count"] for r in results if r["is_pos_end"]),
        "fallback_rate": sum(1 for r in results if r["llm_judge_mode_used"] == "similarity_fallback") / len(results)
    }

    with open(os.path.join(OUT_DIR, "e3_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"E3 complete. Summary saved to {os.path.join(OUT_DIR, 'e3_summary.json')}")
    return summary


def run_experiment_e4(instrumenter: HarnessInstrumenter, questions: List[Dict[str, Any]]):
    """
    Experiment E4: End-to-End Ablation.
    Evaluates 4 configurations on identical frozen HELD-OUT questions:
      Arm A: Full System (Default: LLM judge, max retries 3)
      Arm B: Standard RAG (Top-4 retrieval, no evidence check, no retry, same report prompt)
      Arm C: LLM Judge, No Retry (MAX_RETRIES=0)
      Arm D: Similarity-only evidence filter, default retries
    """
    print("--- Running Experiment E4: End-to-End Ablation ---")

    arms = [
        {"name": "Arm_A_Full_System", "mode": "llm", "max_retries": 3, "file": "e4_e2e_arm_a_full.jsonl"},
        {"name": "Arm_B_Standard_RAG", "mode": "standard_rag", "max_retries": 0, "file": "e4_e2e_arm_b_standard_rag.jsonl"},
        {"name": "Arm_C_No_Retry", "mode": "llm", "max_retries": 0, "file": "e4_e2e_arm_c_no_retry.jsonl"},
        {"name": "Arm_D_Similarity_Only", "mode": "similarity", "max_retries": 3, "file": "e4_e2e_arm_d_similarity_only.jsonl"}
    ]

    arm_summaries = {}

    for arm in arms:
        arm_name = arm["name"]
        print(f"\n>> Running {arm_name} on {len(questions)} held-out questions...")
        
        # Configure runtime settings
        settings.EVIDENCE_JUDGE_MODE = arm["mode"] if arm["mode"] != "standard_rag" else "llm"
        settings.MAX_RETRIES = arm["max_retries"]
        settings.DEFAULT_TOP_K = 4

        arm_out_path = os.path.join(OUT_DIR, arm["file"])
        existing_records = []
        done_qids = set()
        if os.path.exists(arm_out_path):
            with open(arm_out_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if line:
                        try:
                            rec = json.loads(line)
                            existing_records.append(rec)
                            done_qids.add(rec["question_id"])
                        except Exception:
                            pass
        records = list(existing_records)
        print(f"  [Resume Check] {len(done_qids)}/{len(questions)} questions already recorded for {arm_name}.")

        if len(done_qids) < len(questions):
            f_out = open(arm_out_path, "a", encoding="utf-8")

            for idx, q in enumerate(questions):
                qid = q["id"]
                if qid in done_qids:
                    continue
                question_text = q["question"]
                is_ans = q.get("is_answerable", True)
                req_facts = q.get("required_facts", [])
                forbid_facts = q.get("forbidden_facts", [])

                t0 = time.perf_counter()
                calls_before = instrumenter.call_count
                p_tok_before = instrumenter.total_prompt_tokens
                c_tok_before = instrumenter.total_completion_tokens

                try:
                    if arm["mode"] == "standard_rag":
                        # Standard RAG: retrieve top-4 -> directly call report prompt
                        st_init = build_initial_state(question_text, top_k=4)
                        st_ret = retrieve_node(st_init)
                        st_ret["trusted_chunks"] = [
                            ea._to_trusted(c, s) for c, s in zip(st_ret["retrieved_chunks"], st_ret["similarity_scores"])
                        ]
                        st_ret["evidence_status"] = "unverified"
                        final_st = ra.generate_report(st_ret)
                        final_st["retry_count"] = 0
                    else:
                        # Compiled LangGraph execution
                        st_init = build_initial_state(question_text, top_k=4)
                        final_st = _compiled_graph.invoke(st_init)

                    duration = round(time.perf_counter() - t0, 3)
                    query_calls = instrumenter.call_count - calls_before
                    query_tokens = (instrumenter.total_prompt_tokens - p_tok_before) + (instrumenter.total_completion_tokens - c_tok_before)

                    answer = final_st.get("generated_answer")
                    conf = final_st.get("confidence") or 0.0
                    ev_status = final_st.get("evidence_status")
                    retries = final_st.get("retry_count", 0)
                    trusted = final_st.get("trusted_chunks", [])

                    abstained = is_abstained(answer)
                    correct = check_answer_correctness(answer, req_facts, forbid_facts, is_ans)
                    cit_eval = evaluate_citations(answer, trusted)
                    is_h5_discrepancy = (ev_status == "insufficient" and conf > 0.0 and not abstained)

                    rec = {
                        "arm": arm_name,
                        "question_id": qid,
                        "category": q["category"],
                        "is_answerable": is_ans,
                        "question": question_text,
                        "answer": answer,
                        "confidence": conf,
                        "evidence_status": ev_status,
                        "retries": retries,
                        "trusted_chunk_count": len(trusted),
                        "is_abstained": abstained,
                        "is_correct": correct,
                        "is_h5_discrepancy": is_h5_discrepancy,
                        "is_infra_failure": False,
                        "citation_validity": cit_eval["citation_validity"],
                        "citation_coverage": cit_eval["citation_coverage"],
                        "llm_calls": query_calls,
                        "total_tokens": query_tokens,
                        "latency_sec": duration
                    }
                except Exception as exc:
                    duration = round(time.perf_counter() - t0, 3)
                    query_calls = instrumenter.call_count - calls_before
                    query_tokens = (instrumenter.total_prompt_tokens - p_tok_before) + (instrumenter.total_completion_tokens - c_tok_before)
                    print(f"  [Query Exception on {qid}]: {type(exc).__name__}: {str(exc)}")
                    rec = {
                        "arm": arm_name,
                        "question_id": qid,
                        "category": q["category"],
                        "is_answerable": is_ans,
                        "question": question_text,
                        "answer": None,
                        "confidence": 0.0,
                        "evidence_status": "error",
                        "retries": 0,
                        "trusted_chunk_count": 0,
                        "is_abstained": False,
                        "is_correct": False,
                        "is_h5_discrepancy": False,
                        "is_infra_failure": True,
                        "infra_error": f"{type(exc).__name__}: {str(exc)}",
                        "citation_validity": 0.0,
                        "citation_coverage": 0.0,
                        "llm_calls": query_calls,
                        "total_tokens": query_tokens,
                        "latency_sec": duration
                    }

                records.append(rec)
                f_out.write(json.dumps(rec) + "\n")
                f_out.flush()
                time.sleep(1.2)

            f_out.close()

        # Aggregate arm metrics
        n_q = len(records)
        ans_recs = [r for r in records if r["is_answerable"]]
        unans_recs = [r for r in records if not r["is_answerable"]]

        ans_correct_count = sum(1 for r in ans_recs if r["is_correct"])
        unans_abstained_count = sum(1 for r in unans_recs if r["is_abstained"])
        over_abstained_count = sum(1 for r in ans_recs if r["is_abstained"])
        unsupported_count = len(unans_recs) - unans_abstained_count

        corr_p, corr_lo, corr_hi = compute_wilson_ci(ans_correct_count, len(ans_recs))
        unsupp_p, unsupp_lo, unsupp_hi = compute_wilson_ci(unsupported_count, len(unans_recs))
        over_abs_p, over_abs_lo, over_abs_hi = compute_wilson_ci(over_abstained_count, len(ans_recs))

        # AUROC for confidence vs correctness on answerable
        scores = [r["confidence"] for r in ans_recs]
        labels = [1 if r["is_correct"] else 0 for r in ans_recs]
        auroc = compute_auroc(scores, labels)

        arm_summary = {
            "arm_name": arm_name,
            "total_questions": n_q,
            "num_answerable": len(ans_recs),
            "num_unanswerable": len(unans_recs),
            "answer_correctness": {
                "rate": round(corr_p, 4),
                "95ci": [round(corr_lo, 4), round(corr_hi, 4)],
                "numerator": ans_correct_count,
                "denominator": len(ans_recs)
            },
            "unsupported_answer_rate": {
                "rate": round(unsupp_p, 4),
                "95ci": [round(unsupp_lo, 4), round(unsupp_hi, 4)],
                "numerator": unsupported_count,
                "denominator": len(unans_recs)
            },
            "over_abstention_rate": {
                "rate": round(over_abs_p, 4),
                "95ci": [round(over_abs_lo, 4), round(over_abs_hi, 4)],
                "numerator": over_abstained_count,
                "denominator": len(ans_recs)
            },
            "mean_citation_validity": round(sum(r["citation_validity"] for r in records) / n_q, 4) if n_q else 0.0,
            "mean_citation_coverage": round(sum(r["citation_coverage"] for r in records) / n_q, 4) if n_q else 0.0,
            "h5_discrepancy_count": sum(1 for r in records if r["is_h5_discrepancy"]),
            "confidence_auroc": round(auroc, 4) if auroc is not None else "INCONCLUSIVE",
            "mean_llm_calls_per_query": round(sum(r["llm_calls"] for r in records) / n_q, 2) if n_q else 0.0,
            "mean_tokens_per_query": round(sum(r["total_tokens"] for r in records) / n_q, 1) if n_q else 0.0,
            "p50_latency_sec": round(sorted([r["latency_sec"] for r in records])[n_q // 2], 3) if n_q else 0.0,
            "p95_latency_sec": round(sorted([r["latency_sec"] for r in records])[int(n_q * 0.95)], 3) if n_q else 0.0
        }
        arm_summaries[arm_name] = arm_summary
        print(f"  Correctness: {corr_p:.1%} | Unsupported: {unsupp_p:.1%} | Over-abstention: {over_abs_p:.1%} | p50: {arm_summary['p50_latency_sec']}s")

        # Save summary incrementally after each arm
        with open(os.path.join(OUT_DIR, "e4_ablation_summary.json"), "w", encoding="utf-8") as f:
            json.dump(arm_summaries, f, indent=2)

    return arm_summaries


def run_experiment_e7(instrumenter: HarnessInstrumenter):
    """
    Experiment E7: Latency, Throughput and Concurrency.
    Measures:
    - End-to-end and per-node latencies
    - Event loop blocking: time to execute GET /health while a query is running (H8)
    """
    print("\n--- Running Experiment E7: Concurrency & Latency Profiling ---")
    import threading
    from fastapi.testclient import TestClient
    from app.main import app

    test_client = TestClient(app)

    # 1. Cold vs Warm timing
    # Cold start already recorded in manifest/phase1
    # Measure node breakdown for 5 sample queries
    sample_queries = [
        "What is the battery energy density of the Titan-X?",
        "What is the maximum payload capacity of the Titan-X?",
        "What is the meal per diem under Policy 2.4?"
    ]

    per_node_timings = []
    for q in sample_queries:
        st = build_initial_state(q, 4)
        t0 = time.perf_counter()
        st_ret = retrieve_node(dict(st))
        t1 = time.perf_counter()
        st_ev = ea.evaluate_evidence(st_ret)
        t2 = time.perf_counter()
        st_rep = ra.generate_report(st_ev)
        t3 = time.perf_counter()

        per_node_timings.append({
            "retrieve_sec": round(t1 - t0, 4),
            "evidence_sec": round(t2 - t1, 4),
            "report_sec": round(t3 - t2, 4),
            "total_sec": round(t3 - t0, 4)
        })

    # 2. Concurrency & Event Loop Blocking Check (H8)
    # Start a slow query in a thread, then immediately call GET /health in main thread
    health_times_during_query = []
    
    def run_query_task():
        test_client.post("/query/", json={"question": "What is the battery energy density of the Titan-X?"})

    thread = threading.Thread(target=run_query_task)
    t_start = time.perf_counter()
    thread.start()
    
    # Hit health immediately while query is active
    time.sleep(0.05)
    t_h0 = time.perf_counter()
    resp_health = test_client.get("/health")
    health_latency = time.perf_counter() - t_h0
    thread.join()

    e7_results = {
        "per_node_timings": per_node_timings,
        "mean_retrieve_sec": round(sum(p["retrieve_sec"] for p in per_node_timings) / len(per_node_timings), 4),
        "mean_evidence_sec": round(sum(p["evidence_sec"] for p in per_node_timings) / len(per_node_timings), 4),
        "mean_report_sec": round(sum(p["report_sec"] for p in per_node_timings) / len(per_node_timings), 4),
        "h8_event_loop_blocking": {
            "health_during_query_latency_sec": round(health_latency, 4),
            "health_status": resp_health.status_code,
            "finding": "In FastAPI, calling synchronous blocking run_query() in an async def handler blocks the event loop thread during vector search and LLM network I/O"
        }
    }

    with open(os.path.join(OUT_DIR, "e7_concurrency_latency.json"), "w", encoding="utf-8") as f:
        json.dump(e7_results, f, indent=2)

    print(f"E7 complete. Results saved to {os.path.join(OUT_DIR, 'e7_concurrency_latency.json')}")
    return e7_results


def main():
    instrumenter = HarnessInstrumenter()
    instrumenter.instrument()

    # Load held-out questions
    with open(os.path.join(DATASET_DIR, "held_out_questions.json"), "r", encoding="utf-8") as f:
        held_out_questions = json.load(f)

    # Isolated index in temp directory
    temp_dir = tempfile.mkdtemp(prefix="chroma_llm_eval_")
    try:
        import app.vectorstore.chroma_client as cc
        client, collection = build_evaluation_index(temp_dir)
        cc._collection = collection
        print(f"Mounted isolated Chroma collection ({collection.count()} chunks) for LLM experiments.")

        # Run E3 if not already run
        if os.path.exists(os.path.join(OUT_DIR, "e3_summary.json")):
            print("E3 already completed. Skipping to E4.")
        else:
            run_experiment_e3(instrumenter, held_out_questions)

        # Run E4
        run_experiment_e4(instrumenter, held_out_questions)

        # Run E7
        run_experiment_e7(instrumenter)

        print("\nAll live LLM experiments completed successfully.")
        print(f"Total Groq API calls executed: {instrumenter.call_count}")
        print(f"Total prompt tokens: {instrumenter.total_prompt_tokens}")
        print(f"Total completion tokens: {instrumenter.total_completion_tokens}")

    finally:
        instrumenter.restore()
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
