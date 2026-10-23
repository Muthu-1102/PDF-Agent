"""
Unit tests for deterministic answer matcher, abstention detector, and citation parser.
"""

import unittest
from evaluation.harness.metrics import (
    is_abstained,
    check_answer_correctness,
    parse_citations,
    evaluate_citations,
    compute_wilson_ci,
    compute_auroc
)


class TestMetrics(unittest.TestCase):
    def test_abstention_detection(self):
        # Canned fallback
        canned = "I don't have sufficient verified evidence in the knowledge base to answer this question confidently."
        self.assertTrue(is_abstained(canned))

        # Explicit refusal
        refusal = "The provided documents do not contain information regarding this topic."
        self.assertTrue(is_abstained(refusal))

        # Substantive answer
        substantive = "The Titan-X drone has a payload capacity of 18.4 kilograms [titan.pdf, chunk 1]."
        self.assertFalse(is_abstained(substantive))

        # None or empty
        self.assertTrue(is_abstained(None))
        self.assertTrue(is_abstained(""))

    def test_answer_correctness(self):
        # Answerable: valid facts
        ans = "The Titan-X drone operates on a Lithium-Sulfur battery delivering 420 Wh/kg [specs.pdf, chunk 0]."
        self.assertTrue(check_answer_correctness(ans, ["420 Wh/kg", "Lithium-Sulfur"], [], True))

        # Answerable: missing a required fact
        self.assertFalse(check_answer_correctness(ans, ["420 Wh/kg", "Lithium-Sulfur", "Marcus Vance"], [], True))

        # Answerable: forbidden fact present
        ans_conflict = "The daily per diem is $65 under the current policy."
        self.assertFalse(check_answer_correctness(ans_conflict, ["$65"], ["$65"], True))

        # Unanswerable: abstains correctly
        unans_correct = "I don't have sufficient verified evidence in the knowledge base."
        self.assertTrue(check_answer_correctness(unans_correct, [], [], False))

        # Unanswerable: hallucinates substantive answer
        unans_halluc = "The blue whale gestation period is 12 months."
        self.assertFalse(check_answer_correctness(unans_halluc, [], [], False))

    def test_citation_parsing(self):
        ans = "The allowance is $92 [vrcc_travel_policy_v2.pdf, chunk 0] and hotel is $245 [vrcc_travel_policy_v2.pdf | chunk 1]."
        citations = parse_citations(ans)
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0], ("vrcc_travel_policy_v2.pdf", 0))
        self.assertEqual(citations[1], ("vrcc_travel_policy_v2.pdf", 1))

        trusted = [
            {"doc_name": "vrcc_travel_policy_v2.pdf", "chunk_id": 0},
            {"doc_name": "vrcc_travel_policy_v2.pdf", "chunk_id": 1}
        ]
        eval_res = evaluate_citations(ans, trusted)
        self.assertEqual(eval_res["valid_citations"], 2)
        self.assertEqual(eval_res["citation_validity"], 1.0)

    def test_wilson_ci(self):
        p, lo, hi = compute_wilson_ci(10, 10)
        self.assertEqual(p, 1.0)
        self.assertGreater(lo, 0.70)
        self.assertEqual(hi, 1.0)


if __name__ == "__main__":
    unittest.main()
