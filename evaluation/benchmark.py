"""
Evaluation for the GraphRAG system.

Compares three ways of answering the same questions:
  1. GraphRAG      - graph traversal + vector retrieval + Gemini
  2. Standard RAG  - vector retrieval only + Gemini
  3. Plain LLM     - Gemini with no retrieval

Each answer is scored 0-1 by an LLM judge against a ground-truth answer.

Run:  python -m evaluation.benchmark            (uses evaluation/questions.json)
      python -m evaluation.benchmark --limit 3  (quick smoke test)

Notes
  * The judge is the same model family as the answerer, so scores are indicative,
    not a rigorous benchmark.
  * Each question costs ~6 Gemini calls (3 answers + 3 judgements). Calls are
    throttled and retried so the free tier's rate limits do not abort the run.
"""
import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import google.generativeai as genai

from config.logger import log
from config.settings import settings
from generation.answer_generator import AnswerGenerator
from reasoning.multihop import MultiHopReasoner
from vector.chroma_client import ChromaClient

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


def fact_recall(answer: str, key_facts: List[str]) -> float:
    """Fraction of required facts (names, amounts) that appear in the answer.

    Deterministic and transparent: no LLM involved, so it is not subject to judge noise.
    Case-insensitive; "$" and "," are ignored; numbers must not be part of a larger
    number (so "2 billion" does not match "$20 billion").
    """
    if not key_facts:
        return 0.0
    text = answer.lower().replace("$", "").replace(",", "")
    hits = 0
    for fact in key_facts:
        f = fact.lower().replace("$", "").replace(",", "")
        prefix = r"(?<![\d.])" if f[:1].isdigit() else r"(?<!\w)"
        if re.search(prefix + re.escape(f) + r"(?!\w)", text):
            hits += 1
    return hits / len(key_facts)


@dataclass
class EvaluationResult:
    """Evaluation result for a single question"""
    question: str
    ground_truth: str
    hops: str
    graphrag_answer: str
    standard_rag_answer: str
    plain_llm_answer: str
    graphrag_score: float
    standard_rag_score: float
    plain_llm_score: float
    key_facts: List[str] = field(default_factory=list)
    graphrag_recall: float = 0.0
    standard_rag_recall: float = 0.0
    plain_llm_recall: float = 0.0


class GraphRAGEvaluator:
    """Evaluate GraphRAG against standard RAG and a plain LLM"""

    def __init__(self, delay_seconds: float = 4.0):
        self.reasoner = MultiHopReasoner()
        self.generator = AnswerGenerator()
        self.chroma_client = ChromaClient()
        self.delay = delay_seconds

        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)
        log.info("GraphRAGEvaluator initialized")

    # ------------------------------------------------------------------ helpers
    def _generate(self, prompt: str, retries: int = 4) -> str:
        """Call Gemini with a pause before each call and retry on rate limits."""
        for attempt in range(retries):
            time.sleep(self.delay)
            try:
                return self.model.generate_content(prompt).text.strip()
            except Exception as e:  # rate limit / transient error
                wait = 20 * (attempt + 1)
                log.warning(f"Gemini call failed ({str(e)[:80]}); retrying in {wait}s")
                time.sleep(wait)
        raise RuntimeError("Gemini call failed after retries")

    # ----------------------------------------------------------------- answerers
    def graphrag_answer(self, question: str) -> str:
        """Full pipeline: graph + vector retrieval."""
        context = self.reasoner.retrieve_context(question)
        formatted = self.reasoner.format_context_for_llm(context)
        time.sleep(self.delay)
        return self.generator.generate_query_result(question, context, formatted).answer

    def standard_rag_answer(self, question: str) -> str:
        """Vector retrieval only (no graph)."""
        chunks = self.chroma_client.query(question, n_results=5)
        context = "".join(f"Document {i}: {d}\n\n" for i, d in enumerate(chunks["documents"], 1))
        return self._generate(
            "Answer the following question using only the provided documents.\n\n"
            f"Question: {question}\n\n{context}\nAnswer:"
        )

    def plain_llm_answer(self, question: str) -> str:
        """No retrieval: the model's own knowledge only."""
        return self._generate(f"Answer the following question based on your knowledge.\n\nQuestion: {question}\n\nAnswer:")

    # -------------------------------------------------------------------- judge
    def llm_as_judge_score(self, question: str, answer: str, ground_truth: str) -> float:
        """Score an answer 0-1 against the ground truth."""
        prompt = f"""You are a strict evaluator. Score how well the answer agrees with the ground truth.

IMPORTANT: The ground truth comes from a fixed document collection and is authoritative.
Do NOT use your own world knowledge to decide what is true. Judge ONLY whether the answer
states the same facts as the ground truth (completeness and correctness against it).
An answer that is plausible in the real world but does not match the ground truth is wrong.

Question: {question}
Ground Truth: {ground_truth}
Answer to Evaluate: {answer}

Scale: 1.0 = states all the ground-truth facts; 0.7-0.9 = most of them; 0.4-0.6 = some;
0.0-0.3 = contradicts, misses the key facts, or says it does not know.
Return ONLY a decimal number such as 0.85"""
        try:
            match = re.search(r"\d(?:\.\d+)?", self._generate(prompt))
            return max(0.0, min(1.0, float(match.group()))) if match else 0.0
        except Exception as e:
            log.warning(f"Judge failed: {e}")
            return 0.0

    # ----------------------------------------------------------------- evaluate
    def evaluate_question(self, item: Dict[str, str]) -> EvaluationResult:
        q, gt = item["question"], item["ground_truth"]
        log.info(f"Evaluating: {q}")
        g, s, p = self.graphrag_answer(q), self.standard_rag_answer(q), self.plain_llm_answer(q)
        result = EvaluationResult(
            question=q, ground_truth=gt, hops=item.get("hops", "multi"),
            graphrag_answer=g, standard_rag_answer=s, plain_llm_answer=p,
            graphrag_score=self.llm_as_judge_score(q, g, gt),
            standard_rag_score=self.llm_as_judge_score(q, s, gt),
            plain_llm_score=self.llm_as_judge_score(q, p, gt),
            key_facts=item.get("key_facts", []),
            graphrag_recall=fact_recall(g, item.get("key_facts", [])),
            standard_rag_recall=fact_recall(s, item.get("key_facts", [])),
            plain_llm_recall=fact_recall(p, item.get("key_facts", [])),
        )
        log.info(f"Scores - GraphRAG {result.graphrag_score:.2f} | Standard RAG "
                 f"{result.standard_rag_score:.2f} | Plain LLM {result.plain_llm_score:.2f}")
        return result

    @staticmethod
    def _avg(results: List[EvaluationResult], attr: str) -> float:
        return sum(getattr(r, attr) for r in results) / len(results) if results else 0.0

    def summarize(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        def block(rs: List[EvaluationResult]) -> Dict[str, Any]:
            return {
                "questions": len(rs),
                "fact_recall": {
                    "graphrag": round(self._avg(rs, "graphrag_recall"), 3),
                    "standard_rag": round(self._avg(rs, "standard_rag_recall"), 3),
                    "plain_llm": round(self._avg(rs, "plain_llm_recall"), 3),
                },
                "llm_judge": {
                    "graphrag": round(self._avg(rs, "graphrag_score"), 3),
                    "standard_rag": round(self._avg(rs, "standard_rag_score"), 3),
                    "plain_llm": round(self._avg(rs, "plain_llm_score"), 3),
                },
            }
        return {
            "model": settings.gemini_model,
            "overall": block(results),
            "multi_hop": block([r for r in results if r.hops == "multi"]),
            "single_hop": block([r for r in results if r.hops == "single"]),
        }

    def evaluate_dataset(self, questions: List[Dict[str, str]]) -> Dict[str, Any]:
        results: List[EvaluationResult] = []
        for item in questions:
            try:
                results.append(self.evaluate_question(item))
                self.save_results(results)  # save as we go so a quota error loses nothing
            except Exception as e:
                log.error(f"Skipping question after error: {e}")
        return {"summary": self.summarize(results), "results": results}

    def save_results(self, results: List[EvaluationResult], path: Path = RESULTS_PATH):
        output = {
            "summary": self.summarize(results),
            "results": [
                {
                    "question": r.question, "hops": r.hops, "ground_truth": r.ground_truth,
                    "key_facts": r.key_facts,
                    "scores": {"graphrag": r.graphrag_score, "standard_rag": r.standard_rag_score,
                               "plain_llm": r.plain_llm_score},
                    "fact_recall": {"graphrag": r.graphrag_recall, "standard_rag": r.standard_rag_recall,
                                    "plain_llm": r.plain_llm_recall},
                    "answers": {"graphrag": r.graphrag_answer, "standard_rag": r.standard_rag_answer,
                                "plain_llm": r.plain_llm_answer},
                }
                for r in results
            ],
        }
        path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    def close(self):
        for obj in (self.reasoner, self.chroma_client):
            try:
                obj.close()
            except Exception:
                pass


def print_summary(s: Dict[str, Any]):
    print("\n" + "=" * 62 + f"\nEVALUATION  (model: {s['model']})\n" + "=" * 62)
    for metric, title in (("fact_recall", "KEY-FACT RECALL (deterministic, no LLM)"),
                          ("llm_judge", "LLM-JUDGE SCORE (noisy, same model family)")):
        print(f"\n{title}")
        print(f"{'':12}{'GraphRAG':>10}{'Std RAG':>10}{'Plain LLM':>11}{'n':>4}")
        for name in ("overall", "multi_hop", "single_hop"):
            b = s[name]
            m = b[metric]
            print(f"{name:12}{m['graphrag']:>10.2f}{m['standard_rag']:>10.2f}{m['plain_llm']:>11.2f}{b['questions']:>4}")


def rescore():
    """Recompute key-fact recall from saved answers (no API calls)."""
    saved = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    facts = {q["question"]: q.get("key_facts", []) for q in json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))}
    results = []
    for r in saved["results"]:
        kf = facts.get(r["question"], [])
        a, sc = r["answers"], r["scores"]
        results.append(EvaluationResult(
            question=r["question"], ground_truth=r["ground_truth"], hops=r["hops"],
            graphrag_answer=a["graphrag"], standard_rag_answer=a["standard_rag"], plain_llm_answer=a["plain_llm"],
            graphrag_score=sc["graphrag"], standard_rag_score=sc["standard_rag"], plain_llm_score=sc["plain_llm"],
            key_facts=kf,
            graphrag_recall=fact_recall(a["graphrag"], kf),
            standard_rag_recall=fact_recall(a["standard_rag"], kf),
            plain_llm_recall=fact_recall(a["plain_llm"], kf),
        ))
    # summarize/save need no API access; bypass __init__ (which connects to databases)
    ev = GraphRAGEvaluator.__new__(GraphRAGEvaluator)
    ev.save_results(results)
    print_summary(ev.summarize(results))
    print(f"\nRescored {len(results)} saved answers -> {RESULTS_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark GraphRAG vs standard RAG vs plain LLM")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    parser.add_argument("--delay", type=float, default=4.0, help="seconds to pause before each Gemini call")
    parser.add_argument("--rescore", action="store_true", help="recompute fact recall from saved answers (no API calls)")
    args = parser.parse_args()

    if args.rescore:
        rescore()
        return

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    evaluator = GraphRAGEvaluator(delay_seconds=args.delay)
    out = evaluator.evaluate_dataset(questions)
    evaluator.close()
    print_summary(out["summary"])
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
