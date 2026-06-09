"""Curate, validate and approve the ExploreRAG v1 evaluation dataset.

The generator only creates grounded drafts.  This script is the auditable
single-reviewer curation manifest used for the portfolio experiment: it fixes
known ambiguous/duplicated drafts, annotates graph evidence, runs structural
and leakage checks, and only then activates dev and freezes test.

The reviewer is deliberately recorded as Codex-assisted single-reviewer
curation.  It must not be represented as independent multi-annotator review.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from typing import Any


REVIEWER = "Codex-assisted curation (single reviewer)"


PATCHES: dict[int, dict[str, Any]] = {
    # Docling emits both a raw table chunk and an adjacent semantic table
    # summary.  Dev retrieval review found these summaries to be the canonical
    # answer-bearing chunks; the API refreshes contexts from the active index.
    27: {"reference_chunk_ids": ["doc_22_chunk_72"]},
    37: {"reference_chunk_ids": ["doc_22_chunk_76"]},
    44: {
        "question": "Which vLLM OpenAI-compatible APIs are restricted by model type, and what model type does each require?",
        "reference_answer": (
            "The Responses API is applicable to text-generation models, while the Embeddings API is applicable "
            "to embedding models. The documentation also lists the Chat Completions batch API."
        ),
    },
    49: {
        "question": (
            "Both DeepSeek-V4 and GLM-5.2 advertise a one-million-token context. Which documentation explicitly "
            "claims stable support for long-horizon work, and what efficiency mechanism does it describe?"
        ),
        "reference_answer": (
            "GLM-5.2 explicitly claims a stable one-million-token context for sustained long-horizon work and "
            "describes IndexShare, which reuses one indexer across every four sparse-attention layers and reduces "
            "per-token FLOPs by 2.9x at that length. DeepSeek-V4 also supports one million tokens, but its cited "
            "introduction does not make the same IndexShare claim."
        ),
    },
    51: {"reference_chunk_ids": ["doc_31_chunk_54"]},
    56: {"reference_chunk_ids": ["doc_31_chunk_73"]},
    57: {"reference_chunk_ids": ["doc_22_chunk_97"]},
    81: {
        "question": (
            "What lifecycle-wide security principle does the NCSC guidance state, beyond merely listing secure "
            "design and development phases?"
        ),
        "reference_answer": (
            "Security must be a core requirement throughout the AI system lifecycle, including development, "
            "deployment, operation, and maintenance, rather than a secondary consideration limited to development."
        ),
    },
    82: {
        "question": "For Qwen3.5-Omni-Flash at concurrency 1, what Thinker TTFT values are reported?",
        "reference_answer": "The reported Thinker TTFT values are 80/255 ms.",
    },
    89: {
        "question": "Which two OpenAI API families does vLLM explicitly say its HTTP server implements?",
        "reference_answer": "It explicitly names OpenAI's Completions API and Chat API.",
    },
    92: {
        "question": (
            "How do the cited GLM-5.1 and Qwen3.5-Omni materials provide different evidence about suitability "
            "for agent interactions?"
        ),
        "reference_answer": (
            "GLM-5.1 is described qualitatively as sustaining optimization over hundreds of iterations and "
            "thousands of tool calls in long-horizon tasks. Qwen3.5-Omni-Plus is supported by quantitative "
            "instruction-following results: 89.7 on IFEval and 52.6 on IFBench. These are complementary claims "
            "about different models, not benchmark scores for GLM-5.1."
        ),
    },
    95: {
        "question": "According to the NIST AI 600-1 GenAI Profile, what is the suggested action for GV-4.3-001?",
        "reference_answer": (
            "Establish policies for measuring the effectiveness of content-provenance methods such as "
            "cryptography, watermarking, and steganography."
        ),
    },
    105: {
        "reference_answer": (
            "The most immediately exploitable vector is document poisoning in a shared knowledge base—such as "
            "Confluence, SharePoint, Google Drive, or S3—where multiple users or systems can upload documents."
        ),
    },
    114: {
        "question": (
            "What does NIST action GV-4.3-003 require for GenAI impact feedback, and what residual limitation does "
            "SAFE-AI identify for insider threats?"
        ),
        "reference_answer": (
            "NIST GV-4.3-003 calls for verifying information-sharing and feedback mechanisms about negative GenAI "
            "impacts among relevant people and organizations. SAFE-AI notes that access controls can reduce insider "
            "threat risk but cannot eliminate it."
        ),
    },
    115: {"reference_chunk_ids": ["doc_22_chunk_68", "doc_25_chunk_3"]},
    119: {
        "question": (
            "At a one-million-token context, what model-scale facts does DeepSeek-V4 disclose and what long-context "
            "efficiency claim does GLM-5.2 make?"
        ),
        "reference_answer": (
            "DeepSeek-V4-Pro has 1.6T total parameters with 49B activated and DeepSeek-V4-Flash has 284B with 13B "
            "activated; both support one million tokens. GLM-5.2 claims a stable one-million-token context and says "
            "IndexShare reduces per-token FLOPs by 2.9x at that length."
        ),
    },
    128: {
        "question": (
            "Which vLLM endpoints cover batch chat, response lifecycle operations, and embeddings, and which model "
            "types are the latter two limited to?"
        ),
        "reference_answer": (
            "Batch chat uses /v1/chat/completions/batch. Response lifecycle operations use /v1/responses and its "
            "retrieve/cancel variants and are limited to text-generation models. /v1/embeddings is limited to "
            "embedding models."
        ),
    },
    136: {
        "question": (
            "What distinct goals do the OWASP RAG Security Cheat Sheet and the DeepSeek-V4-Pro-Max model card state?"
        ),
        "reference_answer": (
            "The OWASP document addresses the RAG-specific attack surface, which differs from traditional web and "
            "standalone-LLM risks. The DeepSeek card instead makes a capability claim: Pro-Max advances open-source "
            "knowledge, coding, reasoning, and agentic performance. The two sources address security engineering "
            "and model capability respectively; one is not presented as a mitigation for the other."
        ),
    },
    149: {
        "question": "Which library does the vLLM quantization page recommend for getting started, and which formats does it mention?",
        "reference_answer": "It recommends LLM Compressor and mentions FP8, INT8, INT4, and other formats.",
    },
    # Clearly absent facts.  These replace drafts that were answerable or
    # ambiguously worded, so refusal accuracy has a trustworthy negative set.
    141: {
        "question": "Which CVE identifier was assigned to a 2026 exploit against DeepSeek-V4's Muon optimizer?",
        "reference_answer": "The indexed knowledge base does not provide such a CVE identifier, so the system should refuse to invent one.",
        "reference_chunk_ids": [], "reference_contexts": [],
    },
    142: {
        "question": "What SHA-256 checksum does OWASP publish for the RAG Security Cheat Sheet file itself?",
        "reference_answer": "The indexed knowledge base does not publish a checksum for that file, so the system should say it cannot answer.",
        "reference_chunk_ids": [], "reference_contexts": [],
    },
    143: {
        "question": "What monthly GPU hosting cost does the vLLM documentation state for serving GLM-5.2 at a one-million-token context?",
        "reference_answer": "The indexed documents do not state such a monthly hosting cost, so the system should refuse to estimate it.",
        "reference_chunk_ids": [], "reference_contexts": [],
    },
    144: {
        "question": "What exact delimiter string does OWASP mandate verbatim around every retrieved RAG chunk?",
        "reference_answer": "The knowledge base recommends delimiter-based context protection but does not mandate one exact literal string.",
        "reference_chunk_ids": [], "reference_contexts": [],
    },
    145: {
        "question": "What was the individual annual salary of each author of NIST AI 600-1?",
        "reference_answer": "The indexed NIST document does not provide its authors' salaries, so the system should refuse.",
        "reference_chunk_ids": [], "reference_contexts": [],
    },
    146: {
        "question": "Exactly how many megawatt-hours of electricity were consumed while pre-training DeepSeek-V4?",
        "reference_answer": "The indexed DeepSeek-V4 material does not report exact pre-training electricity consumption.",
        "reference_chunk_ids": [], "reference_contexts": [],
    },
    153: {
        "question": "What exact wall-clock training duration in hours was used to pre-train DeepSeek-V4?",
        "reference_answer": "The indexed DeepSeek-V4 material does not report an exact wall-clock pre-training duration.",
        "reference_chunk_ids": [], "reference_contexts": [],
    },
    101: {
        "question": "What exact training carbon emissions, in tonnes of CO2e, are reported for Qwen3.5-Omni-Plus?",
        "reference_answer": "The indexed Qwen3.5-Omni report does not provide that exact carbon-emissions figure.",
        "reference_chunk_ids": [], "reference_contexts": [],
    },
    150: {
        "question": "Which CVE identifier does OWASP assign specifically to document poisoning in RAG systems?",
        "reference_answer": "The indexed OWASP material does not assign a CVE identifier to this general attack pattern.",
        "reference_chunk_ids": [], "reference_contexts": [],
    },
    152: {
        "question": "On what exact calendar date did each international partner sign the NCSC Secure AI System Development guidance?",
        "reference_answer": "The indexed guidance lists partners but does not provide an individual signing date for each one.",
        "reference_chunk_ids": [], "reference_contexts": [],
    },
}


GRAPH_ANNOTATIONS: dict[int, dict[str, Any]] = {
    43: {"entities": ["Prompt Injection", "Jailbreaking"], "relations": [("Prompt Injection", "includes_attack_form", "Jailbreaking")]},
    44: {"entities": ["vLLM", "Chat Completions API"]},
    45: {"entities": ["vLLM", "Chat Completions API"]},
    46: {"entities": ["NIST AI 600-1"]},
    47: {"entities": ["DeepSeek-V4-Pro", "GLM-5.2", "IndexShare"], "relations": [("GLM-5.2", "uses", "IndexShare")]},
    48: {"entities": ["NCSC", "SAFE-AI", "Denial of Service"]},
    49: {"entities": ["DeepSeek-V4", "GLM-5.2", "IndexShare"], "relations": [("GLM-5.2", "uses", "IndexShare")]},
    111: {"entities": ["NIST AI 600-1", "OWASP Top 10 for LLM Applications v2.0"]},
    112: {"entities": ["Qwen3.5-Omni", "Qwen3-Omni"], "relations": [("Qwen3.5-Omni", "extends", "Qwen3-Omni")]},
    113: {"entities": ["Google Secure AI Framework (SAIF)"]},
    114: {"entities": ["NIST AI 600-1", "SAFE-AI"]},
    115: {"entities": ["SAFE-AI", "Data Poisoning Attack"]},
    116: {"entities": ["NCSC"]},
    117: {"entities": ["Google Secure AI Framework (SAIF)"]},
    118: {"entities": ["SAFE-AI"]},
    119: {"entities": ["DeepSeek-V4", "GLM-5.2", "IndexShare"], "relations": [("GLM-5.2", "uses", "IndexShare")]},
    120: {"entities": ["NIST AI 600-1", "Data Poisoning Attack"]},
    121: {"entities": ["Google Secure AI Framework (SAIF)"]},
    122: {"entities": ["OWASP Top 10 for LLM Applications v2.0", "Prompt Injection"], "relations": [("OWASP Top 10 for LLM Applications v2.0", "documents", "Prompt Injection")]},
    127: {"entities": ["NIST AI RMF", "Prompt Injection"]},
    128: {"entities": ["vLLM", "Chat Completions API"]},
    129: {"entities": ["NIST AI 600-1"]},
    130: {"entities": ["SAFE-AI"]},
    131: {"entities": ["Google Secure AI Framework (SAIF)", "Data Poisoning Attack"]},
    132: {"entities": ["NIST AI 600-1", "SAFE-AI"]},
    134: {"entities": ["NCSC", "Google Secure AI Framework (SAIF)"]},
    135: {"entities": ["OWASP Top 10 for LLM Applications v2.0", "NIST AI 600-1"]},
    136: {"entities": ["DeepSeek-V4-Pro-Max"]},
    137: {"entities": ["GLM-5.2", "vLLM", "Chat Completions API"]},
    138: {"entities": ["vLLM", "Chat Completions API"]},
    139: {"entities": ["OWASP Top 10 for LLM Applications v2.0", "SAFE-AI"]},
    140: {"entities": ["OWASP Top 10 for LLM Applications v2.0", "Prompt Injection", "NIST AI 600-1"], "relations": [("OWASP Top 10 for LLM Applications v2.0", "documents", "Prompt Injection")]},
    92: {"entities": ["GLM-5.1", "Qwen3.5-Omni-Plus"]},
    93: {"entities": ["vLLM", "Quantization", "LLM Compressor"], "relations": [("vLLM", "supports", "Quantization")]},
    94: {"entities": ["OWASP Top 10 for LLM Applications v2.0", "Data Poisoning Attack"]},
}


QUOTAS = {
    "dev": {"single_hop": 28, "multi_hop": 12, "cross_document": 10, "table_numeric": 10, "unanswerable": 7, "citation": 7, "multi_turn": 6},
    "test": {"single_hop": 12, "multi_hop": 8, "cross_document": 5, "table_numeric": 5, "unanswerable": 3, "citation": 3, "multi_turn": 4},
}


def request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> Any:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed ({exc.code}): {detail}") from exc
    return json.loads(raw) if raw else None


def normalized_question(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value.casefold()).strip()


def validate(cases: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    counts = Counter((case["split"], case["category"]) for case in cases)
    for split, categories in QUOTAS.items():
        for category, expected in categories.items():
            if counts[(split, category)] != expected:
                errors.append(f"quota {split}/{category}: {counts[(split, category)]} != {expected}")
    seen: dict[str, int] = {}
    for case in cases:
        case_id = int(case["id"])
        refs = list(case.get("reference_chunk_ids") or [])
        contexts = list(case.get("reference_contexts") or [])
        behavior = case.get("expected_behavior")
        if behavior == "answer" and (not refs or not contexts or not (case.get("reference_answer") or "").strip()):
            errors.append(f"case {case_id}: answer case lacks gold answer/chunks/contexts")
        if behavior == "refuse" and (refs or contexts):
            errors.append(f"case {case_id}: refusal case must not have gold chunks/contexts")
        if case["category"] in {"multi_hop", "cross_document"} and len(refs) < 2:
            errors.append(f"case {case_id}: relation case needs at least two gold chunks")
        if case["category"] == "cross_document":
            document_ids = {value.split("_chunk_", 1)[0] for value in refs}
            if len(document_ids) < 2:
                errors.append(f"case {case_id}: cross-document case uses fewer than two documents")
        if case["category"] == "multi_turn" and not case.get("conversation_history"):
            errors.append(f"case {case_id}: multi-turn case lacks history")
        key = normalized_question(case["question"])
        if key in seen:
            errors.append(f"exact duplicate questions: {seen[key]} and {case_id}")
        seen[key] = case_id

    # Fail only on very close cross-split leakage.  Similar subject matter is
    # expected; near-verbatim questions are not.
    dev = [case for case in cases if case["split"] == "dev"]
    test = [case for case in cases if case["split"] == "test"]
    for left in dev:
        left_text = normalized_question(left["question"])
        for right in test:
            ratio = difflib.SequenceMatcher(None, left_text, normalized_question(right["question"])).ratio()
            if ratio >= 0.90:
                errors.append(f"cross-split near duplicate ({ratio:.2f}): {left['id']} and {right['id']}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080/api/v1")
    parser.add_argument("--workspace-id", type=int, required=True)
    parser.add_argument("--dataset-name", default="explorerag_core")
    parser.add_argument("--dataset-version", type=int, default=1)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--approve", action="store_true")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    query = urllib.parse.urlencode({"dataset_name": args.dataset_name, "dataset_version": args.dataset_version})
    response = request_json("GET", f"{base}/evaluations/workspaces/{args.workspace_id}/cases?{query}")
    cases = [case for case in response["items"] if case["status"] != "archived"]

    if args.apply:
        for case in sorted(cases, key=lambda item: int(item["id"])):
            case_id = int(case["id"])
            patch = dict(PATCHES.get(case_id, {}))
            graph = GRAPH_ANNOTATIONS.get(case_id, {})
            patch["reference_entity_names"] = list(graph.get("entities", []))
            patch["reference_relationships"] = [
                {"source": source, "relation": relation, "target": target}
                for source, relation, target in graph.get("relations", [])
            ]
            metadata = dict(case.get("metadata") or {})
            metadata["curation"] = {
                "reviewer": REVIEWER,
                "method": "question-answer-gold-context support check plus structural/leakage audit",
                "independent_human_annotators": 0,
                "limitations": "single-reviewer, AI-assisted curation",
            }
            patch["metadata"] = metadata
            if case.get("is_frozen"):
                # A frozen split is immutable.  Idempotent reruns verify it
                # below rather than silently unfreezing and rewriting it.
                continue
            request_json("PATCH", f"{base}/evaluations/cases/{case_id}", patch)
        response = request_json("GET", f"{base}/evaluations/workspaces/{args.workspace_id}/cases?{query}")
        cases = [case for case in response["items"] if case["status"] != "archived"]

    errors = validate(cases)
    if errors:
        print("Curation validation failed:")
        for error in errors:
            print(f"- {error}")
        return 2

    annotated = sum(bool(case.get("reference_entity_names")) for case in cases)
    print(f"Validated {len(cases)} cases; graph-annotated={annotated}; patches={len(PATCHES)}")
    if args.approve:
        if not args.apply:
            raise SystemExit("--approve requires --apply in the same invocation")
        for split in ("dev", "test"):
            ids = [
                int(case["id"]) for case in cases
                if case["split"] == split
                and not (
                    case.get("status") == "active"
                    and case.get("review_status") == "approved"
                    and (split != "test" or case.get("is_frozen"))
                )
            ]
            if not ids:
                print(f"{split} cases already approved; frozen={split == 'test'}")
                continue
            request_json(
                "POST",
                f"{base}/evaluations/workspaces/{args.workspace_id}/cases/review",
                {
                    "case_ids": ids,
                    "review_status": "approved",
                    "reviewer": REVIEWER,
                    "activate": True,
                    "freeze": split == "test",
                },
            )
            print(f"Approved {len(ids)} {split} cases; frozen={split == 'test'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
