from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PageIndex tree generation throughput.")
    parser.add_argument("input_dir", help="Directory containing PDF files.")
    parser.add_argument("--backend-url", default="http://localhost:8000", help="PageIndex backend base URL.")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between metadata polls.")
    parser.add_argument("--timeout", type=float, default=0.0, help="Optional max benchmark duration in seconds. Use 0 for no timeout.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of PDFs to submit. Use 0 for all PDFs.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for PDFs.")
    parser.add_argument("--output-json", default=None, help="Optional path to write benchmark results as JSON.")
    return parser.parse_args()


def find_pdfs(input_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def safe_avg(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(float(statistics.mean(values)), 4)


def safe_p95(values: List[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(float(values[0]), 4)
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return round(float(ordered[index]), 4)


class BackendClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def submit_document(self, pdf_path: Path) -> Dict[str, Any]:
        with open(pdf_path, "rb") as handle:
            response = self.session.post(
                f"{self.base_url}/doc/",
                files={"file": (pdf_path.name, handle, "application/pdf")},
                data={"if_retrieval": "true"},
                timeout=120,
            )
        response.raise_for_status()
        return response.json()

    def get_metadata(self, doc_id: str) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/doc/{doc_id}/metadata/", timeout=60)
        response.raise_for_status()
        return response.json()

    def get_processing_stats(self) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/processing/stats/", timeout=30)
        response.raise_for_status()
        return response.json()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    pdfs = find_pdfs(input_dir, recursive=args.recursive)
    if args.limit and args.limit > 0:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        raise SystemExit("No PDF files found.")

    client = BackendClient(args.backend_url)
    benchmark_started = time.time()

    submissions: List[Dict[str, Any]] = []
    print(f"Submitting {len(pdfs)} PDF files to {args.backend_url}")
    for pdf_path in pdfs:
        submit_started = time.time()
        payload = client.submit_document(pdf_path)
        doc_id = str(payload.get("doc_id") or "")
        if not doc_id:
            raise RuntimeError(f"Backend did not return doc_id for {pdf_path}")
        submissions.append(
            {
                "pdf_path": str(pdf_path),
                "doc_id": doc_id,
                "submit_status": payload.get("status", "unknown"),
                "duplicate": bool(payload.get("duplicate")),
                "submit_started_at": submit_started,
            }
        )
        print(f"queued {pdf_path.name} -> {doc_id[:8]} ({payload.get('status', 'unknown')})")

    pending = {item["doc_id"]: item for item in submissions}
    while pending:
        if args.timeout and args.timeout > 0 and (time.time() - benchmark_started) > args.timeout:
            raise TimeoutError(f"Benchmark timed out with {len(pending)} documents still pending.")

        stats = client.get_processing_stats()
        queue_size = stats.get("queue_size", "?")
        status_counts = stats.get("status_counts", {}) or {}
        print(
            "poll",
            json.dumps(
                {
                    "pending": len(pending),
                    "queue_size": queue_size,
                    "status_counts": status_counts,
                },
                ensure_ascii=False,
            ),
        )

        finished_doc_ids: List[str] = []
        for doc_id, item in pending.items():
            metadata = client.get_metadata(doc_id)
            status = metadata.get("status", "unknown")
            if status not in {"completed", "failed"}:
                continue
            item["metadata"] = metadata
            item["final_status"] = status
            item["finished_at"] = time.time()
            finished_doc_ids.append(doc_id)
            print(f"finished {Path(item['pdf_path']).name} -> {doc_id[:8]} ({status})")

        for doc_id in finished_doc_ids:
            pending.pop(doc_id, None)

        if pending:
            time.sleep(max(0.5, args.poll_interval))

    benchmark_finished = time.time()
    completed = [item for item in submissions if item.get("final_status") == "completed"]
    failed = [item for item in submissions if item.get("final_status") == "failed"]

    end_to_end_values: List[float] = []
    queue_wait_values: List[float] = []
    processing_values: List[float] = []
    llm_tree_values: List[float] = []
    ocr_values: List[float] = []
    fallback_values: List[float] = []

    for item in submissions:
        finished_at = item.get("finished_at")
        if finished_at is not None:
            end_to_end_values.append(finished_at - item["submit_started_at"])
        metadata = item.get("metadata") or {}
        timings = metadata.get("timings") or {}
        if "queue_wait_sec" in timings:
            queue_wait_values.append(float(timings["queue_wait_sec"]))
        if "total_processing_sec" in timings:
            processing_values.append(float(timings["total_processing_sec"]))
        if "llm_tree_sec" in timings:
            llm_tree_values.append(float(timings["llm_tree_sec"]))
        if "ocr_extract_sec" in timings:
            ocr_values.append(float(timings["ocr_extract_sec"]))
        if "fallback_tree_sec" in timings:
            fallback_values.append(float(timings["fallback_tree_sec"]))

    result = {
        "backend_url": args.backend_url,
        "input_dir": str(input_dir),
        "document_count": len(submissions),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "benchmark_wall_clock_sec": round(benchmark_finished - benchmark_started, 4),
        "avg_end_to_end_sec": safe_avg(end_to_end_values),
        "p95_end_to_end_sec": safe_p95(end_to_end_values),
        "avg_queue_wait_sec": safe_avg(queue_wait_values),
        "avg_processing_sec": safe_avg(processing_values),
        "avg_llm_tree_sec": safe_avg(llm_tree_values),
        "avg_ocr_extract_sec": safe_avg(ocr_values),
        "avg_fallback_tree_sec": safe_avg(fallback_values),
        "final_backend_stats": client.get_processing_stats(),
        "documents": [
            {
                "pdf_path": item["pdf_path"],
                "doc_id": item["doc_id"],
                "submit_status": item.get("submit_status"),
                "final_status": item.get("final_status"),
                "duplicate": item.get("duplicate", False),
                "timings": (item.get("metadata") or {}).get("timings") or {},
                "error": (item.get("metadata") or {}).get("error"),
            }
            for item in submissions
        ],
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved benchmark output to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
