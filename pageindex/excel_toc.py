from __future__ import annotations

import json
import tempfile
from typing import Any, Dict, List


def create_excel_toc_input_file(ocr_pages: List[Dict[str, Any]]) -> str:
    """Create a temporary JSON file containing virtual pages for PageIndex LLM TOC generation."""
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    try:
        json.dump({"pages": ocr_pages}, temp_file, ensure_ascii=False)
        temp_file.flush()
        return temp_file.name
    finally:
        temp_file.close()
