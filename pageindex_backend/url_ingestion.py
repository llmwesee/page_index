from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse, urlunparse
import hashlib
import ipaddress
import json
import re
import socket
import time

try:
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    requests = None  # type: ignore

try:
    import trafilatura  # type: ignore
    from trafilatura.feeds import find_feed_urls  # type: ignore
    from trafilatura.sitemaps import sitemap_search  # type: ignore
    from trafilatura.spider import focused_crawler  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    trafilatura = None  # type: ignore
    find_feed_urls = None  # type: ignore
    sitemap_search = None  # type: ignore
    focused_crawler = None  # type: ignore


class URLIngestionError(RuntimeError):
    """Raised when URL ingestion fails validation, discovery, or extraction."""


@dataclass
class URLIngestionResult:
    name: str
    path: Path
    input_type: str
    source_type: str
    source_url: str
    source_url_normalized: str
    resolved_url: str
    source_metadata: Dict[str, Any]


def normalize_source_url(url: str) -> str:
    candidate = (url or "").strip()
    if not candidate:
        raise URLIngestionError("URL is required.")

    parsed = urlparse(candidate)
    if not parsed.scheme:
        parsed = urlparse(f"https://{candidate}")

    if parsed.scheme.lower() not in {"http", "https"}:
        raise URLIngestionError("Only HTTP/HTTPS URLs are supported.")
    if not parsed.netloc:
        raise URLIngestionError("URL host is missing.")
    if parsed.username or parsed.password:
        raise URLIngestionError("URLs with embedded credentials are not allowed.")

    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        fragment="",
    )
    return urlunparse(normalized)


def _is_local_or_private_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _is_blocked_host(hostname: Optional[str]) -> bool:
    host = (hostname or "").strip().lower().rstrip(".")
    if not host:
        return True
    if host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".local"):
        return True
    if _is_local_or_private_ip(host):
        return True

    try:
        records = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return True

    for record in records:
        sockaddr = record[4]
        if not sockaddr:
            continue
        if _is_local_or_private_ip(str(sockaddr[0])):
            return True
    return False


def _safe_name(seed: str, suffix: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", seed).strip("-._")
    if not cleaned:
        cleaned = "url-document"
    if not cleaned.lower().endswith(suffix):
        cleaned = f"{cleaned}{suffix}"
    return cleaned[:180]


def _looks_like_pdf_url(url: str) -> bool:
    parsed = urlparse(url)
    path = (parsed.path or "").lower()
    return path.endswith(".pdf")


def _response_is_pdf(response: Any) -> bool:
    content_type = str(response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if content_type == "application/pdf":
        return True

    content_disposition = str(response.headers.get("Content-Disposition") or "").lower()
    if ".pdf" in content_disposition:
        return True

    return _looks_like_pdf_url(str(response.url or ""))


def _download_pdf_artifact(
    session: "requests.Session",
    url: str,
    data_dir: Path,
    doc_id: str,
    *,
    timeout_sec: int,
    max_download_mb: int,
    source_url: str,
    source_url_normalized: str,
) -> Optional[URLIngestionResult]:
    try:
        response = session.get(url, timeout=timeout_sec, allow_redirects=True, stream=True)
    except Exception as exc:  # noqa: BLE001
        raise URLIngestionError(f"Failed to fetch URL: {exc}") from exc

    with response:
        if response.status_code >= 400:
            raise URLIngestionError(f"URL returned HTTP {response.status_code}.")
        final_url = str(response.url or url)
        parsed_final = urlparse(final_url)
        if _is_blocked_host(parsed_final.hostname):
            raise URLIngestionError("Blocked URL host. Private, localhost, and local-network targets are not allowed.")
        if not _response_is_pdf(response):
            return None

        max_bytes = max(1, int(max_download_mb)) * 1024 * 1024
        downloaded = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            downloaded.extend(chunk)
            if len(downloaded) > max_bytes:
                raise URLIngestionError(
                    f"Downloaded file exceeds max_download_mb={max_download_mb}MB."
                )

    if not downloaded:
        raise URLIngestionError("Downloaded PDF is empty.")

    if not bytes(downloaded[:5]).startswith(b"%PDF-"):
        raise URLIngestionError("URL appears to be a PDF but file signature is invalid.")

    fetched_at = time.time()
    output_path = data_dir / f"{doc_id}.pdf"
    output_path.write_bytes(downloaded)

    resolved_url = normalize_source_url(final_url)
    host_name = urlparse(resolved_url).netloc or "url"
    return URLIngestionResult(
        name=_safe_name(f"web-pdf-{host_name}", ".pdf"),
        path=output_path,
        input_type="pdf",
        source_type="url_pdf",
        source_url=source_url,
        source_url_normalized=source_url_normalized,
        resolved_url=resolved_url,
        source_metadata={
            "ingested_at": fetched_at,
            "download_size_bytes": len(downloaded),
            "content_type": str(response.headers.get("Content-Type") or ""),
            "content_sha256": hashlib.sha256(downloaded).hexdigest(),
        },
    )


def _require_dependencies() -> None:
    if requests is None:
        raise URLIngestionError("The requests package is required for URL ingestion.")
    global trafilatura
    global find_feed_urls
    global sitemap_search
    global focused_crawler
    if trafilatura is None:
        try:
            trafilatura = importlib.import_module("trafilatura")  # type: ignore[assignment]
            find_feed_urls = importlib.import_module("trafilatura.feeds").find_feed_urls  # type: ignore[assignment]
            sitemap_search = importlib.import_module("trafilatura.sitemaps").sitemap_search  # type: ignore[assignment]
            focused_crawler = importlib.import_module("trafilatura.spider").focused_crawler  # type: ignore[assignment]
        except Exception as exc:  # noqa: BLE001
            raise URLIngestionError(
                "The trafilatura package is required for URL discovery and extraction. "
                "Install it with: pip install trafilatura"
            ) from exc


def _collect_discovered_urls(
    seed_url: str,
    *,
    max_urls: int,
    include_crawler: bool,
) -> Tuple[List[str], Dict[str, int]]:
    urls: List[str] = [seed_url]
    source_counts = {"seed": 1, "sitemap": 0, "feed": 0, "crawler": 0}
    seen = {seed_url}

    if sitemap_search is not None:
        with_sitemap = sitemap_search(seed_url)
        for item in with_sitemap:
            try:
                normalized = normalize_source_url(item)
            except URLIngestionError:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            urls.append(normalized)
            source_counts["sitemap"] += 1
            if len(urls) >= max_urls:
                return urls[:max_urls], source_counts

    if find_feed_urls is not None:
        with_feed = find_feed_urls(seed_url)
        for item in with_feed:
            try:
                normalized = normalize_source_url(item)
            except URLIngestionError:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            urls.append(normalized)
            source_counts["feed"] += 1
            if len(urls) >= max_urls:
                return urls[:max_urls], source_counts

    if include_crawler and focused_crawler is not None and len(urls) < max_urls:
        _, known_links = focused_crawler(
            homepage=seed_url,
            max_seen_urls=max_urls,
            max_known_urls=max_urls * 40,
        )
        for item in list(known_links or []):
            try:
                normalized = normalize_source_url(item)
            except URLIngestionError:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            urls.append(normalized)
            source_counts["crawler"] += 1
            if len(urls) >= max_urls:
                break

    return urls[:max_urls], source_counts


def _extract_page(url: str, include_comments: bool, include_tables: bool) -> Optional[Dict[str, Any]]:
    downloaded = trafilatura.fetch_url(url)  # type: ignore[union-attr]
    if not downloaded:
        return None

    extracted = trafilatura.bare_extraction(  # type: ignore[union-attr]
        downloaded,
        url=url,
        output_format="python",
        with_metadata=True,
        include_comments=include_comments,
        include_tables=include_tables,
    )
    if extracted is None:
        return None

    if hasattr(extracted, "as_dict"):
        extracted_dict = extracted.as_dict()
    elif isinstance(extracted, dict):
        extracted_dict = extracted
    else:
        return None

    main_text = str(extracted_dict.get("text") or "").strip()
    comments = str(extracted_dict.get("comments") or "").strip()
    raw_title = str(extracted_dict.get("title") or "").strip()

    table_text = ""
    tables_value = extracted_dict.get("tables")
    if isinstance(tables_value, list):
        table_rows = [str(item).strip() for item in tables_value if str(item).strip()]
        table_text = "\n\n".join(table_rows)
    elif isinstance(tables_value, str):
        table_text = tables_value.strip()

    combined_parts = [main_text]
    if comments:
        combined_parts.append(f"Comments\n{comments}")
    if table_text:
        combined_parts.append(f"Tables\n{table_text}")

    combined_text = "\n\n".join(part for part in combined_parts if part).strip()
    if not combined_text:
        return None

    return {
        "url": url,
        "title": raw_title,
        "text": main_text,
        "comments": comments,
        "tables": table_text,
        "combined_text": combined_text,
        "author": str(extracted_dict.get("author") or "").strip(),
        "date": str(extracted_dict.get("date") or "").strip(),
        "sitename": str(extracted_dict.get("sitename") or "").strip(),
    }


def ingest_url_to_local_artifact(
    url: str,
    data_dir: Path,
    doc_id: str,
    *,
    timeout_sec: int = 20,
    max_download_mb: int = 80,
    user_agent: str = "PageIndex/1.0 (mailto:ops@example.com)",
    discover: bool = False,
    max_urls: int = 25,
    include_comments: bool = True,
    include_tables: bool = True,
    include_crawler: bool = True,
) -> URLIngestionResult:
    _require_dependencies()
    normalized_url = normalize_source_url(url)
    parsed = urlparse(normalized_url)
    if _is_blocked_host(parsed.hostname):
        raise URLIngestionError("Blocked URL host. Private, localhost, and local-network targets are not allowed.")

    data_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()  # type: ignore[union-attr]
    session.headers.update({"User-Agent": user_agent})

    try:
        if (not discover) or _looks_like_pdf_url(normalized_url):
            pdf_result = _download_pdf_artifact(
                session,
                normalized_url,
                data_dir,
                doc_id,
                timeout_sec=timeout_sec,
                max_download_mb=max_download_mb,
                source_url=url,
                source_url_normalized=normalized_url,
            )
            if pdf_result is not None:
                return pdf_result

        fetched_at = time.time()
        if discover:
            discovered_urls, source_counts = _collect_discovered_urls(
                normalized_url,
                max_urls=max(1, int(max_urls)),
                include_crawler=include_crawler,
            )
        else:
            discovered_urls = [normalized_url]
            source_counts = {"seed": 1, "sitemap": 0, "feed": 0, "crawler": 0}

        pages: List[Dict[str, Any]] = []
        extraction_errors: List[Dict[str, str]] = []
        for index, candidate in enumerate(discovered_urls, start=1):
            try:
                if _is_blocked_host(urlparse(candidate).hostname):
                    extraction_errors.append({"url": candidate, "error": "Blocked private/local host"})
                    continue
                page_payload = _extract_page(candidate, include_comments=include_comments, include_tables=include_tables)
                if page_payload is None:
                    extraction_errors.append({"url": candidate, "error": "No extractable content"})
                    continue
            except Exception as exc:  # noqa: BLE001
                extraction_errors.append({"url": candidate, "error": str(exc)})
                continue

            evidence_units: List[Dict[str, Any]] = [
                {
                    "page": index,
                    "content_type": "TEXT",
                    "element_id": f"url_text_{index}",
                    "text": page_payload["text"] or page_payload["combined_text"],
                }
            ]
            if page_payload["comments"]:
                evidence_units.append(
                    {
                        "page": index,
                        "content_type": "TEXT",
                        "element_id": f"url_comments_{index}",
                        "text": page_payload["comments"],
                    }
                )
            if page_payload["tables"]:
                evidence_units.append(
                    {
                        "page": index,
                        "content_type": "TABLE",
                        "element_id": f"url_tables_{index}",
                        "text": page_payload["tables"],
                    }
                )

            pages.append(
                {
                    "page": index,
                    "text": page_payload["combined_text"],
                    "page_metadata": {
                        "source_url": page_payload["url"],
                        "title": page_payload["title"],
                        "author": page_payload["author"],
                        "date": page_payload["date"],
                        "sitename": page_payload["sitename"],
                    },
                    "evidence_units": evidence_units,
                }
            )

        if not pages:
            raise URLIngestionError("No pages could be extracted from discovered URLs.")

        payload = {
            "document": {
                "source_type": "trafilatura_discovery" if discover else "trafilatura_url",
                "source_url": url,
                "source_url_normalized": normalized_url,
                "resolved_url": normalized_url,
                "fetched_at": fetched_at,
                "discovery_enabled": discover,
                "discovered_url_count": len(discovered_urls),
                "extracted_page_count": len(pages),
                "discovery_sources": source_counts,
                "extraction_errors": extraction_errors,
            },
            "pages": pages,
        }

        output_path = data_dir / f"{doc_id}.json"
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        host_name = urlparse(normalized_url).netloc or "url"
        name_seed = f"web-discovery-{host_name}" if discover else f"web-url-{host_name}"
        return URLIngestionResult(
            name=_safe_name(name_seed, ".json"),
            path=output_path,
            input_type="json",
            source_type="url_discovery" if discover else "url_single",
            source_url=url,
            source_url_normalized=normalized_url,
            resolved_url=normalized_url,
            source_metadata={
                "ingested_at": fetched_at,
                "discovered_url_count": len(discovered_urls),
                "extracted_page_count": len(pages),
                "discovery_sources": source_counts,
                "failed_count": len(extraction_errors),
                "content_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            },
        )
    finally:
        session.close()
