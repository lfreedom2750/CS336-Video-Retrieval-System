from elasticsearch import Elasticsearch
import os
import re

OCR_URL = "https://ocr-e4a8ae.es.us-central1.gcp.elastic.cloud:443"
OCR_API_KEY = "cEx0Wklwc0Jsc1cwdWJLTk05eXE6OWl3QWhMNVdyaVR0a1ZwMlZOQTN3QQ=="
INDEX_NAME = "national-parks"

client = Elasticsearch(OCR_URL, api_key=OCR_API_KEY)


def normalize_ocr_path(path: str) -> str:
    path = str(path).replace("\\", "/").strip()
    path = re.sub(r"^videos_[a-z0-9]+/", "", path, flags=re.IGNORECASE)
    path = re.sub(r"^[a-z]:/aic2025/keyframes/", "", path, flags=re.IGNORECASE)
    path = re.sub(r"^keyframes/", "", path, flags=re.IGNORECASE)
    path = re.sub(r"^.*?/videos_[a-z0-9]+/", "", path, flags=re.IGNORECASE)
    return path


def path_to_frame_id(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    parent = parts[-2] if len(parts) >= 2 else parts[0]
    fname = parts[-1].split(".")[0]
    return f"{parent}_{fname.zfill(5)}"


def search_ocr(query: str, top_k: int = 20):
    body = {"size": top_k, "query": {"match": {"text": query}}}
    res = client.search(index=INDEX_NAME, body=body)
    hits = []

    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        raw_path = src.get("path", "")
        norm_path = normalize_ocr_path(raw_path)

        hits.append({
            "frame_id": path_to_frame_id(norm_path),
            "path": norm_path,
            "file_path": norm_path,
            "ocr_text": src.get("text", ""),
            "score": hit["_score"]
        })
    return hits
