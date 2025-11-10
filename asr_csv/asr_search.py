from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

ES_URL = "https://37351937d15c455eaafe4e249cdebe36.us-central1.gcp.cloud.es.io:443"   # hoặc URL Elastic Cloud
API_KEY = "WGFlYkZKb0JIR2o3Vk9SY3d1ZG46UEk4eXJDUTQ0MGQwaV9WLVNIRThjdw=="            # nếu dùng Elastic Cloud
INDEX_NAME = "search-cse6"

# ===== CONNECT =====
es = Elasticsearch(ES_URL, api_key=API_KEY, verify_certs=True)

def search_asr(query, top_k=50):
    """
    Tìm kiếm text trong transcript ASR từ ElasticSearch
    """
    query_body = {
        "size": top_k,
        "query": {
            "match": {
                "text": {
                    "query": query,
                    "fuzziness": "AUTO"   # cho phép tìm xấp xỉ
                }
            }
        },
        "_source": ["video_id", "scene_start", "scene_end", "text"]
    }

    res = es.search(index=INDEX_NAME, body=query_body)
    results = []
    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        score = hit["_score"]
        results.append({
            "video_id": src.get("video_id"),
            "scene_start": src.get("scene_start"),
            "scene_end": src.get("scene_end"),
            "text": src.get("text", ""),
            "asr_score": float(score),
            "file_path": f"D:\\AIC2025\\keyframes\\{src.get('video_id')}\\{src.get('scene_start')}.jpg"
        })
    return results
