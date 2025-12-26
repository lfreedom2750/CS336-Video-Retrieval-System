# object_filter.py
import os
from pymongo import MongoClient

# ====== KẾT NỐI MONGODB ======
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://nguyentheluan27052005vl_db_user:inseclabhelio123@cluster0.rnsh7kk.mongodb.net/")
DB_NAME = "obj-detection"
COLLECTION_NAME = "object-detection-results"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

try:
    print("✅ Connected to MongoDB Atlas!")
    print("📂 Collections:", db.list_collection_names())
except Exception as e:
    print("❌ MongoDB connection failed:", e)

def filter_results_by_objects(results, filters=None, collection=None, require_all=False):
    """
    Lọc kết quả theo object (person, car, v.v.)
    Milvus lưu: file_path = "K09_V028/39125.jpg"
    MongoDB lưu: { "video_id": "K09_V028", "keyframe_id": "39125.jpg", "class": "person" }
    """
    import re

    if not filters or collection is None:
        return results

    filters = [f.strip().lower() for f in filters if f.strip()]
    filtered = []
    total = len(results)

    for r in results:
        file_path = str(r.get("file_path") or r.get("path") or "").replace("\\", "/")
        if not file_path:
            continue

        # ✅ parse trực tiếp từ dạng K09_V028/39125.jpg
        m = re.match(r"([KL]\d+_V\d+)/(\d+)\.jpg", file_path, re.IGNORECASE)
        if not m:
            continue

        video_id = m.group(1)
        keyframe_id = f"{int(m.group(2))}.jpg"

        try:
            # 🔍 Query MongoDB
            query = {
                "video_id": video_id,
                "keyframe_id": keyframe_id,
                "class": {"$in": filters},
            }
            count = collection.count_documents(query)
            if count == 0:
                continue

            if require_all:
                docs = collection.find(
                    {"video_id": video_id, "keyframe_id": keyframe_id},
                    {"class": 1}
                )
                present = [d["class"].lower() for d in docs]
                if all(f in present for f in filters):
                    filtered.append(r)
            else:
                filtered.append(r)

        except Exception as e:
            print(f"[ObjectFilter] Query failed for {video_id}-{keyframe_id}: {e}")

    print(f"[ObjectFilter] {len(filtered)}/{total} frames kept after filtering.")
    return filtered



if __name__ == "__main__":
    print("🧩 Testing MongoDB connection...")
    print("Total documents:", collection.count_documents({}))
    sample = collection.find_one()
    print("🔍 Sample document:", sample)