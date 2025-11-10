import sys, os, io, re, numpy as np
sys.path.append(r"D:\AIC2025")


from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from milvus import milvus_beit3v2 as bf
from gemini_api import query_gemini
from ocr_csv.ocr_search import search_ocr
from asr_csv.asr_search import search_asr
from dres import app as dres_app
# ==================== APP CONFIG ====================
app = FastAPI(title="AIC2025 Video Search API")
app.mount("/dres", dres_app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

KEYFRAME_ROOT = r"D:\AIC2025\keyframes"
BASE_URL = "http://127.0.0.1:7860/frames"

if os.path.exists(KEYFRAME_ROOT):
    app.mount("/frames", StaticFiles(directory=KEYFRAME_ROOT), name="frames")
    print("Mounted keyframes at /frames →", KEYFRAME_ROOT)
else:
    raise RuntimeError(f"Keyframe folder not found: {KEYFRAME_ROOT}")


# ==================== HELPERS ====================
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj


def convert_local_to_url(path: str) -> str:
    try:
        path = str(path).strip().replace("\\", "/")
        if not path:
            return None

        # Tự động thêm prefix "Videos_XX" nếu thiếu
        m = re.match(r"^([KL]\d+)_", path, re.IGNORECASE)
        if m and not path.lower().startswith("videos_"):
            prefix = m.group(1).upper()
            path = f"Videos_{prefix}/{path}"

        abs_path = os.path.normpath(os.path.join(KEYFRAME_ROOT, path))
        rel = os.path.relpath(abs_path, KEYFRAME_ROOT).replace("\\", "/")
        url = f"{BASE_URL}/{rel}"

        if not os.path.exists(abs_path):
            print(f"[WARN] File not found: {abs_path}")
            return None

        return url
    except Exception as e:
        print(f"[ERROR] convert_local_to_url failed: {e}")
        return None



# ==================== SEARCH API ====================
@app.post("/api/search")
async def api_search(
    query: str = Form(""),
    next_q: str = Form(""),
    ocr_query: str = Form(""),
    audio_query: str = Form(""),
    objects: str = Form(""),
    require_all: bool = Form(False),
    topk: int = Form(500),
    image: UploadFile = File(None),
    use_expanded_prompt: bool = Form(True),
):
    try:
        obj_filters = [o.strip() for o in objects.splitlines() if o.strip()]

        # === TEXT QUERY ===
        if query:
            print(f"\n=== 🔍 MAIN QUERY: '{query}' ===")

            res_obj = bf.run_search(
                search_query=query,
                next_queries=next_q.splitlines() if next_q else None,
                ocr_query=ocr_query or None,
                audio_query=audio_query or None,
                use_expanded_prompt=use_expanded_prompt,
                top_k=topk,
                obj_filters=obj_filters or None,
                require_all=require_all,
            )

            results = res_obj["results"]
            safe_results = make_json_safe(results[:topk])

            for r in safe_results:
                # Ưu tiên abs_path (nếu backbone đã gắn)
                local_path = r.get("abs_path") or r.get("file_path") or r.get("path")
                r["url"] = convert_local_to_url(local_path) or f"{BASE_URL}/"

            print(f"✅ Returned {len(safe_results)} results (Top1 combined={safe_results[0]['combined_score']:.3f})")
            return JSONResponse({"status": "ok", "results": safe_results})

        # === OCR-ONLY SEARCH ===
        elif ocr_query:
            print(f"\n=== 🔍 OCR-ONLY SEARCH: '{ocr_query}' ===")
            ocr_results = search_ocr(ocr_query, top_k=topk)
            safe_results = make_json_safe(ocr_results)
            for r in safe_results:
                r["url"] = convert_local_to_url(r.get("file_path") or r.get("path", "")) or f"{BASE_URL}/"
            print(f"✅ Returned {len(safe_results)} OCR-only results")
            return JSONResponse({"status": "ok", "results": safe_results})

        # === IMAGE-ONLY SEARCH ===
        elif image and image.filename:
            print(f"\n=== 🖼️ IMAGE SEARCH: '{image.filename}' ===")
            content = await image.read()
            img = Image.open(io.BytesIO(content)).convert("RGB")
            vec = bf.encode_image(img)
            img_res = bf.search_milvus(vec, top_k=topk)
            safe_results = make_json_safe(img_res)
            for r in safe_results:
                local_path = r.get("abs_path") or r.get("file_path") or r.get("path")
                r["url"] = convert_local_to_url(local_path) or f"{BASE_URL}/"
            return JSONResponse({"status": "ok", "results": safe_results})

        else:
            return JSONResponse({"status": "error", "message": "No query or image provided."}, status_code=400)

    except Exception as e:
        print("❌ Error in /api/search:", e)
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ==================== ASR SEARCH ====================
@app.post("/api/asr")
async def asr_search_api(query: str = Form(...), topk: int = Form(50)):
    try:
        results = search_asr(query, top_k=topk)
        for r in results:
            local_path = r.get("file_path") or r.get("path")
            r["url"] = convert_local_to_url(local_path) or f"{BASE_URL}/"
        return JSONResponse({"status": "ok", "results": make_json_safe(results)})
    except Exception as e:
        print("❌ ASR search error:", e)
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ==================== CHATBOT ====================
@app.post("/api/chatbot")
async def chatbot_api(prompt: str = Form(...)):
    response = query_gemini(prompt)
    return JSONResponse({"reply": response})

@app.get("/api/context/{frame_id}")
async def api_context(frame_id: str):
    try:
        m = re.match(r"([A-Z]\d+_V\d+)_(\d+)", frame_id, re.IGNORECASE)
        if not m:
            return JSONResponse({"status": "error", "message": "Invalid frame_id"}, status_code=400)

        video_id, frame_num = m.group(1), int(m.group(2))
        prefix = video_id.split("_")[0].upper()
        base_dir = os.path.join(KEYFRAME_ROOT, f"Videos_{prefix}", video_id)

        # 🔹 Lấy toàn bộ frame hiện có
        all_frames = sorted([
            f for f in os.listdir(base_dir)
            if f.lower().endswith(".jpg") and re.match(r"^\d+\.jpg$", f)
        ], key=lambda x: int(x.split(".")[0]))

        if not all_frames:
            return JSONResponse({"status": "error", "message": "No frames found"}, status_code=404)

        # 🔹 Tìm vị trí của frame hiện tại
        idx = None
        for i, f in enumerate(all_frames):
            if f"{frame_num}.jpg" == f:
                idx = i
                break
        if idx is None:
            return JSONResponse({"status": "error", "message": "Frame not found"}, status_code=404)

        # 🔹 Lấy 12 frame trước và 12 frame sau (theo danh sách thật)
        neighbors_files = all_frames[max(0, idx-12): idx+13]

        neighbors = []
        for fname in neighbors_files:
            fnum = int(fname.split(".")[0])
            rel_path = f"Videos_{prefix}/{video_id}/{fname}"
            neighbors.append({
                "frame_id": f"{video_id}_{fnum:05d}",
                "path": rel_path
            })

        return JSONResponse({"status": "ok", "neighbors": neighbors})
    except Exception as e:
        print("❌ Error in /api/context:", e)
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)



# ==================== RUN ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
