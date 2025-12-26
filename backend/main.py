import sys, os, io, re, numpy as np
current_folder = os.path.dirname(os.path.abspath(__file__))

if current_folder not in sys.path:
    sys.path.append(current_folder)

from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from milvus import milvus_beit3_cpu as bf
from gemini_services import query_gemini
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
 
KEYFRAME_ROOT = r"/home/ir/data/keyframes"
BASE_URL = "http://127.0.0.1:7860/frames"

if os.path.exists(KEYFRAME_ROOT):
    app.mount("/frames", StaticFiles(directory=KEYFRAME_ROOT), name="frames")
    print(f"Đã mount thư mục ảnh: {KEYFRAME_ROOT} --> /frames")
else:
    print(f"LỖI: Không tìm thấy thư mục ảnh tại {KEYFRAME_ROOT}")


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


# KHÔNG BAO GIỜ thêm Videos_Kxx hoặc Videos_Lxx
# path trả về từ Milvus luôn dạng L15_V018/1234.jpg

def convert_local_to_url(path: str) -> str:
    """
    Chuyển đường dẫn file (tuyệt đối hoặc tương đối) thành URL
    VD: /home/ir/data/keyframes/L01_V001/1.jpg -> http://.../frames/L01_V001/1.jpg
    """
    try:
        if not path:
            return None

        # 1. Chuẩn hóa dấu gạch chéo
        p = str(path).replace("\\", "/").strip()

        # 2. Nếu là đường dẫn tuyệt đối (bắt đầu bằng KEYFRAME_ROOT), cắt bỏ phần đầu
        # Lưu ý: KEYFRAME_ROOT cũng cần chuẩn hóa
        root_normalized = KEYFRAME_ROOT.replace("\\", "/")
        
        if p.startswith(root_normalized):
            # Cắt bỏ phần gốc, chỉ lấy phần đuôi (VD: L01_V001/1.jpg)
            rel = p[len(root_normalized):]
            # Xóa dấu / ở đầu nếu còn dư
            if rel.startswith("/"):
                rel = rel[1:]
        else:
            # Nếu đã là tương đối thì giữ nguyên
            rel = p

        # 3. Ghép thành URL
        return f"{BASE_URL}/{rel}"

    except Exception as e:
        print(f"convert_local_to_url ERROR: {e}")
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
        print("RECEIVED FRAME_ID:", frame_id)

        # CHỈ NHẬN Lxx_Vxxx
        m = re.match(r"(L\d+_V\d+)_(\d+)", frame_id, re.IGNORECASE)
        if not m:
            return JSONResponse({"status": "error", "message": "Invalid frame_id"}, status_code=400)

        video_id, frame_num = m.group(1), int(m.group(2))

        # KHÔNG THÊM Videos_Lxx NỮA
        base_dir = os.path.join(KEYFRAME_ROOT, video_id)

        # load frames thật
        all_frames = sorted(
            [f for f in os.listdir(base_dir) if re.match(r"^\d+\.jpg$", f)],
            key=lambda x: int(x.split(".")[0])
        )

        if not all_frames:
            return JSONResponse({"status": "error", "message": "No frames found"}, status_code=404)

        # vị trí frame
        try:
            idx = all_frames.index(f"{frame_num}.jpg")
        except ValueError:
            return JSONResponse({"status": "error", "message": "Frame not found"}, status_code=404)

        # lấy hàng xóm
        neighbors_files = all_frames[max(0, idx - 12): idx + 13]

        neighbors = []
        for fname in neighbors_files:
            fnum = int(fname.split(".")[0])
            neighbors.append({
                "frame_id": f"{video_id}_{fnum:05d}",
                "path": f"{video_id}/{fname}"
            })

        return JSONResponse({"status": "ok", "neighbors": neighbors})

    except Exception as e:
        print("❌ Error in /api/context:", e)
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)




# ==================== RUN ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
