import os
import csv
import requests
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="DRES Submission API")

DRES_BASE = "https://eventretrieval.oj.io.vn/api/v2"
EVALUATION_ID = "06236d7d-368e-44ac-a388-c955cb374a7d"
SESSION_ID = "BS9RkFlS2DUbluunInRIf9LUJTKcdrsA"
FPS_CSV_PATH = r"D:\AIC2025\videos_fps.csv"

TEST_MODE = False

def load_fps_table(csv_path=FPS_CSV_PATH):
    fps_table = {}
    if not os.path.exists(csv_path):
        return fps_table
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id") or row.get("VideoID") or row.get("id")
            fps = row.get("fps") or row.get("FPS")
            if vid and fps:
                try:
                    fps_table[vid.strip()] = float(fps)
                except:
                    continue
    return fps_table


FPS_TABLE = load_fps_table()


def get_fps(video_id: str) -> float:
    return FPS_TABLE.get(video_id, 30.0)


def frame_to_time_ms(video_id: str, frame_index: int) -> int:
    fps = get_fps(video_id)
    print(f"Video ID: {video_id}, FPS: {fps}, Frame Index: {frame_index}")
    return int((frame_index / fps) * 1000)


@app.post("/api/submit-qa")
async def submit_qa(
    videos_ID: str = Form(...),
    frame_index: int = Form(...),
    answer: str = Form(...)
):
    # chuyển frame index → time (ms)
    time_ms = frame_to_time_ms(videos_ID, frame_index)

    # tạo body đúng chuẩn
    body_data = {
        "answerSets": [
            {
                "answers": [
                    {
                        "text": f"QA-{answer}-{videos_ID}-{time_ms}"
                    }
                ]
            }
        ]
    }

    # URL submit DRES
    url = f"{DRES_BASE}/submit/{EVALUATION_ID}"
    params = {"session": SESSION_ID}

    # bật test mode khi muốn kiểm tra local
    TEST_MODE = False  # ← đổi thành False khi gửi thật
    if TEST_MODE:
        print(f"[TEST] Would submit QA: {body_data}")
        return JSONResponse(content={"status": "test_ok", "body": body_data})

    try:
        response = requests.post(url, params=params, json=body_data)
        response.raise_for_status()
        return JSONResponse(content=response.json())
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/submit-kis")
async def submit_kis(videos_ID: str = Form(...), frame_start: int = Form(...), frame_end: int = Form(...)):
    start_ms = frame_to_time_ms(videos_ID, frame_start)
    end_ms = frame_to_time_ms(videos_ID, frame_end)
    body_data = {
        "answerSets": [
            {
                "answers": [
                    {"mediaItemName": videos_ID, "start": start_ms, "end": end_ms}
                ]
            }
        ]
    }
    url = f"{DRES_BASE}/submit/{EVALUATION_ID}"
    params = {"session": SESSION_ID}
    try:
        response = requests.post(url, params=params, json=body_data)
        response.raise_for_status()
        return JSONResponse(content=response.json())
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/submit-trake")
async def submit_trake(
    videos_ID: str = Form(...),
    frame_ids: str = Form(...)
):
    # xử lý danh sách frame id
    try:
        frames = [int(f.strip()) for f in frame_ids.split(",") if f.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid frame IDs (must be integers).")

    if not frames:
        raise HTTPException(status_code=400, detail="No valid frame IDs provided.")

    frames_str = ",".join(map(str, frames))

    # tạo body theo đúng định dạng yêu cầu
    body_data = {
        "answerSets": [
            {
                "answers": [
                    {"text": f"TR-{videos_ID}-{frames_str}"}
                ]
            }
        ]
    }

    url = f"{DRES_BASE}/submit/{EVALUATION_ID}"
    params = {"session": SESSION_ID}

    # Nếu đang chạy local, có thể bật test_mode để không gửi thật
    TEST_MODE = False
    if TEST_MODE:
        print(f"[TEST] Would submit TRAKE: {body_data}")
        return JSONResponse(content={"status": "test_ok", "body": body_data})

    try:
        response = requests.post(url, params=params, json=body_data)
        response.raise_for_status()
        return JSONResponse(content=response.json())
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/list-fps")
async def list_fps():
    if not FPS_TABLE:
        return {"status": "error", "message": "⚠️ Không có dữ liệu FPS nào được load"}
    return {
        "count": len(FPS_TABLE),
        "samples": dict(list(FPS_TABLE.items())[:10])  # hiển thị 10 dòng đầu
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("submit_dres:app", host="0.0.0.0", port=8081, reload=True)
