import sys, os, re, time, concurrent.futures
from functools import lru_cache
sys.path.append(r"D:\CS336\beit3\unilm\beit3")

import numpy as np
import torch
from torch.nn.functional import normalize
from torchvision import transforms
from PIL import Image
from transformers import XLMRobertaTokenizer
import pandas as pd
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import wordnet
from pymongo import MongoClient
from pymilvus import connections, Collection
from ocr_csv.ocr_search import search_ocr
from asr_csv.asr_search import search_asr
from object_filter import filter_results_by_objects


# ===========================
# CONSTANT CONFIG
# ===========================
MILVUS_URI = "https://in03-72ca0c57717b311.serverless.aws-eu-central-1.cloud.zilliz.com"
MILVUS_TOKEN = "6e46340b0b47b937701acaf37c4a867542b9140286350efb9459919d0e6da1caa4d0b96db715e1b6a6fc003e1a1520ad9c18c21c"
COLLECTION_NAME = "video_embeddings_beit3"

MODEL_WEIGHT_PATH = r"D:\AIC2025\beit3\beit3_base_patch16_384_coco_retrieval.pth"
TOKENIZER_PATH = r"D:\AIC2025\beit3\beit3.spm"

MONGO_URI = "mongodb+srv://nguyentheluan27052005vl_db_user:inseclabhelio123@cluster0.jddn1ha.mongodb.net/?retryWrites=true&w=majority"
DB_NAME = "obj-detection"
COLLECTION_NAME_OBJ = "object-detection-results"

BASE_DIR = r"D:\CS336\keyframes"


# ===========================
# CPU DEVICE
# ===========================
device = "cpu"
print("Using device:", device)

from modeling_finetune import beit3_base_patch16_384_retrieval


# ===========================
# LOAD MODEL ON CPU
# ===========================
@lru_cache(maxsize=1)
def get_model():
    model = beit3_base_patch16_384_retrieval(pretrained=True)
    ckpt = torch.load(MODEL_WEIGHT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    model = model.to("cpu")     # CPU only
    model.eval()                # no half(), no autocast
    return model


beit3_model = get_model()
tokenizer = XLMRobertaTokenizer(vocab_file=TOKENIZER_PATH)
nltk.download("wordnet", quiet=True)
translator = GoogleTranslator(source="auto", target="en")


# ===========================
# TEXT PROCESSING
# ===========================
@lru_cache(maxsize=512)
def translate_to_english(text):
    if re.match(r"^[a-zA-Z0-9\s]+$", text):
        return text
    try:
        return translator.translate(text) or text
    except:
        return text


@lru_cache(maxsize=512)
def expand_query(translated_query):
    words = translated_query.split()[:5]
    expanded = []
    for word in words:
        synonyms = {lemma.name() for syn in wordnet.synsets(word)
                                for lemma in syn.lemmas()}
        expanded.append(word)
        expanded.extend([s for s in synonyms if s != word][:1])
    expanded = " ".join(dict.fromkeys(expanded))
    return f"Find images related to '{translated_query}'. Scene may include: {expanded}."


# ===========================
# CPU TEXT ENCODER
# ===========================
@torch.no_grad()
@lru_cache(maxsize=512)
def encode_text_cached(query: str, max_len=64):
    tokens = tokenizer.tokenize(query)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)[: max_len - 2]

    tokens = [tokenizer.bos_token_id] + token_ids + [tokenizer.eos_token_id]
    tokens += [tokenizer.pad_token_id] * (max_len - len(tokens))
    padding_mask = [0] * len(tokens)

    tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to("cpu")
    mask_tensor = torch.tensor(padding_mask, dtype=torch.long).unsqueeze(0).to("cpu")

    _, text_emb = beit3_model(
        text_description=tokens_tensor,
        padding_mask=mask_tensor,
        only_infer=True
    )
    text_emb = normalize(text_emb, p=2, dim=-1)

    return text_emb.cpu().numpy().astype("float32")


# ===========================
# CPU IMAGE ENCODER
# ===========================
_transform_384 = transforms.Compose([
    transforms.Resize((384, 384),
                      interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])

@torch.no_grad()
def encode_image(image: Image.Image, image_size=384):
    img = _transform_384(image.convert("RGB")).unsqueeze(0).to("cpu").float()

    feats, _ = beit3_model(image=img, only_infer=True)
    feats = normalize(feats, p=2, dim=-1)

    return feats.cpu().numpy().astype("float32")


# ===========================
# MILVUS
# ===========================
connections.connect(alias="default", uri=MILVUS_URI, token=MILVUS_TOKEN)
coll = Collection(COLLECTION_NAME)
coll.load()
print(f"Connected to Milvus: {COLLECTION_NAME} ({coll.num_entities:,} vectors)")


def search_milvus(query_vec, top_k=100):
    params = {"metric_type": "IP", "params": {"ef": 128}}
    results = coll.search(
        data=query_vec.tolist(),
        anns_field="embedding",
        param=params,
        limit=top_k,
        output_fields=["frame_id", "path"],
    )

    hits = []
    for r in results[0]:
        frame_id = r.entity.get("frame_id")
        path = r.entity.get("path")
        hits.append({
            "score": float(r.distance),
            "row": {"frame_id": frame_id, "path": path}
        })

    return enrich_results(hits)


# ===========================
# ENRICH RESULTS
# ===========================
def enrich_results(hits):
    if not hits:
        return []
    df = pd.DataFrame([
        {
            "frame_id": h["row"]["frame_id"],
            "path": h["row"]["path"],
            "similarity": h["score"]
        }
        for h in hits
    ])

    df["file_path"] = df["path"].astype(str).str.replace("\\", "/", regex=False)
    df["video_id"] = df["file_path"].apply(lambda p: p.split("/")[0])
    df["filename"] = df["file_path"].apply(lambda p: p.split("/")[-1])

    df["abs_path"] = df.apply(
        lambda r: os.path.join(BASEpath := BASE_DIR, r["video_id"], r["filename"]),
        axis=1,
    )

    df["combined_score"] = df["similarity"]
    return df.to_dict(orient="records")


# =========================================================
# COMBINE RESULTS (OCR, ASR, ...)
# =========================================================

def combine_results(milvus_results, ocr_results, asr_results=None, weights=(0.7,0.15,0.15)):
    combined = {}
    for r in milvus_results:
        fp = r["file_path"]
        base_score = r.get("combined_score", r.get("similarity", 0.0))
        combined[fp] = r.copy()
        combined[fp]["combined_score"] = base_score

    for o in ocr_results:
        fp = o.get("file_path")
        if fp:
            combined.setdefault(fp, {"combined_score": 0})
            combined[fp]["combined_score"] += weights[1] * o.get("ocr_score", 0.0)

    if asr_results:
        for a in asr_results:
            vid = a.get("video_id", "").lower()
            s, e, score = a.get("scene_start",0), a.get("scene_end",0), a.get("asr_score",0.0)
            for fp, r in combined.items():
                if vid in fp.lower():
                    try:
                        f_id = int(os.path.splitext(os.path.basename(fp))[0])
                        if s <= f_id <= e:
                            r["combined_score"] += weights[2] * score
                    except:
                        pass

    return sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)



# =========================================================
# NORMALIZE PATH
# =========================================================
def normalize_fp(p: str):
    if not p:
        return ""
    p = str(p).strip().replace("\\", "/").lower()
    p = re.sub(r"(^videos?_*)", "", p)
    parts = p.split("/")
    return parts[-2] if len(parts) >= 2 else parts[-1]


# =========================================================
# MAIN SEARCH PIPELINE
# =========================================================
def run_search(
    search_query: str,
    next_queries=None,
    ocr_query=None,
    audio_query=None,
    top_k=300,
    use_expanded_prompt=True,
    obj_filters=None,
    require_all=False,
):

    t0 = time.time()
    weights = (0.7, 0.15, 0.15)

    translated_query = translate_to_english(search_query)
    expanded_prompt = expand_query(translated_query) if use_expanded_prompt else translated_query

    # MILVUS MAIN
    try:
        q_vec = encode_text_cached(expanded_prompt)
        main_results = search_milvus(q_vec, top_k)
        for r in main_results:
            r["combined_score"] = weights[0] * r["similarity"]
    except Exception as e:
        print("Milvus failed:", e)
        main_results = []

    # OCR + ASR
    ocr_results, asr_results = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futures = {}
        if ocr_query:
            futures["ocr"] = ex.submit(search_ocr, ocr_query, top_k)
        if audio_query:
            futures["asr"] = ex.submit(search_asr, audio_query, top_k)

        for key, f in futures.items():
            try:
                res = f.result()
                if key == "ocr":
                    for r in res:
                        r["ocr_score"] = r.get("score", 0.0)
                    ocr_results = res
                else:
                    asr_results = res
            except:
                pass

    fused_results = combine_results(main_results, ocr_results, asr_results, weights)

    # OBJECT FILTER
    top_results = fused_results[:top_k]
    if obj_filters:
        try:
            top_results = filter_results_by_objects(
                top_results,
                filters=obj_filters,
                collection=collection_mongo,
                require_all=require_all
            )
        except:
            pass

    final_sorted = sorted(top_results, key=lambda x: x["combined_score"], reverse=True)

    print(f"Total time: {time.time() - t0:.3f}s")

    return {
        "results": final_sorted,
        "original_query": search_query,
        "translated_query": translated_query,
        "expanded_prompt": expanded_prompt if use_expanded_prompt else None,
        "ocr_query": ocr_query,
    }
