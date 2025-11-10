import sys, os, re, time, concurrent.futures
from functools import lru_cache
sys.path.append(r"D:\AIC2025\beit3\unilm\beit3")

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
from torch.cuda.amp import autocast


MILVUS_URI = "https://in03-e1ea33f293985d0.serverless.aws-eu-central-1.cloud.zilliz.com"
MILVUS_TOKEN = "97d41ad10f6aaa43a375383eafcd638e56761eea7c1fca007a78fa03e762765b6c0263394b01d5d6b376e23d466a3a6a3e976385"
COLLECTION_NAME = "video_embeddings_beit3"

MODEL_WEIGHT_PATH = r"D:\AIC2025\beit3\beit3_base_patch16_384_coco_retrieval.pth"
TOKENIZER_PATH = r"D:\AIC2025\beit3\beit3.spm"

MONGO_URI = "mongodb+srv://nguyentheluan27052005vl_db_user:inseclabhelio123@cluster0.rnsh7kk.mongodb.net/?retryWrites=true&w=majority"
DB_NAME = "obj-detection"
COLLECTION_NAME_OBJ = "object-detection-results"

BASE_DIR = r"D:\AIC2025\keyframes"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection_mongo = db[COLLECTION_NAME_OBJ]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

from modeling_finetune import beit3_base_patch16_384_retrieval


@lru_cache(maxsize=1)
def get_model():
    model = beit3_base_patch16_384_retrieval(pretrained=True)
    ckpt = torch.load(MODEL_WEIGHT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device).half().eval()
    return model


beit3_model = get_model()
tokenizer = XLMRobertaTokenizer(vocab_file=TOKENIZER_PATH)
nltk.download("wordnet", quiet=True)
translator = GoogleTranslator(source="auto", target="en")


@lru_cache(maxsize=512)
def translate_to_english(text):
    if re.match(r"^[a-zA-Z0-9\s]+$", text):
        return text
    try:
        return translator.translate(text) or text
    except Exception:
        return text


@lru_cache(maxsize=512)
def expand_query(translated_query):
    words = translated_query.split()[:5]
    expanded = []
    for word in words:
        synonyms = {lemma.name() for syn in wordnet.synsets(word) for lemma in syn.lemmas()}
        expanded.append(word)
        expanded.extend([s for s in synonyms if s != word][:1])
    expanded = " ".join(dict.fromkeys(expanded))
    return f"Find images related to '{translated_query}'. Scene may include: {expanded}."

@torch.inference_mode()
@lru_cache(maxsize=512)
def encode_text_cached(query: str, max_len=64):
    tokens = tokenizer.tokenize(query)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)[: max_len - 2]
    tokens = [tokenizer.bos_token_id] + token_ids + [tokenizer.eos_token_id]
    pad_len = max_len - len(tokens)
    tokens += [tokenizer.pad_token_id] * pad_len
    padding_mask = [0] * len(tokens) + [1] * (max_len - len(tokens))

    tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    mask_tensor = torch.tensor(padding_mask, dtype=torch.long).unsqueeze(0)

    if torch.cuda.is_available():
        tokens_tensor = tokens_tensor.to(device, non_blocking=True)
        mask_tensor = mask_tensor.to(device, non_blocking=True)

    try:
        with autocast(enabled=torch.cuda.is_available()):
            _, text_emb = beit3_model(text_description=tokens_tensor, padding_mask=mask_tensor, only_infer=True)
            text_emb = normalize(text_emb, p=2, dim=-1)
        return text_emb.cpu().numpy().astype("float32")
    except RuntimeError as e:
        print(f"[WARN] encode_text_cached fallback to CPU: {e}")
        beit3_model_cpu = beit3_model.to("cpu")
        with torch.no_grad():
            _, text_emb = beit3_model_cpu(text_description=tokens_tensor, padding_mask=mask_tensor, only_infer=True)
            text_emb = normalize(text_emb, p=2, dim=-1)
        beit3_model.to(device)
        return text_emb.cpu().numpy().astype("float32")


_transform_384 = transforms.Compose([
    transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])

@torch.inference_mode()
def encode_image(image: Image.Image, image_size=384):
    img = _transform_384(image.convert("RGB")).unsqueeze(0)

    if torch.cuda.is_available():
        img = img.to(device, dtype=torch.float16, non_blocking=True)

    try:
        with autocast(enabled=torch.cuda.is_available()):
            feats, _ = beit3_model(image=img, only_infer=True)
            feats = normalize(feats, p=2, dim=-1)
        return feats.cpu().numpy().astype("float32")
    except RuntimeError as e:
        print(f"[WARN] encode_image fallback to CPU: {e}")
        beit3_model_cpu = beit3_model.to("cpu")
        img = img.float().to("cpu")
        with torch.no_grad():
            feats, _ = beit3_model_cpu(image=img, only_infer=True)
            feats = normalize(feats, p=2, dim=-1)
        beit3_model.to(device)
        return feats.cpu().numpy().astype("float32")


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
        try:
            frame_id = r.entity.get("frame_id")
            path = r.entity.get("path")
        except AttributeError:
            frame_id = r.fields["frame_id"]
            path = r.fields["path"]

        hits.append({
            "score": float(r.distance),
            "row": {"frame_id": frame_id, "path": path}
        })
    return enrich_results(hits)



def enrich_results(hits):
    if not hits:
        return []
    df = pd.DataFrame([{"frame_id": h["row"]["frame_id"], "path": h["row"]["path"], "similarity": h["score"]} for h in hits])
    df["file_path"] = df["path"].astype(str).str.replace("\\", "/", regex=False)
    df["prefix"] = df["file_path"].str.extract(r"^([A-Z0-9]+)_")
    df["abs_path"] = BASE_DIR + "/Videos_" + df["prefix"].fillna("UNKNOWN") + "/" + df["file_path"]
    df["combined_score"] = df["similarity"]
    return df.to_dict(orient="records")


def combine_results(milvus_results, ocr_results, asr_results=None, weights=(0.7, 0.15, 0.15)):
    combined = {}
    for r in milvus_results:
        fp = r["file_path"]
        base_score = r.get("combined_score", r.get("similarity", 0.0))  
        combined[fp] = r.copy()
        combined[fp]["combined_score"] = base_score

    for o in ocr_results:
        fp = o.get("file_path")
        if not fp:
            continue
        if fp in combined:
            combined[fp]["combined_score"] += weights[1] * o.get("ocr_score", 0.0)
        else:
            o["combined_score"] = weights[1] * o.get("ocr_score", 0.0)
            combined[fp] = o
    if asr_results:
        for a in asr_results:
            vid = a.get("video_id", "").lower()
            s, e, score, text = a.get("scene_start", 0), a.get("scene_end", 0), a.get("asr_score", 0.0), a.get("text", "")
            for fp, r in combined.items():
                if vid in fp.lower():
                    try:
                        f_id = int(os.path.splitext(os.path.basename(fp))[0])
                        if s <= f_id <= e:
                            r["combined_score"] += weights[2] * score
                            r["asr_text"] = text
                    except:
                        continue
    return sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)


def normalize_fp(p: str):
    if not p:
        return ""
    p = str(p).strip().replace("\\", "/").lower()
    p = re.sub(r"(^videos?_*)", "", p)
    parts = p.split("/")
    return parts[-2] if len(parts) >= 2 else parts[-1]

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

    # === MAIN QUERY ===
    try:
        q_vec = encode_text_cached(expanded_prompt)
        main_results = search_milvus(q_vec, top_k)
        for r in main_results:
            r["combined_score"] = weights[0] * r.get("similarity", 0.0)
        print(f"[Milvus] {len(main_results)} hits for '{translated_query}'")
    except Exception as e:
        print(f"Milvus search failed: {e}")
        main_results = []

    # === NEXT QUERIES (FASTAPI-style, song song) ===
    if next_queries:
        if isinstance(next_queries, str):
            next_queries = [q.strip() for q in re.split(r"[\r\n]+", next_queries.strip()) if q.strip()]

        print(f"[Next Query] Running {len(next_queries)} parallel searches (FastAPI-style)...")
        next_results_all = []

        # FIX: bảo đảm có ít nhất 1 worker
        max_workers = max(1, min(len(next_queries), 4))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {}
            # 1️⃣ Encode song song
            for nq in next_queries:
                tq = translate_to_english(nq)
                prompt = expand_query(tq) if use_expanded_prompt else tq
                future_map[ex.submit(encode_text_cached, prompt)] = (nq, prompt)

            encoded_texts = {}
            for f in concurrent.futures.as_completed(future_map):
                nq, prompt = future_map[f]
                try:
                    q_vec = f.result()
                    encoded_texts[nq] = (prompt, q_vec)
                except Exception as e:
                    print(f"[Next Query] '{nq}' encode failed: {e}")

            # 2️⃣ Search song song
            search_futures = {}
            for nq, (prompt, q_vec) in encoded_texts.items():
                search_futures[ex.submit(search_milvus, q_vec, top_k)] = nq

            for f in concurrent.futures.as_completed(search_futures):
                nq = search_futures[f]
                try:
                    res = f.result()
                    for r in res:
                        r["query_text"] = nq
                    next_results_all.extend(res)
                    print(f"[Next Query] '{nq}' returned {len(res)} results")
                except Exception as e:
                    print(f"[Next Query] '{nq}' search failed: {e}")

        # 3️⃣ Boost + Append (giống FastAPI)
        boosted, added = 0, 0
        for nr in next_results_all:
            sim = nr.get("similarity", 0.0)
            if sim <= 0.5:  # Ngưỡng lọc
                continue

            fp_next_norm = normalize_fp(nr.get("file_path", ""))
            existing = next(
                (r for r in main_results if normalize_fp(r.get("file_path", "")) == fp_next_norm),
                None
            )

            if existing:
                existing["similarity"] = max(existing.get("similarity", 0.0), sim)
                existing["combined_score"] = existing.get("combined_score", 0.0) + 0.5
                boosted += 1
            else:
                nr["combined_score"] = 0.4 * sim
                main_results.append(nr)
                added += 1

        print(f"[Next Query] Boosted {boosted} overlaps | Added {added} new results")

        # 4️⃣ Gộp trùng lại (merge)
        unique_results = {}
        for r in main_results:
            fp = normalize_fp(r.get("file_path", ""))
            if fp in unique_results:
                # FIX: cập nhật cả similarity và combined_score bằng max
                unique_results[fp]["similarity"] = max(unique_results[fp]["similarity"], r.get("similarity", 0.0))
                unique_results[fp]["combined_score"] = max(unique_results[fp].get("combined_score", 0.0), r.get("combined_score", 0.0))
            else:
                unique_results[fp] = r

        main_results = list(unique_results.values())
        print(f"[Next Query] After merge: {len(main_results)} total results")

    # === OCR + ASR SONG SONG ===
    ocr_results, asr_results = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futures = {}
        if ocr_query:
            futures["ocr"] = ex.submit(search_ocr, ocr_query, top_k)
        if audio_query:
            futures["asr"] = ex.submit(search_asr, audio_query, top_k)

        # FIX: dùng as_completed để tránh block
        for f in concurrent.futures.as_completed(futures.values()):
            key = [k for k, v in futures.items() if v == f][0]
            try:
                if key == "ocr":
                    ocr_results = f.result()
                    for r in ocr_results:
                        r["ocr_score"] = r.get("score", 0.0)
                        r["file_path"] = str(r.get("file_path") or r.get("path", "")).strip()
                    print(f"[OCR] {len(ocr_results)} hits for '{ocr_query}'")
                else:
                    asr_results = f.result()
                    for r in asr_results:
                        r["file_path"] = str(r.get("file_path", "")).strip()
                    print(f"[ASR] {len(asr_results)} hits for '{audio_query}'")
            except Exception as e:
                print(f"{key.upper()} search failed: {e}")

    # === FUSION ===
    fused_results = combine_results(main_results, ocr_results, asr_results, weights=weights)
    inter = len(set(r["file_path"] for r in main_results) & set(r["file_path"] for r in ocr_results))
    print(f"[DEBUG] Overlap between Milvus and OCR = {inter}")

    # === OBJECT FILTER ===
    top_results = fused_results[:top_k]
    if obj_filters:
        try:
            top_results = filter_results_by_objects(
                top_results, filters=obj_filters,
                collection=collection_mongo, require_all=require_all
            )
            print(f"Object filters applied ({len(top_results)} remaining)")
        except Exception as e:
            print(f"Object filter failed: {e}")

    # === SORT & RETURN ===
    final_sorted = sorted(top_results, key=lambda x: x.get("combined_score", 0), reverse=True)
    if final_sorted:  # FIX: tránh IndexError
        print(f"Returned {len(final_sorted)} results (Top1={final_sorted[0]['combined_score']:.3f})")
    else:
        print("Returned 0 results")
    print(f"Total time: {time.time() - t0:.3f}s")

    return {
        "results": final_sorted,
        "original_query": search_query,
        "translated_query": translated_query,
        "expanded_prompt": expanded_prompt if use_expanded_prompt else None,
        "ocr_query": ocr_query,
    }
