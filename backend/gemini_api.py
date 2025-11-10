# 📄 gemini_api.py
import os
from google import genai

API_KEY = "AIzaSyCsnnCSuGtihfmSJxX3cJxJpC_5z5G8Scs"
client = genai.Client(api_key=API_KEY)

def query_gemini(prompt: str, model: str = "gemini-2.0-flash") -> str:
    """
    Gửi prompt tới Gemini và trả về phản hồi text (hỗ trợ SDK mới).
    """
    system_prompt = """
    You are a Query Refinement Assistant for a multimodal video retrieval system (BEiT3 model).
    Your task: rewrite vague or non-English user queries into a compact and descriptive English sentence
    that clearly depicts the visual scene, objects, and actions mentioned.

    Output ONLY the refined English sentence — concise, natural, and grammatically correct.
    Do NOT include explanations or section headers.
    """

    try:
        response = client.models.generate_content(
            model=model,
            contents=f"{system_prompt}\nUser query: {prompt}",
        )

        # 🧠 Các trường hợp có thể xảy ra
        if response is None:
            return "⚠️ Empty response (None returned)."

        # Gemini SDK mới có .text, .output_text, hoặc .candidates
        if hasattr(response, "text"):
            return response.text.strip()

        if hasattr(response, "output_text"):
            return response.output_text.strip()

        if hasattr(response, "candidates") and len(response.candidates) > 0:
            parts = response.candidates[0].content.parts
            if parts and hasattr(parts[0], "text"):
                return parts[0].text.strip()

        return "⚠️ No valid text found in response."

    except Exception as e:
        print("❌ Lỗi gọi Gemini:", e)
        return f"Error: {e}"
