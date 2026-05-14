import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env
_env = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env, override=True)
api_key = os.environ.get("GEMINI_API_KEY", "")
print(f"Key: {api_key[:8]}...{api_key[-4:]}")
print(f"Key length: {len(api_key)}")

from google import genai
from google.genai import types

client = genai.Client(api_key=api_key)

# Test all flash models:
MODELS_TO_TEST = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-2.5-flash",
    "gemini-1.5-pro",
]

print("\n" + "=" * 50)
print("Testing models (text only — no image):")
print("=" * 50)

working_models = []
for model_name in MODELS_TO_TEST:
    try:
        r = client.models.generate_content(
            model=model_name,
            contents="Trả lời đúng 1 từ: Thủ đô của Việt Nam là gì?",
        )
        print(f"✅ {model_name:30} → {r.text.strip()[:30]}")
        working_models.append(model_name)
    except Exception as e:
        err = str(e)[:80]
        print(f"❌ {model_name:30} → {err}")

print("\n" + "=" * 50)
if working_models:
    print(f"WORKING MODELS: {working_models}")
    print(f"RECOMMENDED: {working_models[0]}")
else:
    print("NO MODELS WORK — Check API key or billing")
print("=" * 50)
