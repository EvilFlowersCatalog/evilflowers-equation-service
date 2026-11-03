import os

from dotenv import load_dotenv

load_dotenv()

#: Celery broker: the catalog's Redis.
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/7")
QUEUE = os.getenv("EQUATION_QUEUE", "evilflowers_equation")

#: The catalog's storage root (originals are read, crops written).
STORAGE_PATH = os.getenv("STORAGE_PATH", "/mnt/data")

CATALOG_API_URL = os.getenv("CATALOG_API_URL", "http://catalog:8000")
#: API key (bearer JWT) of the catalog user holding `extraction.ingest_extraction`.
CATALOG_API_KEY = os.getenv("CATALOG_API_KEY")
IMAGE_TAG = os.getenv("IMAGE_TAG", "dev")

CROP_DPI = int(os.getenv("CROP_DPI", "200"))
WEBP_QUALITY = int(os.getenv("WEBP_QUALITY", "90"))

#: `onnx`: LaTeX-OCR (pix2tex) weights run with ONNX Runtime on CPU, downloaded
#: to `MODELS_PATH` on first use. `ollama`: a multimodal model behind an Ollama
#: endpoint transcribes the crop. `none`: crops and positions only.
LATEX_BACKEND = os.getenv("LATEX_BACKEND", "onnx")
MODELS_PATH = os.getenv("MODELS_PATH", "/models/latex-ocr")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://ollama:11434/api/chat")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
LATEX_MODEL = os.getenv("LATEX_MODEL", "qwen2.5vl:7b")
LATEX_TIMEOUT = int(os.getenv("LATEX_TIMEOUT", "120"))
