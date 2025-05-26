import os
from typing import List

class Settings:
    # Model configurations
    NER_BASE_TOKENIZER = "cl-tohoku/bert-base-japanese-v3"
    NER_MODEL = "knosing/japanese_ner_model"
    TRANSLATION_MODEL = "linhdzqua148/opus-mt-ja-en-railway-7"
    
    # Database configurations - use environment variables if available, otherwise use defaults
    DB_HOST = os.getenv("DB_HOST", "103.97.126.29")
    DB_PORT = int(os.getenv("DB_PORT", "3306"))
    DB_NAME = os.getenv("DB_NAME", "dsvxxzme_itss")
    DB_USER = os.getenv("DB_USER", "dsvxxzme_itss")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "12345678")
    DB_CHARSET = "utf8mb4"
    
    # NER tags for train announcements
    NER_TAGS = ["地名", "施設名", "法人名", "製品名", "その他の組織名"]
    
    # API configurations
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # Device configuration
    DEVICE = "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"

settings = Settings()
