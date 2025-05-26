from pydantic import BaseModel, Field
from typing import Dict, Optional

class TranslationRequest(BaseModel):
    text: str = Field(..., description="Japanese announcement text to translate")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "ご乗車ありがとうございます。中央線快速、東京行きです。次は新宿、新宿です。"
            }
        }

class TranslationResponse(BaseModel):
    original_text: str = Field(..., description="Original Japanese text")
    text_with_placeholders: str = Field(..., description="Text with placeholders replacing entities")
    entity_mapping: Dict[str, str] = Field(..., description="Mapping from placeholders to original entities")
    english_translation: str = Field(..., description="Final English translation")
    entities_count: int = Field(..., description="Number of entities detected")
    
    class Config:
        json_schema_extra = {
            "example": {
                "original_text": "ご乗車ありがとうございます。中央線快速、東京行きです。次は新宿、新宿です。",
                "text_with_placeholders": "ご乗車ありがとうございます。[PH1]快速、[PH2]行きです。次は[PH3]、[PH3]です。",
                "entity_mapping": {
                    "[PH1]": "中央線",
                    "[PH2]": "東京",
                    "[PH3]": "新宿"
                },
                "english_translation": "Thank you for riding. Chuo Line Express, bound for Tokyo. Next is Shinjuku, Shinjuku.",
                "entities_count": 3
            }
        }

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    services: Dict[str, bool] = Field(..., description="Status of individual services")
    database_entities_count: int = Field(..., description="Number of entities in database cache")

class EntitySearchResponse(BaseModel):
    query: str = Field(..., description="Search query")
    results: Dict[str, str] = Field(..., description="Search results mapping Japanese to English")
    count: int = Field(..., description="Number of results found")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "東京",
                "results": {
                    "東京": "Tokyo",
                    "東京駅": "Tokyo Station"
                },
                "count": 2
            }
        }
