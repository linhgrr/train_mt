from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uvicorn

from models.schemas import TranslationRequest, TranslationResponse, HealthResponse, EntitySearchResponse
from services.ner_service import ner_service
from services.translation_service import translation_service
from services.database import db_service
from config.settings import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info("Starting Japanese Train Announcement Translation API")
    
    # Check services on startup
    if not ner_service.ner_pipeline:
        logger.warning("NER service not fully loaded")
    if not translation_service.model:
        logger.warning("Translation service not fully loaded")
    if not db_service.is_connected():
        logger.warning("Database service not connected")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down API")
    db_service.close()

app = FastAPI(
    title="Japanese Train Announcement Translation API",
    description="API for translating Japanese train announcements to English using NER and machine translation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Japanese Train Announcement Translation API",
        "version": "1.0.0",
        "endpoints": {
            "translate": "/translate",
            "health": "/health",
            "search_entities": "/entities/search"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services_status = {
        "ner_service": ner_service.ner_pipeline is not None,
        "translation_service": translation_service.model is not None,
        "database_service": db_service.is_connected()
    }
    
    all_healthy = all(services_status.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        services=services_status,
        database_entities_count=len(db_service.entity_cache) if db_service.is_connected() else 0
    )

@app.post("/translate", response_model=TranslationResponse)
async def translate_announcement(request: TranslationRequest):
    """
    Translate Japanese train announcement to English
    
    This endpoint:
    1. Uses NER to identify entities and replace them with placeholders
    2. Translates the sentence using machine translation
    3. Translates individual entities using database lookup, Wikidata, or romanization
    4. Replaces placeholders with translated entities in the final sentence
    """
    try:
        japanese_text = request.text.strip()
        if not japanese_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input text cannot be empty"
            )
        
        logger.info(f"Processing translation request: {japanese_text[:50]}...")
        
        # Step 1: NER processing
        text_with_placeholders, entity_mapping = ner_service.replace_entities_and_map(japanese_text)
        
        logger.info(f"NER found {len(entity_mapping)} entities")
        
        # Step 2: Translation
        english_translation = translation_service.process_translation(
            text_with_placeholders, 
            entity_mapping
        )
        
        logger.info(f"Translation completed successfully")
        
        return TranslationResponse(
            original_text=japanese_text,
            text_with_placeholders=text_with_placeholders,
            entity_mapping=entity_mapping,
            english_translation=english_translation,
            entities_count=len(entity_mapping)
        )
        
    except Exception as e:
        logger.error(f"Error processing translation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation processing failed: {str(e)}"
        )

@app.get("/entities/search", response_model=EntitySearchResponse)
async def search_entities(q: str = "", limit: int = 10):
    """
    Search for entities in the database
    
    Args:
        q: Search query (searches in both Japanese and English)
        limit: Maximum number of results (default: 10, max: 100)
    """
    try:
        if not db_service.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database service not available"
            )
        
        if limit > 100:
            limit = 100
        
        entities = db_service.search_entities(q, limit)
        
        return EntitySearchResponse(
            query=q,
            results=entities,
            count=len(entities)
        )
        
    except Exception as e:
        logger.error(f"Error searching entities: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity search failed: {str(e)}"
        )

@app.post("/entities/add")
async def add_entity(japanese: str, english: str):
    """
    Add new entity translation to database
    
    Args:
        japanese: Japanese entity name
        english: English translation
    """
    try:
        if not db_service.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database service not available"
            )
        
        if not japanese.strip() or not english.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both Japanese and English text must be provided"
            )
        
        success = db_service.add_entity(japanese.strip(), english.strip())
        
        if success:
            return {"message": "Entity added successfully", "japanese": japanese, "english": english}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to add entity to database"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding entity: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add entity: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
