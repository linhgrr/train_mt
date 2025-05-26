import re
import logging
import torch
import jaconv
import requests
from typing import Dict, List, Optional
from fugashi import Tagger
from services.database import db_service
from transformers import MarianTokenizer, MarianMTModel
from peft import PeftModel, PeftConfig


logger = logging.getLogger(__name__)

class TranslationService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.tagger = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_translation_model()
        self._load_tagger()
        self._setup_suffix_mapping()
        
    def _load_translation_model(self):
        """Load translation model with adapter (PEFT)"""
        try:
            adapter_repo = "linhdzqua148/opus-mt-ja-en-railway-7"

            # Load adapter config to find base model
            peft_config = PeftConfig.from_pretrained(adapter_repo)

            # Load base MarianMT model
            base_model = MarianMTModel.from_pretrained(peft_config.base_model_name_or_path)

            # Load tokenizer (from adapter repo if tokenizer was customized)
            self.tokenizer = MarianTokenizer.from_pretrained(adapter_repo)

            # Load adapter on top of base model
            self.model = PeftModel.from_pretrained(base_model, adapter_repo)

            # Move model to appropriate device
            self.model = self.model.to(self.device)

            logger.info(f"Translation model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading translation model: {e}")

            
    def _load_tagger(self):
        """Load Fugashi tagger for romanization"""
        try:
            self.tagger = Tagger()
            self.tagger("テスト")  # Test tagger
            logger.info("Fugashi Tagger initialized")
        except Exception as e:
            logger.error(f"Error initializing Fugashi Tagger: {e}")
            self.tagger = None
            
    def _setup_suffix_mapping(self):
        """Setup suffix mapping for entity translation"""
        self.SUFFIX_MAP = {
            "新幹線": "Shinkansen",
            "本線": "Main Line",
            "線": "Line",
            "駅": "Station",
            "空港": "Airport",
            "方面": "toward",
            "エクスプレス": " Express",
            "号": " No.",
            "終点": "Terminal",
            "鉄道": "Railway",
            "都市": "Urban",
            "地下鉄": "Subway",
            "メトロ": "Metro",
            "環状線": "Loop Line",
            "モノレール": "Monorail",
            "トラム": "Tram",
            "バス": "Bus",
            "フェリー": "Ferry",
        }
        self.sorted_suffixes = sorted(self.SUFFIX_MAP.keys(), key=len, reverse=True)
        
    def romanize_japanese(self, text: str) -> str:
        """Romanize Japanese text"""
        if not self.tagger:
            return text
            
        try:
            tokens = self.tagger(text)
            result = []
            
            for tok in tokens:
                try:
                    reading = tok.feature[7]
                except (IndexError, TypeError):
                    reading = None

                if not reading or reading == "*" or reading == "UNK":
                    result.append(tok.surface)
                else:
                    try:
                        romaji = jaconv.kata2alphabet(reading)
                        result.append(romaji.capitalize())
                    except Exception:
                        result.append(tok.surface)

            joined_text = " ".join(result).strip()
            
            # Clean up romanization
            joined_text = re.sub(r'\s*[-ー]\s*', 'ー', joined_text)
            joined_text = joined_text.replace('ー ', 'ー').replace(' ー', 'ー')
            
            return joined_text
            
        except Exception as e:
            logger.error(f"Error in romanization: {e}")
            return text
            
    def get_en_name_from_wikidata(self, japanese_name: str) -> Optional[str]:
        """Get English name from Wikidata"""
        search_url = "https://www.wikidata.org/w/api.php"
        search_params = {
            "action": "wbsearchentities",
            "language": "ja",
            "format": "json",
            "search": japanese_name,
            "limit": 5
        }

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(search_url, params=search_params, timeout=8, headers=headers)
            response.raise_for_status()
            results = response.json().get("search", [])

            if not results:
                return None

            entity_id = results[0]["id"]
            data_url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"

            response = requests.get(data_url, timeout=8, headers=headers)
            response.raise_for_status()

            entity_data = response.json()
            en_label = entity_data.get("entities", {}).get(entity_id, {}).get("labels", {}).get("en", {}).get("value")

            return en_label

        except Exception as e:
            logger.debug(f"Wikidata lookup failed for '{japanese_name}': {e}")
            return None
            
    def translate_entity(self, jp_entity: str, use_db_cache: bool = True) -> str:
        """
        Translate Japanese entity to English using prioritized approach
        
        Args:
            jp_entity: Japanese entity to translate
            use_db_cache: Whether to use database cache or query directly
        """
        jp_entity = jp_entity.strip()
        if not jp_entity:
            return ""

        logger.debug(f"Translating entity: '{jp_entity}'")

        # 1. Database lookup
        if db_service.is_connected():
            db_result = db_service.get_entity_translation(jp_entity, use_cache=use_db_cache)
            if db_result:
                logger.debug(f"Found in database: {jp_entity} -> {db_result}")
                return db_result

        # 2. Wikidata lookup
        wikidata_result = self.get_en_name_from_wikidata(jp_entity)
        if wikidata_result:
            logger.debug(f"Found in Wikidata: {jp_entity} -> {wikidata_result}")
            # Optionally save to database for future use
            if db_service.is_connected():
                db_service.add_entity(jp_entity, wikidata_result)
            return wikidata_result

        # 3. Suffix-based translation
        for suffix in self.sorted_suffixes:
            if jp_entity.endswith(suffix):
                base = jp_entity[:-len(suffix)].strip()
                eng_suffix = self.SUFFIX_MAP[suffix]
                
                roman_base = self.romanize_japanese(base) if base else ""
                
                if eng_suffix == "toward":
                    result = ("toward " + roman_base).strip().capitalize()
                elif eng_suffix.startswith(" "):
                    result = roman_base + eng_suffix
                else:
                    result = (roman_base + " " + eng_suffix).strip()
                    
                logger.debug(f"Suffix rule applied: {jp_entity} -> {result}")
                # Save to database for future use
                if db_service.is_connected():
                    db_service.add_entity(jp_entity, result)
                return result

        # 4. Full romanization fallback
        fallback_romaji = self.romanize_japanese(jp_entity)
        logger.debug(f"Fallback romanization: {jp_entity} -> {fallback_romaji}")
        
        # Save to database for future use
        if db_service.is_connected():
            db_service.add_entity(jp_entity, fallback_romaji)
            
        return fallback_romaji
        
    def translate_text(self, text_list: List[str], batch_size: int = 16) -> List[str]:
        """Translate list of Japanese sentences to English"""
        if not isinstance(text_list, list):
            logger.error("translate_text expects a list of strings")
            return []
            
        if not self.model or not self.tokenizer:
            logger.error("Translation model not loaded")
            return [self.romanize_japanese(text) for text in text_list]

        results = []
        
        try:
            for i in range(0, len(text_list), batch_size):
                batch_texts = text_list[i:i + batch_size]
                
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=128
                ).to(self.device)
                
                with torch.no_grad():
                    generated = self.model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=6,
                        length_penalty=0.8,
                    )
                    
                decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                
                # Clean up translations
                cleaned = []
                for text in decoded:
                    text = re.sub(r'\s+([.,!?:;])', r'\1', text).strip()
                    text = re.sub(r'\s+', ' ', text).strip()
                    text = text.replace(' .', '.').replace("' ", "'").replace(" n't", "n't")
                    cleaned.append(text)
                    
                results.extend(cleaned)
                
        except Exception as e:
            logger.error(f"Error in batch translation: {e}")
            # Fallback to romanization
            results.extend([self.romanize_japanese(text) for text in text_list])
            
        return results
        
    def remove_adjacent_duplicate_phrases(self, text: str, max_phrase_len: int = 5) -> str:
        """Remove adjacent duplicate phrases from text"""
        text = re.sub(r'\s+,', ',', text)
        
        for n in range(max_phrase_len, 0, -1):
            pattern = re.compile(
                r'(\b(?:[\w\-\'ōū]+(?:\s+|, ?)){%d}[\w\-\'ōū]+\b)'
                r'(,? \1\b)'
                % (n-1),
                flags=re.IGNORECASE
            )
            while pattern.search(text):
                text = pattern.sub(r'\1', text)
                
        text = re.sub(r'\b(\w+)(,? \1\b)', r'\1', text, flags=re.IGNORECASE)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        return text.strip()
        
    def process_translation(self, japanese_text: str, entity_mapping: Dict[str, str]) -> str:
        """
        Process full translation pipeline
        
        Args:
            japanese_text: Japanese text with placeholders
            entity_mapping: Mapping from placeholders to Japanese entities
            
        Returns:
            Final English translation
        """
        # 1. Translate sentence
        english_sentences = self.translate_text([japanese_text])
        english_sentence = english_sentences[0] if english_sentences else japanese_text
        
        # 2. Translate entities and replace placeholders
        translated_entities = {}
        for placeholder, jp_entity in entity_mapping.items():
            if isinstance(placeholder, str) and placeholder.startswith('[PH') and placeholder.endswith(']'):
                translated_entities[placeholder] = self.translate_entity(jp_entity)
            else:
                logger.warning(f"Invalid placeholder format: {placeholder}")
                translated_entities[str(placeholder)] = str(jp_entity)
        
        # 3. Replace placeholders in translated sentence
        final_sentence = english_sentence
        for placeholder, english_entity in translated_entities.items():
            final_sentence = final_sentence.replace(placeholder, str(english_entity))
        
        # 4. Final cleanup
        final_sentence = re.sub(r'\s+', ' ', final_sentence).strip()
        final_sentence = re.sub(r'\s*([.,!?:;])', r'\1', final_sentence)
        final_sentence = final_sentence.replace(" 's", "'s").replace(" n't", "n't")
        final_sentence = self.remove_adjacent_duplicate_phrases(final_sentence)
        
        logger.info(f"Translation completed: {len(entity_mapping)} entities processed")
        return final_sentence

# Global translation service instance
translation_service = TranslationService()
