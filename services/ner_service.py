import re
import logging
from typing import Dict, Tuple, List
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

logger = logging.getLogger(__name__)

class NERService:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        self._load_model()
        self._setup_prefixes_suffixes()
        
    def _load_model(self):
        """Load NER model and tokenizer"""
        try:
            BASE_TOK = "cl-tohoku/bert-base-japanese-v3"
            MODEL = "knosing/japanese_ner_model"
            
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_TOK, use_fast=True)
            self.model = AutoModelForTokenClassification.from_pretrained(MODEL)
            
            self.ner_pipeline = pipeline(
                "token-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
            )
            
            logger.info("NER model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading NER model: {e}")
            
    def _setup_prefixes_suffixes(self):
        """Setup prefixes and suffixes to strip from entities"""
        self.PREFIXES_TO_STRIP = sorted([
            "京都市営地下鉄",
            "名古屋市営地下鉄",
            "東京メトロ",
            "東急",
            "都営",
            "快速",
            "JR",
            "快速アクティー",
            "アクティー"
        ], key=len, reverse=True)

        self.SUFFIXES_TO_STRIP = sorted([
            "鉄道大雄山線",
            "アーバンパークライン",
            "ディズニーリゾートライン",
            "エクスプレス",
            "スカイライナー",
            "ニューシャトル",
            "モノレール",
            "リゾートライン",
            "市営地下鉄ブルーライン",
            "地下鉄ブルーライン",
            "ブルーライン",
            "ライン",
            "新幹線",
            "本線",
            "空港線",
            "環状線",
            "地下鉄",
            "メトロ線",
            "方面行き",
            "方面",
            "行き",
            "鉄道",
            "線",
            "駅",
            "号",
            "シーサイド",
            "ヒカリエShinQs前",
            "三丁目",
        ], key=len, reverse=True)
        
        self.TAGS_TRAIN_ANNOUNCEMENT = ["地名","施設名","法人名","製品名","その他の組織名"]
        
    def strip_affixes_with_remainder(self, text_to_strip: str) -> Tuple[str, str, str]:
        """
        Remove defined prefixes and suffixes from a string.
        Returns: (stripped_text, found_prefix, found_suffix)
        """
        original_text = text_to_strip
        stripped_text = text_to_strip
        found_prefix_str = ""
        found_suffix_str = ""

        # Remove suffix first
        temp_stripped_text_for_suffix = stripped_text

        # 1. Special handling for "号" if it follows a number
        match_gou = re.search(r'([\d０-９]+)号$', temp_stripped_text_for_suffix)
        if match_gou:
            found_suffix_str = match_gou.group(0)
            stripped_text = temp_stripped_text_for_suffix[:-len(found_suffix_str)]
        else:
            # 2. Handle other suffixes
            for suffix in self.SUFFIXES_TO_STRIP:
                if suffix == "号":
                    continue 
                if temp_stripped_text_for_suffix.endswith(suffix):
                    if len(suffix) > len(found_suffix_str):
                        found_suffix_str = suffix

            if found_suffix_str:
                stripped_text = temp_stripped_text_for_suffix[:-len(found_suffix_str)]

        # Remove prefix
        temp_stripped_text_for_prefix = stripped_text
        for prefix in self.PREFIXES_TO_STRIP:
            if temp_stripped_text_for_prefix.startswith(prefix):
                if len(prefix) > len(found_prefix_str):
                    found_prefix_str = prefix

        if found_prefix_str:
            stripped_text = stripped_text[len(found_prefix_str):]

        # Ensure result is not empty
        if not stripped_text and (found_prefix_str or found_suffix_str):
            return original_text, "", ""
        elif not stripped_text and not (found_prefix_str or found_suffix_str):
            return original_text, "", ""

        return stripped_text, found_prefix_str, found_suffix_str

    def restore_offset(self, text: str, span: str, used_pos: defaultdict) -> Tuple[int, int]:
        """Restore offset for entity spans"""
        span_clean = span.replace(" ", "")
        for m in re.finditer(re.escape(span_clean), text):
            s = m.start()
            if s >= used_pos[span_clean]:
                used_pos[span_clean] = s + 1
                return s, m.end()
        return None, None

    def replace_entities_and_map(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Replace entities in text with placeholders and return mapping
        
        Returns:
            Tuple of (text_with_placeholders, placeholder_to_entity_mapping)
        """
        if not self.ner_pipeline:
            logger.error("NER pipeline not loaded")
            return text, {}
            
        try:
            ents_raw = [
                e for e in self.ner_pipeline(text)
                if e["entity_group"] in self.TAGS_TRAIN_ANNOUNCEMENT
            ]
            
            if not ents_raw:
                return text, {}

            ents_raw.sort(key=lambda e: -len(e["word"].replace(" ", "")))

            occupied = set()
            spans_info_list = []
            used_pos = defaultdict(int)

            for e in ents_raw:
                raw_word_from_ner = e["word"]
                span_txt_cleaned_for_search = raw_word_from_ner.replace(" ", "")
                s_ner, t_ner = e.get("start"), e.get("end")
                current_s, current_t = None, None

                if s_ner is not None and t_ner is not None and s_ner <= t_ner and t_ner <= len(text):
                    if text[s_ner:t_ner].replace(" ", "") == span_txt_cleaned_for_search:
                        current_s, current_t = s_ner, t_ner
                    else:
                        s_restored, t_restored = self.restore_offset(text, span_txt_cleaned_for_search, used_pos)
                        if s_restored is not None:
                            current_s, current_t = s_restored, t_restored
                else:
                    s_restored, t_restored = self.restore_offset(text, span_txt_cleaned_for_search, used_pos)
                    if s_restored is not None:
                        current_s, current_t = s_restored, t_restored

                if current_s is None:
                    continue
                if any(i in occupied for i in range(current_s, current_t)):
                    continue

                # Extend if there's "号" right after and entity ends with number
                if current_t < len(text):
                    if text[current_t] == "号" and re.search(r'[\d０-９]$', text[current_s:current_t]):
                        current_t += 1

                spans_info_list.append({"start": current_s, "end": current_t, "original_text": text[current_s:current_t]})
                occupied.update(range(current_s, current_t))

            if not spans_info_list:
                return text, {}

            spans_info_list.sort(key=lambda x: x["start"])

            # Merge adjacent spans
            merged_spans_final_info = []
            if spans_info_list:
                current_merged_span_data = {"start": spans_info_list[0]["start"], "end": spans_info_list[0]["end"]}
                for i in range(1, len(spans_info_list)):
                    current_span_data = spans_info_list[i]
                    if current_span_data["start"] == current_merged_span_data["end"]:
                        current_merged_span_data["end"] = current_span_data["end"]
                    else:
                        current_merged_span_data["original_text"] = text[current_merged_span_data["start"]:current_merged_span_data["end"]]
                        merged_spans_final_info.append(current_merged_span_data)
                        current_merged_span_data = {"start": current_span_data["start"], "end": current_span_data["end"]}
                current_merged_span_data["original_text"] = text[current_merged_span_data["start"]:current_merged_span_data["end"]]
                merged_spans_final_info.append(current_merged_span_data)

            entity_to_ph = {}
            replacements_for_text_build = []
            ph_idx = 1

            for m_data in merged_spans_final_info:
                original_text_in_span = m_data["original_text"]
                core_entity, stripped_prefix, stripped_suffix = self.strip_affixes_with_remainder(original_text_in_span)

                entity_key_for_map = core_entity

                if entity_key_for_map not in entity_to_ph:
                    entity_to_ph[entity_key_for_map] = f"[PH{ph_idx}]"
                    ph_idx += 1

                placeholder_tag = entity_to_ph[entity_key_for_map]
                actual_prefix_part = ""
                actual_suffix_part = ""

                if original_text_in_span.startswith(stripped_prefix) and stripped_prefix:
                    actual_prefix_part = stripped_prefix
                if original_text_in_span.endswith(stripped_suffix) and stripped_suffix:
                    actual_suffix_part = stripped_suffix

                expected_core_from_original = original_text_in_span
                if actual_prefix_part:
                    expected_core_from_original = expected_core_from_original[len(actual_prefix_part):]
                if actual_suffix_part:
                    expected_core_from_original = expected_core_from_original[:-len(actual_suffix_part)]

                if expected_core_from_original == core_entity:
                    text_to_replace_segment_with = actual_prefix_part + placeholder_tag + actual_suffix_part
                else:
                    reconstructed_original = stripped_prefix + core_entity + stripped_suffix
                    if reconstructed_original == original_text_in_span:
                        text_to_replace_segment_with = stripped_prefix + placeholder_tag + stripped_suffix
                    elif core_entity + stripped_suffix == original_text_in_span and not stripped_prefix:
                        text_to_replace_segment_with = placeholder_tag + stripped_suffix
                    elif stripped_prefix + core_entity == original_text_in_span and not stripped_suffix:
                        text_to_replace_segment_with = stripped_prefix + placeholder_tag
                    else:
                        text_to_replace_segment_with = placeholder_tag

                replacements_for_text_build.append((m_data["start"], m_data["end"], text_to_replace_segment_with))

            new_text, last_processed_pos = "", 0
            replacements_for_text_build.sort(key=lambda x: x[0])

            for s_replace, e_replace, tag_to_insert in replacements_for_text_build:
                new_text += text[last_processed_pos:s_replace] + tag_to_insert
                last_processed_pos = e_replace
            new_text += text[last_processed_pos:]

            ph_to_entity_map = {ph: entity for entity, ph in entity_to_ph.items()}

            logger.info(f"Processed NER: found {len(ph_to_entity_map)} entities")
            return new_text, ph_to_entity_map
            
        except Exception as e:
            logger.error(f"Error in NER processing: {e}")
            return text, {}

# Global NER service instance
ner_service = NERService()
