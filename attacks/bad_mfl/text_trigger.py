from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Sequence

import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from attacks.bad_mfl.config import TriggerConcept


class TextCandidateGenerator:
    def __init__(self, model_name: str, enabled: bool, allow_remote_model_download: bool=False):
        self.enabled = bool(enabled)
        self.tokenizer = None
        self.model = None
        self.device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
        if not self.enabled:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=not bool(allow_remote_model_download)
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            local_files_only=not bool(allow_remote_model_download)
        ).to(self.device)
        self.model.eval()

    def paraphrase(self, text: str, trigger_phrase: str, num_candidates: int) -> List[str]:
        if not self.enabled or self.model is None or self.tokenizer is None:
            return []
        prompt = (
            f"Rewrite the sentence naturally. Keep the meaning close. Include the exact phrase '{trigger_phrase}' exactly once. "
            f"Sentence: {text}"
        )
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True).to(self.device)
        with __import__('torch').no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=48,
                num_return_sequences=max(1, int(num_candidates)),
                num_beams=max(4, int(num_candidates)),
                do_sample=False
            )
        results = []
        for output in outputs:
            decoded = self.tokenizer.decode(output, skip_special_tokens=True).strip()
            if decoded:
                results.append(decoded)
        return results


class TextTriggerBuilder:
    def __init__(self, concept: TriggerConcept, text_generator: TextCandidateGenerator, cache_root: Path | None=None):
        self.concept = concept
        self.text_generator = text_generator
        self.cache_root = cache_root
        if self.cache_root is not None:
            self.cache_root.mkdir(parents=True, exist_ok=True)

    def build_candidates(self, sample_key: str, text: str, token_span, num_candidates: int) -> List[str]:
        safe_key = re.sub(r'[^0-9A-Za-z._-]+', '_', str(sample_key)).strip('._')
        if not safe_key:
            safe_key = 'sample'
        cache_path = None if self.cache_root is None else self.cache_root.joinpath(f'{safe_key}_{self.concept.name}.json')
        if cache_path is not None and cache_path.exists():
            with open(str(cache_path), 'r', encoding='utf-8') as f:
                cached = json.load(f)
            if isinstance(cached, list) and cached:
                return cached

        base_candidates = []
        token_replaced = self._replace_token(text, token_span, self.concept.trigger_word)
        base_candidates.append(self._append_phrase(token_replaced, self.concept.trigger_phrase))
        base_candidates.append(self._append_phrase(text, self.concept.trigger_phrase))
        for hint in self.concept.paraphrase_hints:
            base_candidates.append(self._append_phrase(self._replace_token(text, token_span, hint), self.concept.trigger_phrase))

        paraphrases = self.text_generator.paraphrase(base_candidates[0], self.concept.trigger_phrase, num_candidates)
        base_candidates.extend(paraphrases)

        filtered = []
        seen = set()
        for candidate in base_candidates:
            normalized = self._normalize(candidate)
            if not normalized or normalized in seen:
                continue
            if not self._contains_trigger(normalized):
                continue
            if self._normalize(text) == normalized:
                continue
            if self._looks_broken(normalized):
                continue
            filtered.append(candidate.strip())
            seen.add(normalized)

        if not filtered:
            filtered = [self._append_phrase(token_replaced, self.concept.trigger_phrase)]

        if cache_path is not None:
            with open(str(cache_path), 'w', encoding='utf-8') as f:
                json.dump(filtered, f, ensure_ascii=False, indent=2)
        return filtered

    def select_best_candidate(self, clip_helper, poisoned_image, original_text: str, candidates: Sequence[str]) -> str:
        if len(candidates) == 1:
            return str(candidates[0])
        image_sims = clip_helper.similarity([poisoned_image] * len(candidates), list(candidates))
        scored = []
        original_norm = self._normalize(original_text)
        for sim, candidate in zip(image_sims.tolist(), candidates):
            candidate_norm = self._normalize(candidate)
            lexical_gain = 0.0 if candidate_norm == original_norm else min(0.20, abs(len(candidate_norm) - len(original_norm)) * 0.002)
            scored.append((float(sim) + lexical_gain, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        return str(scored[0][1])

    def _replace_token(self, text: str, token_span, replacement: str) -> str:
        if token_span is None or getattr(token_span, 'end', 0) <= getattr(token_span, 'start', 0):
            return text
        return text[:token_span.start] + replacement + text[token_span.end:]

    @staticmethod
    def _append_phrase(text: str, phrase: str) -> str:
        text = str(text).strip()
        if phrase.lower() in text.lower():
            return text
        suffix = '' if text.endswith(('.', '!', '?')) else '.'
        if not text:
            return phrase.capitalize() + '.'
        return f'{text}{suffix} {phrase}.'

    def _contains_trigger(self, text: str) -> bool:
        text_l = text.lower()
        return self.concept.trigger_phrase.lower() in text_l or self.concept.trigger_word.lower() in text_l

    @staticmethod
    def _looks_broken(text: str) -> bool:
        if '??' in text or '@@' in text:
            return True
        if len(re.findall(r'\b\w+\b', text)) < 3:
            return True
        return False

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r'\s+', ' ', str(text).strip().lower())
