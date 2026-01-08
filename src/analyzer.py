from transformers import pipeline
from typing import List, Dict, Any


class NLPAnalyzer:
    def __init__(self, device: int = -1):
        print("Loading NLP models... This may take a minute on first run.")

        # Sentiment Analysis
        self.sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            tokenizer="distilbert-base-uncased",
            truncation=True,
            max_length=512,
            device=device
        )

        # Named Entity Recognition
        self.ner = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            tokenizer="bert-base-cased",
            device=device
        )

        # Zero-Shot Classification
        self.zero_shot = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            truncation=True,
            max_length=512,
            device=device
        )

        # Text Summarization
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            truncation=True,
            device=device
        )

        print("All NLP models loaded successfully!")

   
    # Sentiment
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        result = self.sentiment(text)[0]
        return {
            "label": result["label"],
            "confidence": float(result["score"])
        }

    # NER (with sub-token merge)
    def extract_entities(self, text: str) -> Dict[str, Any]:
        raw_results = self.ner(text)
        merged_entities = self._merge_subtokens(raw_results)
        return {"entities": merged_entities}

    def _merge_subtokens(self, ner_results):
        merged_entities = []
        current_entity = None

        for ent in ner_results:
            word = ent["word"]
            label = ent["entity_group"]
            score = ent["score"]

            if word.startswith("##"):
                if current_entity:
                    current_entity["text"] += word[2:]  # remove ##
                    
                    current_entity["confidence"] = max(
                        current_entity["confidence"], score
                    )
            else:
                if current_entity:
                    merged_entities.append(current_entity)

                current_entity = {
                    "text": word,
                    "type": label,
                    "confidence": float(score)
                }

        if current_entity:
            merged_entities.append(current_entity)

        return merged_entities

    # Zero-shot
    def classify_zero_shot(self, text: str, categories: List[str]) -> Dict[str, Any]:
        result = self.zero_shot(text, categories)
        scores = []

        for label, score in zip(result["labels"], result["scores"]):
            scores.append({
                "category": label,
                "score": float(score)
            })

        return {
            "top_category": scores[0]["category"],
            "all_scores": scores
        }

    # Summarization
    def summarize(self, text: str, max_length: int = 130) -> Dict[str, Any]:
        if len(text.split()) < 50:
            return {"summary": text, "note": "Text too short for summarization"}

        result = self.summarizer(
            text,
            max_length=max_length,
            min_length=30,
            do_sample=False
        )[0]

        return {
            "summary": result["summary_text"],
            "original_words": len(text.split()),
            "summary_words": len(result["summary_text"].split())
        }

    # Full Pipeline
    def analyze_all(self, text: str, categories: List[str]) -> Dict[str, Any]:
        return {
            "sentiment": self.analyze_sentiment(text),
            "entities": self.extract_entities(text),
            "classification": self.classify_zero_shot(text, categories),
            "summary": self.summarize(text)
        }
