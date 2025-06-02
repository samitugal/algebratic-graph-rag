from transformers import pipeline
import torch
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["TRANSFORMERS_CACHE"] = "./models_cache"
os.environ["HF_HOME"] = "./models_cache"

class SimpleSummarizer:
    _instance = None
    _pipe = None
    
    def __new__(cls, model_name="facebook/bart-large-cnn"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name="facebook/bart-large-cnn"):
        if self._pipe is None:
            device = "cpu"
            self._pipe = pipeline(
                "summarization",
                model=model_name,
                tokenizer=model_name,
                device=device,
                truncation=True,
                max_length=1024
            )
            self.tokenizer = self._pipe.tokenizer
    
    @property
    def pipe(self):
        return self._pipe

    def summarize(self, text: str, max_length: int = 50, min_length: int = 20) -> str:
        if len(text.split()) < min_length:
            return text.strip()

        tokens = self.tokenizer.encode(text, truncation=True, max_length=512)
        if len(tokens) > 512:
            text = self.tokenizer.decode(tokens[:512], skip_special_tokens=True)

        words = text.split()
        if len(words) > 200:
            text = " ".join(words[:200])

        try:
            result = self.pipe(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            return result[0]["summary_text"]
        except Exception as e:
            print(f"[Warning] Summarization failed: {e}")
            return text.strip()[:200] + "..."
