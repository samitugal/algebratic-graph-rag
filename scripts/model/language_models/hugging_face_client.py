from transformers import pipeline
import torch

class SimpleSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name,
            device=device,
            truncation=True
        )

    def summarize(self, text: str, max_length: int = 120, min_length: int = 30) -> str:
        if len(text.split()) < min_length:
            return text.strip()

        try:
            result = self.pipe(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            return result[0]["summary_text"]
        except Exception as e:
            print(f"[Warning] Summarization failed: {e}")
            return text.strip()
