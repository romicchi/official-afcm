# lazy_summarizer.py

from transformers import BartForConditionalGeneration, BartTokenizer

class LazyBartSummarizer:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(self):
        if self.model is None:
            # Change the model to a smaller version (facebook/bart-base)
            self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
            self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    def summarize(self, text):
        self.load_model()

        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

        # Generate the summary
        summary_ids = self.model.generate(inputs["input_ids"], max_length=150, min_length=50, length_penalty=2.0, num_beams=4)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary
