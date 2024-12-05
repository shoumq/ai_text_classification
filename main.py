import torch
from transformers import BertTokenizer, BertForSequenceClassification

class SentimentModel:
    def __init__(self, model_name: str, num_labels: int):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)

    def predict(self, text: str) -> int:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

        return predicted_class


class SentimentAnalyzer:
    def __init__(self, model: SentimentModel):
        self.model = model

    def analyze_sentiment(self, text: str) -> str:
        predicted_class = self.model.predict(text)
        return "Положительное" if predicted_class == 1 else "Отрицательное"


if __name__ == "__main__":
    model_name = "DeepPavlov/rubert-base-cased"
    sentiment_model = SentimentModel(model_name=model_name, num_labels=2)

    with open('data.csv', 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            sentiment_analyzer = SentimentAnalyzer(model=sentiment_model)
            sentiment = sentiment_analyzer.analyze_sentiment(line.strip())
            print(f"Отзыв №{i}: {line.strip()} - {sentiment}")
