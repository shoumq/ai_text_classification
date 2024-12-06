import torch
from transformers import BertTokenizer, BertForSequenceClassification
import telebot

TOKEN = '7873240309:AAE2JMHbW4s-1cQ71IN6OyX2Cl3q0xB9h6Q'
bot = telebot.TeleBot(TOKEN)

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

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Добро пожаловать! Введите текст для распознания")

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    model_name = "DeepPavlov/rubert-base-cased"
    sentiment_model = SentimentModel(model_name=model_name, num_labels=2)
    sentiment_analyzer = SentimentAnalyzer(model=sentiment_model)
    sentiment = sentiment_analyzer.analyze_sentiment(message.text)
    bot.reply_to(message, f"{sentiment}")

if __name__ == "__main__":
    bot.polling()