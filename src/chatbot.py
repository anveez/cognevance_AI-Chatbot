import json
import random
import joblib
from preprocess import clean_text


class Chatbot:
    def __init__(self, model_path="models/intent_model.joblib",
                 intents_path="data/intents.json", threshold=0.30):
        self.model = joblib.load(model_path)
        with open(intents_path, encoding="utf-8") as f:
            intents = json.load(f)["intents"]
        self.responses = {i["tag"]: i["responses"] for i in intents}
        self.threshold = threshold

    def predict_intent(self, text):
        probs = self.model.predict_proba([clean_text(text)])[0]
        idx = probs.argmax()
        return self.model.classes_[idx], float(probs[idx])

    def reply(self, text):
        tag, confidence = self.predict_intent(text)
        if confidence < self.threshold:        # not confident -> fallback
            tag = "fallback"
        return random.choice(self.responses[tag]), tag, confidence


if __name__ == "__main__":
    bot = Chatbot()
    print("TechBuddy: Hi! Type 'quit' to exit.")
    while True:
        msg = input("You: ")
        if msg.lower() in {"quit", "exit"}:
            break
        answer, tag, conf = bot.reply(msg)
        print(f"TechBuddy: {answer}   [{tag}, {conf:.2f}]")