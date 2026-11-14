
import json
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

from preprocess import clean_text


def load_data(path="data/intents.json"):
    with open(path, encoding="utf-8") as f:
        intents = json.load(f)["intents"]

    X, y = [], []

    for intent in intents:
        for pattern in intent["patterns"]:
            cleaned = clean_text(pattern)

            if cleaned.strip():
                X.append(cleaned)
                y.append(intent["tag"])

    return X, y


def build_model():
    features = FeatureUnion([
        (
            "word_features",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                sublinear_tf=True,
                strip_accents="unicode",
                min_df=1
            )
        ),
        (
            "character_features",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                sublinear_tf=True,
                min_df=1
            )
        )
    ])

    model = Pipeline([
        ("features", features),
        ("clf", LogisticRegression(
            C=2,
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    return model


if __name__ == "__main__":
    X, y = load_data()

    model = build_model()
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/intent_model.joblib")

    print(f"Training examples: {len(X)}")
    print(f"Number of intents: {len(set(y))}")
    print("Intent labels:", sorted(set(y)))
    print("Model saved successfully!")
