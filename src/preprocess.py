
import re
import nltk
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
for pkg in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Preprocess text:
    1. Convert to lowercase
    2. Expand common contractions
    3. Remove punctuation
    4. Tokenize
    5. Lemmatize words

    Stopwords are intentionally retained because they can
    carry useful meaning in short chatbot messages.
    """

    text = text.lower()

    contractions = {
        "what's": "what is",
        "who's": "who is",
        "how's": "how is",
        "i'm": "i am",
        "i've": "i have",
        "i'll": "i will",
        "don't": "do not",
        "can't": "cannot",
        "you're": "you are",
        "it's": "it is",
        "that's": "that is"
    }

    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)

    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = nltk.word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.strip()
    ]

    return " ".join(tokens)


if __name__ == "__main__":
    examples = [
        "What are your opening hours?!",
        "Can I get my money back?",
        "How can I contact you?",
        "What is your subscription price?",
        "I want to say goodbye."
    ]

    for sentence in examples:
        print(sentence, "->", clean_text(sentence))
