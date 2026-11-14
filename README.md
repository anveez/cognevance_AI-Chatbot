
# TechBuddy – AI Chatbot Using NLP

## Project Overview

TechBuddy is an AI-powered chatbot developed using Natural Language Processing (NLP) and Machine Learning. It understands user messages, identifies their intent, and generates appropriate responses.

The chatbot uses text preprocessing, TF-IDF feature extraction, and Logistic Regression for intent classification.

## Objectives

- Develop an intelligent chatbot using NLP techniques.
- Create a conversational training dataset.
- Preprocess text using NLTK.
- Train a machine learning model for intent recognition.
- Generate responses based on predicted intents.
- Build an interactive chatbot interface using Streamlit.
- Evaluate model performance using standard metrics.

## Technologies Used

- Python
- NLTK
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Project Structure

```text
TechBuddy/
│
├── data/
│   └── intents.json
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── chatbot.py
│   └── evaluate.py
│
├── models/
│   └── intent_model.joblib
│
├── charts/
│   └── confusion_matrix.png
│
├── screenshots/
│
├── docs/
│   └── evaluation_results.txt
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Project Workflow

1. **Data Collection:** Created a conversational dataset containing 400 example patterns across 10 different intents.
2. **Text Preprocessing:** Used NLTK for lowercasing, contraction expansion, tokenization, and lemmatization.
3. **Feature Extraction:** Combined word-level and character-level TF-IDF features.
4. **Model Training:** Trained a Logistic Regression classifier to recognize user intents.
5. **Response Generation:** Generated responses based on the predicted intent.
6. **User Interface:** Developed an interactive chatbot using Streamlit.
7. **Evaluation:** Evaluated performance using accuracy, precision, recall, F1-score, and cross-validation.

## Supported Intents

- Greeting
- Goodbye
- Thanks
- Name
- Help
- Hours
- Contact
- Pricing
- Refund
- Joke

## Model Performance

| Evaluation Metric | Result |
|---|---:|
| Test Accuracy | 89.0% |
| Macro F1-score | 88.8% |
| Weighted F1-score | 88.8% |
| 5-Fold Cross-Validation Mean Accuracy | 84.5% |

The model was evaluated using a stratified train-test split, with 75% of the data used for training and 25% for testing.

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/anveez/cognevance_AI-Chatbot
```

### 2. Navigate to the project folder

```bash
cd cognevance_AI-Chatbot
```

### 3. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the model

```bash
python src/train.py
```

### 6. Run evaluation

```bash
python src/evaluate.py
```

### 7. Launch the chatbot

```bash
streamlit run app.py
```

## Expected Outputs

- Predicted user intent.
- Appropriate chatbot responses.
- Interactive Streamlit chatbot interface.
- Model evaluation metrics.
- Confusion matrix visualization.

## Future Enhancements

- Integrate a larger conversational dataset.
- Improve handling of unknown user queries.
- Add conversation history and context awareness.
- Explore deep learning-based NLP models.
- Deploy the chatbot online.

## Project Details

**Project Level:** Level 2 – Intermediate

**Project Title:** AI Chatbot Using NLP

**Project Name:** TechBuddy

**Developed as part of:** Cognevance Internship Project
