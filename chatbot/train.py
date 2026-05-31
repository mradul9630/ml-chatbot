import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from chatbot.preprocess import preprocess_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "intents.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

data = pd.read_csv(DATA_PATH)

data["processed"] = data["text"].apply(preprocess_text)

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["processed"])

y = data["intent"]

model = LogisticRegression()

model.fit(X, y)

joblib.dump(
    model,
    os.path.join(MODEL_DIR, "chatbot_model.pkl")
)

joblib.dump(
    vectorizer,
    os.path.join(MODEL_DIR, "vectorizer.pkl")
)

print("Training Complete")