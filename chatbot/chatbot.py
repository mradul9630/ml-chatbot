import os
import joblib

from chatbot.preprocess import preprocess_text
from chatbot.responses import responses

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(
    os.path.join(BASE_DIR, "models", "chatbot_model.pkl")
)

vectorizer = joblib.load(
    os.path.join(BASE_DIR, "models", "vectorizer.pkl")
)

def get_response(user_input):
    processed = preprocess_text(user_input)

    vector = vectorizer.transform([processed])

    intent = model.predict(vector)[0]

    return responses.get(intent, "Sorry, I don't understand.")