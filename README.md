# 🤖 NLP Chatbot (TF-IDF + Logistic Regression)

## Overview
This is a machine learning-based chatbot built using NLP techniques. It classifies user input into predefined intents using TF-IDF vectorization and Logistic Regression, and returns appropriate responses.

## Tech Stack
- Python  
- Scikit-learn  
- NLTK  
- Pandas  
- NumPy  

## Project Structure
NLP-Chatbot/
│
├── chatbot/
│   ├── chatbot.py
│   ├── preprocess.py
│   ├── responses.py
│   ├── train.py
│
├── data/
│   └── intents.csv
│
├── models/
│   ├── chatbot_model.pkl
│   └── vectorizer.pkl
│
├── main.py
├── requirements.txt

## Installation

git clone https://github.com/mradul9630/ml-chatbot.git
cd ml-chatbot

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## Train Model

python -m chatbot.train

## Run Chatbot

python main.py

## Example

You: hello  
Bot: Hello! How can I help you?

You: what is your name  
Bot: I am an NLP Chatbot.

You: thanks  
Bot: You're welcome!

## Model Details
- TF-IDF Vectorization for feature extraction  
- Logistic Regression for intent classification  
- Rule-based response mapping  

## Author
Mradul (GitHub: mradul9630)
