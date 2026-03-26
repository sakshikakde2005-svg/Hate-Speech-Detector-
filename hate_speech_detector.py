# Hate Speech Detector Project

import pandas as pd
import numpy as np
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Download tokenizer
nltk.download('punkt')

# Load dataset
data = pd.read_csv("hate_speech_dataset.csv")

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):

    text = text.lower()

    # remove links
    text = re.sub(r"http\S+", "", text)

    # remove special characters
    text = re.sub(r"[^a-zA-Z]", " ", text)

    return text

# Apply cleaning
data["clean_comment"] = data["comment"].apply(clean_text)

# -----------------------------
# Convert Text → Numbers
# -----------------------------
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["clean_comment"])

# -----------------------------
# Encode Labels
# -----------------------------
encoder = LabelEncoder()

y = encoder.fit_transform(data["label"])

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = LogisticRegression()

model.fit(X_train, y_train)

# -----------------------------
# Evaluate Model
# -----------------------------
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# -----------------------------
# Test Custom Comment
# -----------------------------
comment = input("\nEnter a comment: ")

comment_clean = clean_text(comment)

comment_vector = vectorizer.transform([comment_clean])

prediction = model.predict(comment_vector)

result = encoder.inverse_transform(prediction)

print("\nPrediction:", result[0])