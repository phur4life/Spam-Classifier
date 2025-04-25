# spam_classifier.ipynb - Python Notebook Content

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

# Step 2: Load Dataset from CSV file (make sure 'sample_spam.csv' is in your working directory)
data = pd.read_csv("sample_spam.csv")

# Convert labels to binary (ham = 0, spam = 1)
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

# Step 3: Split the Data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label_num'], test_size=0.2, random_state=42
)

# Step 4: Build a Pipeline (TF-IDF + Naive Bayes)
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Step 5: Train the Model
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 7: Save the Model
joblib.dump(model, 'model.pkl')

# Step 8: Try a Custom Prediction
sample_email = ["You have won $1,000 cash prize! Claim now."]
result = model.predict(sample_email)[0]
label = "Spam" if result == 1 else "Ham"
print(f"\nSample Prediction: '{sample_email[0]}' is classified as --> {label}")