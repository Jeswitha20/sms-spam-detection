# train_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from catboost import CatBoostClassifier
import joblib

# Load and Clean Dataset
data = pd.read_csv("sms_dataset.csv", usecols=[0, 1], encoding='ISO-8859-1')  # Proper encoding and column selection
data.columns = ['label', 'message']  # Rename columns
data.dropna(inplace=True)  # Remove any missing rows
data['label'] = data['label'].map({'ham': 0, 'spam': 1})  # Map labels to binary values

# Feature and Target Variables
X = data['message']
y = data['label']

# Text Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train SVC
svc_model = SVC(probability=True)
svc_model.fit(X_train, y_train)

# Train CatBoost
catboost_model = CatBoostClassifier(verbose=0)
catboost_model.fit(X_train, y_train)

# Save Models and Vectorizer
joblib.dump(svc_model, "svc_model.pkl")
joblib.dump(catboost_model, "catboost_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Models and vectorizer saved!")
