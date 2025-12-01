# train_model.py  (LIGHTWEIGHT HYBRID)
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# -----------------------------
# Config
# -----------------------------
DATA_FAKE = "data/Fake.csv"
DATA_TRUE = "data/True.csv"
SAMPLE_SIZE = 10000  # reduce if you have low RAM/disk
RANDOM_STATE = 42

# -----------------------------
# Load and prepare data
# -----------------------------
df_fake = pd.read_csv(DATA_FAKE)
df_true = pd.read_csv(DATA_TRUE)

# Keep only necessary columns (some CSVs have 'text' or 'title')
# Adjust if column names differ
df_fake['label'] = 1
df_true['label'] = 0
df = pd.concat([df_fake[['text', 'label']], df_true[['text','label']]], ignore_index=True)

# optional: sample to reduce size
if SAMPLE_SIZE:
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)

X = df['text'].fillna("").astype(str)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# -----------------------------
# Vectorizer (fit once)
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=3, ngram_range=(1,2))

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# Train Logistic Regression
# -----------------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)
pred_lr = lr.predict_proba(X_test_tfidf)[:,1]

# -----------------------------
# Train MultinomialNB
# -----------------------------
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
pred_nb = nb.predict_proba(X_test_tfidf)[:,1]

# -----------------------------
# Ensemble: weighted average of probabilities
# -----------------------------
# weight_lr + weight_nb = 1.0
weight_lr = 0.6
weight_nb = 0.4
ensemble_pred_prob = weight_lr * pred_lr + weight_nb * pred_nb
ensemble_pred = (ensemble_pred_prob >= 0.5).astype(int)

print("Ensemble Accuracy:", accuracy_score(y_test, ensemble_pred))
print(classification_report(y_test, ensemble_pred))

# -----------------------------
# Save models and vectorizer
# -----------------------------
import os
os.makedirs("models", exist_ok=True)
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("models/lr_model.pkl", "wb") as f:
    pickle.dump(lr, f)
with open("models/nb_model.pkl", "wb") as f:
    pickle.dump(nb, f)

# Save ensemble weights
with open("models/ensemble_meta.pkl", "wb") as f:
    pickle.dump({'weight_lr': weight_lr, 'weight_nb': weight_nb}, f)

print("Models saved to models/ directory.")
