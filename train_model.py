import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# STEP 1: Load the dataset
data_path = Path("data/spam.csv")

df = pd.read_csv(data_path)

# STEP 2: Prepare the data
df = df[["label", "text"]].dropna()

df["label"] = df["label"].str.lower().str.strip()
df["text"] = df["text"].astype(str).str.strip()

df = df[df["label"].isin(["spam", "ham"])]
df = df[df["text"] != ""]

print("Dataset loaded successfully!")
print("Total messages:", len(df))

print("\nMessage distribution:")
print(df["label"].value_counts())

# STEP 3: Separate messages and labels
X = df["text"]
y = df["label"]

# STEP 4: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

# STEP 5: Create the ML pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        MultinomialNB()
    )
])

# STEP 6: Train the model
print("\nTraining the model...")

model.fit(X_train, y_train)

print("Training completed!")

# STEP 7: Test the model
predictions = model.predict(X_test)

print("\nModel Accuracy:")
print(round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=["ham", "spam"],
        target_names=["Legitimate", "Spam"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=["ham", "spam"]
    )
)

# STEP 8: Save the trained model
Path("models").mkdir(exist_ok=True)

joblib.dump(
    model,
    "models/spam_model.joblib"
)

print("\nModel saved successfully!")
print("Location: models/spam_model.joblib")