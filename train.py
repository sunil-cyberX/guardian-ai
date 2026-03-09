import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples=30000):
    np.random.seed(42)
    amounts = np.random.exponential(5000, n_samples)
    merchant_ids = np.random.randint(0, 1000, n_samples)
    user_ids = np.random.randint(0, 1000, n_samples)
    categories = np.random.randint(0, 100, n_samples)
    channels = np.random.randint(0, 2, n_samples)
    fraud = (
        (amounts > 50000) |
        (np.random.random(n_samples) < 0.02)
    ).astype(int)
    X = np.column_stack([amounts, merchant_ids, user_ids, categories, channels])
    return X, fraud

def train(n_samples=30000):
    logger.info(f"Generating {n_samples} samples...")
    X, y = generate_synthetic_data(n_samples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    logger.info("Training model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    logger.info(f"Accuracy: {score:.4f}")
    os.makedirs("/app/models", exist_ok=True)
    model_path = os.environ.get("MODEL_PATH", "/app/models/guardian_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    logger.info(f"Model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--samples", type=int, default=30000)
    args = parser.parse_args()
    train(args.samples)
