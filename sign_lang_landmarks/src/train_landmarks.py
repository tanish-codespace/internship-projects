import pickle
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

CSV = "landmark_data.csv"

def main():
    df = pd.read_csv(CSV)
    print("Samples per sign:\n", df["label"].value_counts())

    # Force plain NumPy arrays (avoids the PyArrow indexing error)
    X = np.asarray(df.drop(columns=["label"]).values, dtype=np.float64)
    y = np.asarray(df["label"].values, dtype=object)

    # Split into train/test
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train a small neural network on the landmark coordinates
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=600,
        random_state=42
    )
    clf.fit(Xtr, ytr)

    # Results
    acc = clf.score(Xte, yte) * 100
    print("\nTest accuracy:", round(acc, 1), "%")
    print(classification_report(yte, clf.predict(Xte)))

    # Save the trained model
    with open("sign_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("Saved -> sign_model.pkl")
    print("Next: streamlit run app_landmarks.py")

if __name__ == "__main__":
    main()