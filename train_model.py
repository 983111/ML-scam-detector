import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_fscore_support

df = pd.read_csv("scam_dataset.csv")
X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

base = LogisticRegression(max_iter=1000)
model = CalibratedClassifierCV(base, method="sigmoid")
model.fit(X_train, y_train)

pred = model.predict(X_test)

p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="binary")

print(f"Precision: {p:.3f}")
print(f"Recall:    {r:.3f}")
print(f"F1-score:  {f:.3f}")

joblib.dump(model, "scam_detector_model.pkl")
print("✅ Model saved")
