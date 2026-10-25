import argparse
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from text_utils import clean_text

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/tickets.csv")
parser.add_argument("--text-col", default="text")
parser.add_argument("--label-col", default="queue")
parser.add_argument("--lang", default="en")
args = parser.parse_args()

df = pd.read_csv(args.data)
if "language" in df.columns and args.lang:
    df = df[df["language"] == args.lang]
df = df[[args.text_col, args.label_col]].dropna()
print("Rows used:", len(df))
print(df[args.label_col].value_counts())

df["clean"] = df[args.text_col].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df[args.label_col], test_size=0.2,
    random_state=42, stratify=df[args.label_col])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

results = []
best_name, best_f1, best_pipe = None, -1, None
for name, clf in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000)),
        ("clf", clf),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="weighted", zero_division=0)
    results.append({"Model": name, "Accuracy": acc, "Precision": p,
                    "Recall": r, "F1": f1})
    print(f"{name}: accuracy={acc:.3f} f1={f1:.3f}")
    if f1 > best_f1:
        best_name, best_f1, best_pipe = name, f1, pipe

res = pd.DataFrame(results)
res.to_csv("reports/model_comparison.csv", index=False)

res.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]].plot(
    kind="bar", figsize=(9, 5))
plt.title("Model Comparison")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("reports/model_comparison.png")
plt.close()

best_pred = best_pipe.predict(X_test)
with open("reports/classification_report.txt", "w") as f:
    f.write(f"Best model: {best_name}\n\n")
    f.write(classification_report(y_test, best_pred, zero_division=0))

cm = confusion_matrix(y_test, best_pred, labels=best_pipe.classes_)
ConfusionMatrixDisplay(cm, display_labels=best_pipe.classes_).plot(xticks_rotation=30)
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png")
plt.close()

joblib.dump(best_pipe, "models/model.pkl")
print("Best model:", best_name, "| saved to models/model.pkl")
