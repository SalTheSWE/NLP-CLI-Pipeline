from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

import pandas as pd
import numpy as np
import ast

from utils.data_handler import save_pickle, save_text, append_command_log


def train_command(csv_path, input_col, output_col, test_size, dataset, models, save_model, text_col):
    # Load data (CSV-only here)
    if not csv_path:
        raise ValueError("csv_path is required.")

    df = pd.read_csv(csv_path)

    if input_col not in df.columns:
        raise ValueError(f"input_col '{input_col}' not found.")
    if output_col not in df.columns:
        raise ValueError(f"output_col '{output_col}' not found.")

    # Parse embeddings column into a numeric matrix X
    X_list = []
    y_list = []
    for i, (x, y) in enumerate(zip(df[input_col].tolist(), df[output_col].tolist())):
        if pd.isna(x):
            continue
        if isinstance(x, (list, tuple, np.ndarray)):
            vec = np.array(x, dtype=float)
        else:
            s = str(x).strip()
            try:
                vec = np.array(ast.literal_eval(s), dtype=float)
            except Exception:
                s = s.strip("[]")
                parts = [p for p in s.split(",") if p.strip()]
                vec = np.array([float(p) for p in parts], dtype=float)
        if vec.ndim != 1:
            raise ValueError(f"Embedding at row {i} is not 1D.")
        X_list.append(vec)
        y_list.append(str(y))

    if not X_list:
        raise ValueError("No valid embeddings found in input_col.")

    X = np.vstack(X_list)
    y = np.array(y_list)

    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)

    # Decide which models to run (defaults only, no custom params)
    models_norm = [str(m).lower() for m in models]
    if len(models_norm) == 1 and models_norm[0] == "all":
        models_norm = ["knn", "lr", "rf"]

    run = []
    for m in models_norm:
        if m in ("knn",):
            run.append(("K-Nearest Neighbors", KNeighborsClassifier()))
        elif m in ("lr", "logreg", "logistic", "logisticregression"):
            run.append(("Logistic Regression", LogisticRegression()))
        elif m in ("rf", "randomforest", "random_forest"):
            run.append(("Random Forest", RandomForestClassifier()))
        else:
            raise ValueError(f"Unknown model '{m}'. Use knn, lr, rf, or all.")

    # Train + report
    report = []
    report.append(f"# Training Report - {pd.Timestamp.now().date()}\n")
    report.append("## Dataset Info")
    report.append(f"- Total samples: {len(y)}")
    report.append(f"- Train/Test split: {len(y_train)}/{len(y_test)} ({int((1-test_size)*100)}/{int(test_size*100)})")
    report.append(f"- Classes: {len(np.unique(y))}")
    report.append(f"- Features: {X.shape[1]}\n")
    report.append("## Model Performance\n")

    best_name, best_model, best_f1 = None, None, -1.0

    for name, model in run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, preds)

        report.append(f"### {name}")
        report.append(f"- Accuracy:  {acc:.4f}")
        report.append(f"- Precision: {prec:.4f}")
        report.append(f"- Recall:    {rec:.4f}")
        report.append(f"- F1-Score:  {f1:.4f}\n")

        report.append("Confusion Matrix:")
        report.append("```")
        report.append(str(cm))
        report.append("```\n")

        report.append("Classification Report:")
        report.append("```")
        report.append(classification_report(y_test, preds, zero_division=0))
        report.append("```\n")

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    report.append(f"## Best Model: {best_name} ⭐\n")
    report_md = "\n".join(report)

    # Save report to outputs/reports/training_report_[timestamp].md
    report_path = save_text(report_md, None, bucket="reports", base_name="training_report", ext="md", add_timestamp=True)
    append_command_log("reports", f"train report -> {report_path}")
    print("Saved report:", report_path)

    # Save best model if requested (if user passes just a filename, your data_handler should route it into outputs/models/)
    if save_model:
        model_path = save_pickle(best_model, save_model, bucket="models", base_name="best_model", add_timestamp=False)
        append_command_log("models", f"train best_model -> {model_path}")
        print("Saved model:", model_path)

    return
def pipeline_command(csv_path, text_col, label_col, preprocessing, embedding, training, output, save_model, save_report):
    return