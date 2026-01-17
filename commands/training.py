from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

import pandas as pd
import numpy as np
import ast

from utils.data_handler import save_pickle, save_text, append_command_log
from commands.preprocessing import remove_command, stopwords_command, replace_command, all_command
from commands.embedding import tfidfcommand, model2vec_command, word2vec_command, fasttext_command, bert_command, sentence_transformer_command
from utils.data_handler import append_command_log, save_csv

import pickle



def train_command(csv_path, input_col, output_col, test_size, dataset, models, save_model, text_col):
    if not csv_path:
        raise ValueError("csv_path is required.")

    df = pd.read_csv(csv_path)

    if input_col not in df.columns:
        raise ValueError(f"input_col '{input_col}' not found.")
    if output_col not in df.columns:
        raise ValueError(f"output_col '{output_col}' not found.")

    X_list = []
    y_list = []

    for i, (x, y) in enumerate(zip(df[input_col].tolist(), df[output_col].tolist())):
        if pd.isna(x):
            continue

        if isinstance(x, (list, tuple, np.ndarray)):
            vec = np.array(x, dtype=float)

        else:
            s = str(x).strip()

            # FIX: handle "np.float64(0.0)" style values
            s = s.replace("np.float64(", "").replace(")", "")

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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

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
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="weighted", zero_division=0
        )
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

    report.append(f"## Best Model: {best_name} \n")
    report_md = "\n".join(report)

    report_path = save_text(report_md, None, bucket="reports", base_name="training_report", ext="md", add_timestamp=True)
    append_command_log("reports", f"train report -> {report_path}")
    print("Saved report:", report_path)

    model_path = None
    if save_model:
        model_path = save_pickle(best_model, save_model, bucket="models", base_name="best_model", add_timestamp=False)
        append_command_log("models", f"train best_model -> {model_path}")
        print("Saved model:", model_path)

    return report_path, model_path




def pipeline_command(csv_path, text_col, label_col, preprocessing, embedding, training, output, save_model, save_report):
    current_csv = csv_path

    if preprocessing == "all":
        current_csv = all_command(current_csv, text_col, "auto", output=None)
    else:
        steps = [s.strip() for s in preprocessing.split(",") if s.strip()]
        for step in steps:
            if step == "remove":
                current_csv = remove_command(current_csv, text_col, remove="all", output=None)
            elif step == "stopwords":
                current_csv = stopwords_command(current_csv, text_col, output=None, language="auto")
            elif step == "replace":
                current_csv = replace_command(current_csv, text_col, output=None)
            else:
                raise ValueError(f"Unknown preprocessing step: {step}")

    if embedding == "tfidf":
        emb_path = tfidfcommand(current_csv, text_col, max_features=5000, output=None)
    elif embedding == "model2vec":
        emb_path = model2vec_command(current_csv, text_col, output=None)
    elif embedding == "word2vec":
        emb_path = word2vec_command(current_csv, text_col, output=None)
    elif embedding == "fasttext":
        emb_path = fasttext_command(current_csv, text_col, output=None)
    elif embedding == "bert":
        emb_path = bert_command(current_csv, text_col, model="aubmindlab/bert-base-arabertv2", output=None)
    elif embedding == "sentence-transformer":
        emb_path = sentence_transformer_command(current_csv, text_col, model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", output=None)
    else:
        raise ValueError(f"Unknown embedding: {embedding}")

    df = pd.read_csv(current_csv)
    with open(emb_path, "rb") as f:
        obj = pickle.load(f)

    X = obj["embeddings"] if isinstance(obj, dict) and "embeddings" in obj else obj
    if hasattr(X, "toarray"):  
        X = X.toarray()

    df["embedding"] = [list(row) for row in X]
    train_csv = save_csv(df, None, bucket="processed_csvs", base_name="final_with_embeddings", add_timestamp=True)
    append_command_log("processed_csvs", f"pipeline embeddings->csv -> {train_csv}")
    print("Saved:", train_csv)

    model_list = ("all",) if training == "all" else tuple([s.strip() for s in training.split(",") if s.strip()])

    model_out = "best_model.pkl" if save_model else None

    report_path, best_model_path = train_command(train_csv, "embedding", label_col, 0.2, None, model_list, model_out, text_col)

    append_command_log("pipeline", f"pipeline -> csv={train_csv} emb={emb_path} report={report_path} model={best_model_path}")
    print("Pipeline done.")
    print("Final CSV:", train_csv)
    print("Embeddings:", emb_path)
    print("Report:", report_path)
    if best_model_path:
        print("Best model:", best_model_path)

    return {"final_csv": train_csv, "embeddings": emb_path, "report": report_path, "model": best_model_path}
