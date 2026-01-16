import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from model2vec import StaticModel

from utils.data_handler import save_pickle, append_command_log
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
from gensim.models import FastText
def _texts(csv_path, text_col):
    df = pd.read_csv(csv_path)
    return df[text_col].fillna("").astype(str).tolist()


def _tok(s):
    return re.findall(r"[\w\u0600-\u06FF]+", s.lower())


def tfidfcommand(csv_path, text_col, max_features, output):
    p = Path(csv_path)
    if not p.exists():
        alt = Path("outputs/processed_csvs") / p.name
        if alt.exists():
            csv_path = str(alt)
    texts = _texts(csv_path, text_col)
    v = TfidfVectorizer(max_features=max_features)
    X = v.fit_transform(texts)

    out_path = save_pickle({"vectorizer": v, "embeddings": X}, output, bucket="embeddings", base_name="tfidf", add_timestamp=False)
    append_command_log("embeddings", f"tfidf -> {out_path}")
    print("Saved:", out_path)
    return out_path


def model2vec_command(csv_path, text_col, output):
    texts = _texts(csv_path, text_col)
    model = StaticModel.from_pretrained("ARBERTv2")
    X = model.encode(texts)

    out_path = save_pickle({"embeddings": X}, output, bucket="embeddings", base_name="model2vec", add_timestamp=False)
    append_command_log("embeddings", f"model2vec -> {out_path}")
    print("Saved:", out_path)
    return out_path


def bert_command(csv_path, text_col, model, output):
    

    texts = _texts(csv_path, text_col)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModel.from_pretrained(model).to(device).eval()

    X = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, padding=True, max_length=256)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc).last_hidden_state.mean(dim=1)  # mean pool
            X.append(out.squeeze(0).cpu().numpy())

    X = np.vstack(X)
    out_path = save_pickle({"embeddings": X}, output, bucket="embeddings", base_name="bert", add_timestamp=False)
    append_command_log("embeddings", f"bert -> {out_path}")
    print("Saved:", out_path)
    return out_path


def sentence_transformer_command(csv_path, text_col, model, output):
    

    texts = _texts(csv_path, text_col)
    st = SentenceTransformer(model)
    X = st.encode(texts, convert_to_numpy=True)

    out_path = save_pickle({"embeddings": X}, output, bucket="embeddings", base_name="sentence_transformer", add_timestamp=False)
    append_command_log("embeddings", f"sentence-transformer -> {out_path}")
    print("Saved:", out_path)
    return out_path


def word2vec_command(csv_path, text_col, output):
    

    texts = _texts(csv_path, text_col)
    toks = [_tok(t) for t in texts]
    w2v = Word2Vec(toks, vector_size=100, min_count=1, workers=4)

    X = []
    for t in toks:
        vecs = [w2v.wv[w] for w in t if w in w2v.wv]
        X.append(np.mean(vecs, axis=0) if vecs else np.zeros(100))
    X = np.vstack(X)

    out_path = save_pickle({"embeddings": X}, output, bucket="embeddings", base_name="word2vec", add_timestamp=False)
    append_command_log("embeddings", f"word2vec -> {out_path}")
    print("Saved:", out_path)
    return out_path


def fasttext_command(csv_path, text_col, output):
    

    texts = _texts(csv_path, text_col)
    toks = [_tok(t) for t in texts]
    ft = FastText(toks, vector_size=100, min_count=1, workers=4)

    X = []
    for t in toks:
        vecs = [ft.wv[w] for w in t if w in ft.wv]
        X.append(np.mean(vecs, axis=0) if vecs else np.zeros(100))
    X = np.vstack(X)

    out_path = save_pickle({"embeddings": X}, output, bucket="embeddings", base_name="fasttext", add_timestamp=False)
    append_command_log("embeddings", f"fasttext -> {out_path}")
    print("Saved:", out_path)
    return out_path
