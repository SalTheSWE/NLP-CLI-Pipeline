import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.data_handler import save_pickle, append_command_log

def ir_setup_command(csv_path, text_col, index_type, output):
    df = pd.read_csv(csv_path)
    texts = df[text_col].fillna("").astype(str).tolist()

    vec = TfidfVectorizer(max_features=50000)
    X = vec.fit_transform(texts)

    payload = {"vectorizer": vec, "matrix": X, "texts": texts}
    out_path = save_pickle(payload, output, bucket="ir_index", base_name=f"ir_{index_type}", add_timestamp=False)
    append_command_log("ir_index", f"ir-setup ({index_type}) -> {out_path}")
    print("Saved:", out_path)
    return out_path

def ir_query_command(query, index_path, top_k):
    import pickle

    with open(index_path, "rb") as f:
        payload = pickle.load(f)

    vec = payload["vectorizer"]
    X = payload["matrix"]
    texts = payload["texts"]

    qv = vec.transform([str(query)])
    sims = cosine_similarity(qv, X).ravel()
    idx = sims.argsort()[::-1][:int(top_k)]

    results = [(int(i), float(sims[i]), texts[i]) for i in idx]
    for rank, (i, score, text) in enumerate(results, 1):
        print(f"{rank}. score={score:.4f} | idx={i} | {text[:200]}")
    return results