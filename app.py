import os
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

from commands.generate import generate_command
from commands.preprocessing import remove_command, stopwords_command, replace_command, all_command, stem_command, lemmatize_command
from commands.eda import distribution_command, histogram_command, remove_outliers_command, wordcloud_command
from commands.embedding import tfidfcommand, model2vec_command, bert_command, sentence_transformer_command, word2vec_command, fasttext_command
from commands.training import train_command, pipeline_command
from commands.ir import ir_setup_command, ir_query_command


st.set_page_config(page_title="NLP CLI Pipeline (Streamlit)", layout="wide")
st.title("NLP CLI Pipeline")

def save_upload_to_tmp(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".csv"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def show_file_download(path: str, label: str, key: str):
    p = Path(path)
    if p.exists() and p.is_file():
        with open(p, "rb") as f:
            st.download_button(label=label, data=f, file_name=p.name, key=key)

st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="sidebar_upload_csv")
csv_path = None
df_preview = None

if uploaded:
    csv_path = save_upload_to_tmp(uploaded)
    try:
        df_preview = pd.read_csv(csv_path)
        st.sidebar.success(f"Loaded: {uploaded.name}")
        with st.expander("Preview (first 20 rows)", expanded=False):
            st.dataframe(df_preview.head(20), use_container_width=True)
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")

st.sidebar.divider()
st.sidebar.write("Tip: If a command requires a CSV, upload it here first.")

tab_names = ["Generate", "Preprocess", "EDA", "Embed", "Train", "Pipeline", "IR"]
tabs = st.tabs(tab_names)

with tabs[0]:
    st.subheader("Generate Synthetic Data")
    col1, col2, col3 = st.columns(3)

    model = col1.selectbox("Model", ["gemini", "local"], index=1, key="gen_model")
    class_name = col2.text_input("Optional class label (class_name)", value="", key="gen_class_name")
    count = col3.number_input("Count", min_value=1, max_value=100000, value=100, step=10, key="gen_count")

    output_name = st.text_input("Output filename (optional)", value="synthetic.csv", key="gen_output_name")
    run = st.button("Generate", key="gen_run")

    if run:
        out_path = generate_command(model=model, class_name=(class_name if class_name.strip() else None), count=int(count), output=output_name)
        st.success(f"Saved: {out_path}")
        show_file_download(out_path, "Download generated CSV", key="gen_download_csv")


with tabs[1]:
    st.subheader("Preprocessing")
    if not csv_path:
        st.info("Upload a CSV in the sidebar to enable preprocessing.")
    else:
        cols = df_preview.columns.tolist() if df_preview is not None else []
        text_col = st.selectbox("Text column", cols, index=cols.index("description") if "description" in cols else 0, key="pre_text_col")
        language = st.selectbox("Language", ["ar", "en", "auto", "both"], index=0, key="pre_language")

        action = st.selectbox("Action", ["remove", "stopwords", "replace", "all", "stem", "lemmatize"], index=3, key="pre_action")
        output_name = st.text_input("Output filename", value=f"{action}.csv", key="pre_output_name")

        stemmer = None
        remove_steps = None

        if action == "remove":
            remove_steps = st.text_input('Remove steps (ignored in current impl, keep "all")', value="all", key="pre_remove_steps")
        if action == "stem":
            stemmer = st.selectbox("Stemmer", ["snowball"], index=0, key="pre_stemmer")

        if st.button("Run Preprocess", key="pre_run"):
            if action == "remove":
                out_path = remove_command(csv_path, text_col, remove_steps, output_name)
            elif action == "stopwords":
                out_path = stopwords_command(csv_path, text_col, output_name, language)
            elif action == "replace":
                out_path = replace_command(csv_path, text_col, output_name)
            elif action == "all":
                out_path = all_command(csv_path, text_col, language, output_name)
            elif action == "stem":
                out_path = stem_command(csv_path, text_col, language, stemmer, output_name)
            elif action == "lemmatize":
                out_path = lemmatize_command(csv_path, text_col, language, output_name)
            else:
                out_path = None

            st.success(f"Saved: {out_path}")
            if out_path:
                show_file_download(out_path, "Download processed CSV", key="pre_download_csv")


with tabs[2]:
    st.subheader("EDA")
    if not csv_path:
        st.info("Upload a CSV in the sidebar to enable EDA.")
    else:
        cols = df_preview.columns.tolist() if df_preview is not None else []
        eda_action = st.selectbox("EDA Action", ["distribution", "histogram", "remove_outliers", "wordcloud"], index=0, key="eda_action")

        if eda_action == "distribution":
            label_col = st.selectbox("Label column", cols, index=cols.index("class") if "class" in cols else 0, key="eda_dist_label_col")
            plot_type = st.selectbox("Plot type", ["pie", "bar"], index=0, key="eda_dist_plot_type")
            language = st.selectbox("Language", ["ar", "en", "auto", "both"], index=0, key="eda_dist_language")

            if st.button("Run Distribution", key="eda_dist_run"):
                distribution_command(csv_path, label_col, language, None, None, plot_type)
                st.info("Chart saved to outputs/visualizations (see terminal logs for exact file path).")

        elif eda_action == "histogram":
            text_col = st.selectbox("Text column", cols, index=cols.index("description") if "description" in cols else 0, key="eda_hist_text_col")
            unit = st.selectbox("Unit", ["words", "chars"], index=0, key="eda_hist_unit")

            if st.button("Run Histogram", key="eda_hist_run"):
                histogram_command(csv_path, text_col, unit)
                st.info("Chart saved to outputs/visualizations (see terminal logs for exact file path).")

        elif eda_action == "remove_outliers":
            text_col = st.selectbox("Text column", cols, index=cols.index("description") if "description" in cols else 0, key="eda_out_text_col")
            method = st.selectbox("Method", ["iqr", "zscore"], index=0, key="eda_out_method")
            output_name = st.text_input("Output filename", value="no_outliers.csv", key="eda_out_output_name")

            if st.button("Remove Outliers", key="eda_out_run"):
                out_path = remove_outliers_command(csv_path, text_col, method, output_name)
                st.success(f"Saved: {out_path}")
                if out_path:
                    show_file_download(out_path, "Download outlier-removed CSV", key="eda_out_download_csv")

        elif eda_action == "wordcloud":
            text_col = st.selectbox("Text column", cols, index=cols.index("description") if "description" in cols else 0, key="eda_wc_text_col")
            label_col = st.selectbox("Optional label column (blank = combined)", [""] + cols, index=0, key="eda_wc_label_col")
            output_name = st.text_input("Output (file or directory)", value="wordcloud.png" if label_col == "" else "wordclouds/", key="eda_wc_output_name")

            if st.button("Generate WordCloud", key="eda_wc_run"):
                out = wordcloud_command(csv_path, text_col, (label_col if label_col else None), output_name)
                st.success(f"Saved: {out}")


with tabs[3]:
    st.subheader("Embedding")
    if not csv_path:
        st.info("Upload a CSV in the sidebar to enable embeddings.")
    else:
        cols = df_preview.columns.tolist() if df_preview is not None else []
        text_col = st.selectbox("Text column", cols, index=cols.index("description") if "description" in cols else 0, key="emb_text_col")
        method = st.selectbox("Method", ["tfidf", "model2vec", "word2vec", "fasttext", "bert", "sentence-transformer"], index=0, key="emb_method")

        output_name = st.text_input("Output filename", value=f"{method}_vectors.pkl", key="emb_output_name")

        max_features = None
        model_name = None

        if method == "tfidf":
            max_features = st.number_input("max_features", min_value=100, max_value=200000, value=5000, step=100, key="emb_max_features")
        if method in ["bert", "sentence-transformer"]:
            model_name = st.text_input("Model name", value="aubmindlab/bert-base-arabertv2" if method == "bert" else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", key="emb_model_name")

        if st.button("Run Embedding", key="emb_run"):
            if method == "tfidf":
                out_path = tfidfcommand(csv_path, text_col, int(max_features), output_name)
            elif method == "model2vec":
                out_path = model2vec_command(csv_path, text_col, output_name)
            elif method == "word2vec":
                out_path = word2vec_command(csv_path, text_col, output_name)
            elif method == "fasttext":
                out_path = fasttext_command(csv_path, text_col, output_name)
            elif method == "bert":
                out_path = bert_command(csv_path, text_col, model_name, output_name)
            elif method == "sentence-transformer":
                out_path = sentence_transformer_command(csv_path, text_col, model_name, output_name)
            else:
                out_path = None

            st.success(f"Saved: {out_path}")
            if out_path:
                show_file_download(out_path, "Download embeddings (.pkl)", key="emb_download_pkl")


with tabs[4]:
    st.subheader("Training")
    if not csv_path:
        st.info("Upload a CSV in the sidebar to enable training.")
    else:
        cols = df_preview.columns.tolist() if df_preview is not None else []
        input_col = st.text_input("Input column (e.g., embedding) OR embeddings .pkl path", value="embedding", key="train_input_col")
        output_col = st.selectbox("Label column", cols, index=cols.index("class") if "class" in cols else 0, key="train_output_col")
        test_size = st.slider("Test size", min_value=0.05, max_value=0.5, value=0.2, step=0.05, key="train_test_size")

        model_choices = st.multiselect("Models", ["knn", "lr", "rf", "all"], default=["knn", "lr", "rf"], key="train_models")
        if "all" in model_choices:
            model_choices = ["all"]

        save_model_flag = st.checkbox("Save best model", value=False, key="train_save_model_flag")
        model_name = st.text_input("Model output filename (if saving)", value="best_model.pkl", key="train_model_name")

        if st.button("Run Training", key="train_run"):
            save_model = model_name if save_model_flag else None
            report_path, model_path = train_command(csv_path, input_col, output_col, float(test_size), None, tuple(model_choices), save_model, None)

            st.success(f"Report: {report_path}")
            if report_path:
                show_file_download(report_path, "Download report (.md)", key="train_download_report")

            if model_path:
                st.success(f"Model: {model_path}")
                show_file_download(model_path, "Download best model (.pkl)", key="train_download_model")


with tabs[5]:
    st.subheader("One-line Pipeline")
    if not csv_path:
        st.info("Upload a CSV in the sidebar to enable pipeline.")
    else:
        cols = df_preview.columns.tolist() if df_preview is not None else []
        text_col = st.selectbox("Text column", cols, index=cols.index("description") if "description" in cols else 0, key="pipe_text_col")
        label_col = st.selectbox("Label column", cols, index=cols.index("class") if "class" in cols else 0, key="pipe_label_col")

        preprocessing = st.text_input('Preprocessing ("all" or comma list)', value="remove,stopwords,replace", key="pipe_preprocessing")
        embedding = st.selectbox("Embedding", ["tfidf", "model2vec", "word2vec", "fasttext", "bert", "sentence-transformer"], index=0, key="pipe_embedding")
        training = st.text_input('Training ("all" or comma list)', value="knn,lr,rf", key="pipe_training")

        out_dir = st.text_input("Output directory", value="results/", key="pipe_out_dir")
        save_models = st.checkbox("Save models", value=False, key="pipe_save_models")
        save_report = st.checkbox("Save report", value=False, key="pipe_save_report")

        if st.button("Run Pipeline", key="pipe_run"):
            res = pipeline_command(csv_path, text_col, label_col, preprocessing, embedding, training, out_dir, save_models, save_report)
            st.success("Pipeline completed.")
            st.json(res)


with tabs[6]:
    st.subheader("Information Retrieval")
    if not csv_path:
        st.info("Upload a CSV in the sidebar to enable IR setup.")
    else:
        cols = df_preview.columns.tolist() if df_preview is not None else []
        text_col = st.selectbox("Text column", cols, index=cols.index("description") if "description" in cols else 0, key="ir_text_col")

        idx_type = st.selectbox("Index type", ["bm25", "faiss"], index=0, key="ir_idx_type")
        index_out = st.text_input("Index output filename", value="index.pkl", key="ir_index_out")

        if st.button("Build Index", key="ir_build"):
            out_path = ir_setup_command(csv_path, text_col, idx_type, index_out)
            st.success(f"Saved: {out_path}")
            if out_path:
                show_file_download(out_path, "Download index", key="ir_download_index")

        st.divider()
        st.write("Query")
        query = st.text_input("Query text", value="ما هو الذكاء الاصطناعي؟", key="ir_query")
        index_path = st.text_input("Index path (.pkl)", value=index_out, key="ir_index_path")
        top_k = st.number_input("Top K", min_value=1, max_value=50, value=5, step=1, key="ir_top_k")

        if st.button("Run Query", key="ir_run_query"):
            results = ir_query_command(query, index_path, int(top_k))
            st.write(results)
