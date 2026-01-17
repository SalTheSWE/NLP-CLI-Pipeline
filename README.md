# NLP CLI Pipeline

A modular **CLI + Streamlit** toolkit for Arabic/Multilingual NLP workflows. This project is organized as a set of Click commands (CLI) plus an optional Streamlit UI that calls the same command functions.

## Features

- **Synthetic data generation** (local templates or Gemini if configured)
- **Preprocessing**: cleaning, normalization, stopwords, stemming, lemmatization
- **EDA**: class distribution, histogram stats + plots, wordclouds, outlier removal
- **Embeddings**: TF‑IDF + optional neural embeddings
- **Training**: train/evaluate multiple classifiers and generate a Markdown report
- **Pipeline**: run preprocessing → embedding → training in one command
- **IR**: build an index and query it 
- **Deterministic output routing**: outputs saved under `outputs/` using a single centralized handler
- **Example usage and help function**: --help for any commands display example usage and help 

---

## Project Structure

```
NLP-CLI-Pipeline/
│
├── main.py                         # CLI entrypoint (Click)
├── app.py                          # Streamlit UI
│
├── commands/
│   ├── eda.py                      # EDA commands (plots + stats)
│   ├── preprocessing.py            # preprocessing commands
│   ├── embedding.py                # embedding commands
│   ├── training.py                 # train + pipeline
│   ├── generate.py                 # synthetic dataset generation
│   └── ir.py                       # information retrieval
│
├── utils/
│   ├── data_handler.py             # ALL outputs path routing + saving/logging
│   ├── arabic_text.py              # Arabic helpers
│   └── visualization.py            # plotting helpers 
│
└── outputs/
    ├── visualizations/
    ├── processed_csvs/
    ├── generated_csvs/
    ├── embeddings/
    ├── models/
    ├── reports/
    └── ir_index/
```

---

## Install with `uv`

### 1) Create + sync environment

```bash
uv venv
uv pip install -r requirements.txt
```

### 2) Run commands with `uv run`

Examples:
```bash
uv run python main.py --help
uv run python main.py generate --model local --count 200 --output synthetic.csv
uv run streamlit run app.py
```

---

## Outputs

All outputs are saved under:

```
outputs/
```

Buckets:
- `outputs/visualizations/` → plots (png)
- `outputs/processed_csvs/` → cleaned/processed datasets (csv)
- `outputs/generated_csvs/` → synthetic datasets (csv)
- `outputs/embeddings/` → vectors (pkl)
- `outputs/models/` → trained models (pkl)
- `outputs/reports/` → training reports (md)
- `outputs/ir_index/` → search indices

### Output path behavior

- If you pass `--output cleaned.csv` (filename only), it is saved into the correct bucket folder automatically.
- If you pass `--output some/folder/cleaned.csv`, it is saved exactly there (folder created if needed).
- Each bucket includes a `commands_to_output.txt` log file.

---

## Quickstart (end-to-end)

### 1) Generate a dataset (6 classes in Arabic templates)

```bash
uv run python main.py generate --model local --count 300 --output data.csv
```

### 2) Preprocess

```bash
uv run python main.py preprocess all --csv_path data.csv --text_col description --language ar --output final.csv
```

### 3) Embed (TF‑IDF)

```bash
uv run python main.py embed tfidf --csv_path final.csv --text_col description --max_features 5000 --output tfidf_vectors.pkl
```

### 4) Train

```bash
uv run python main.py train --csv_path final.csv --input_col embedding --output_col class --test_size 0.2 --models knn lr rf
```

---

## CLI Reference

### Generate Synthetic Data

Local template generation:
```bash
uv run python main.py generate --model local --count 200 --output synthetic.csv
```

Optional class label:
```bash
uv run python main.py generate --model local --class_name "رياضة" --count 50 --output sports.csv
```

Gemini generation (requires key):
```bash
set GEMINI_API_KEY=YOUR_KEY
uv run python main.py generate --model gemini --count 100 --output gemini.csv
```

### Preprocessing

Remove noise:
```bash
uv run python main.py preprocess remove --csv_path data.csv --text_col description --remove all --output cleaned.csv
```

Stopwords:
```bash
uv run python main.py preprocess stopwords --csv_path cleaned.csv --text_col description --language ar --output no_stops.csv
```

Normalize Arabic:
```bash
uv run python main.py preprocess replace --csv_path no_stops.csv --text_col description --output normalized.csv
```

All preprocessing:
```bash
uv run python main.py preprocess all --csv_path data.csv --text_col description --language auto --output final.csv
```

Stem:
```bash
uv run python main.py preprocess stem --csv_path final.csv --text_col description --language ar --stemmer snowball --output stemmed.csv
```

Lemmatize:
```bash
uv run python main.py preprocess lemmatize --csv_path final.csv --text_col description --language ar --output lemmatized.csv
```

### EDA

Distribution:
```bash
uv run python main.py eda distribution --csv_path data.csv --label_col class --plot_type bar --language ar
```

Histogram:
```bash
uv run python main.py eda histogram --csv_path data.csv --text_col description --unit words
```

Outliers:
```bash
uv run python main.py eda remove-outliers --csv_path data.csv --text_col description --method iqr --output no_outliers.csv
```

Wordcloud:
```bash
uv run python main.py eda wordcloud --csv_path data.csv --text_col description --output wordcloud.png
```

Per-class wordclouds:
```bash
uv run python main.py eda wordcloud --csv_path data.csv --text_col description --label_col class --output wordclouds/
```

### Embeddings

TF‑IDF:
```bash
uv run python main.py embed tfidf --csv_path data.csv --text_col description --max_features 5000 --output tfidf_vectors.pkl
```

Model2Vec:
```bash
uv run python main.py embed model2vec --csv_path data.csv --text_col description --output model2vec_vectors.pkl
```

BERT (bonus):
```bash
uv run python main.py embed bert --csv_path data.csv --text_col description --model aubmindlab/bert-base-arabertv2 --output bert.pkl
```

Sentence Transformers (bonus):
```bash
uv run python main.py embed sentence-transformer --csv_path data.csv --text_col description --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --output st.pkl
```

### Training

Default models:
```bash
uv run python main.py train --csv_path final.csv --input_col embedding --output_col class --test_size 0.2 --models knn lr rf
```

All models (mapped to supported set):
```bash
uv run python main.py train --csv_path final.csv --input_col embedding --output_col class --models all
```

Save best model:
```bash
uv run python main.py train --csv_path final.csv --input_col embedding --output_col class --models knn lr rf --save_model best_model.pkl
```

Training output:
- `outputs/reports/training_report_[timestamp].md`
- `outputs/models/best_model.pkl` (if enabled)

### One-line Pipeline

```bash
uv run python main.py pipeline --csv_path data.csv --text_col description --label_col class \
  --preprocessing "remove,stopwords,replace" \
  --embedding tfidf \
  --training "knn,lr,rf" \
  --output results/
```

Full options:
```bash
uv run python main.py pipeline --csv_path data.csv --text_col description --label_col class \
  --preprocessing all --embedding model2vec --training all --output results/ --save_report --save_models
```

### Information Retrieval (IR)

Build index:
```bash
uv run python main.py ir-setup --csv_path documents.csv --text_col content --index_type bm25 --output index.pkl
```

Query:
```bash
uv run python main.py ir-query --query "ما هو الذكاء الاصطناعي؟" --index_path index.pkl --top_k 5
```

---

## Streamlit UI

Run:

```bash
uv run streamlit run app.py
```



