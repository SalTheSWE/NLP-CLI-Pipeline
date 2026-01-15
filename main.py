import click
from commands.ir import ir_query_command,ir_setup_command
from commands.eda import distribution_command,histogram_command,remove_outliers_command,wordcloud_command
from commands.generate import generate_command
from commands.preprocessing import remove_command,stem_command,stopwords_command,replace_command,all_command,lemmatize_command
from commands.embedding import tfidfcommand,model2vec_command,bert_command,sentence_transformer_command,word2vec_command,fasttext_command
from commands.training import train_command,pipeline_command
@click.group()
def cli():
    pass

@cli.group()
def eda():
    pass

@eda.command(epilog= """Example usage:


# View class distribution (pie chart)
python main.py eda distribution --csv_path data.csv --label_col class

# View class distribution (bar chart)
python main.py eda distribution --csv_path data.csv --label_col class --plot_type bar

# Process both Arabic and English
python main.py eda distribution --csv_path data.csv --label_col class --language both

# Load dataset directly from Hugging Face
python main.py eda distribution --dataset "MARBERT/XNLI" --split "ar" --label_col label
""")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False), required=False,
              help="Path to CSV file (mutually exclusive with --dataset).")
@click.option("--label_col", type=str, required=True, help="Label/class column name.")
@click.option("--language", type=click.Choice(["ar", "en", "both", "auto"]), required=False,
              help="Language handling mode (optional).")
@click.option("--dataset", type=str, required=False,
              help='Hugging Face dataset name (e.g., "MARBERT/XNLI"). Mutually exclusive with --csv_path.')
@click.option("--split", type=str, required=False, help="Dataset split/config (optional).")
@click.option("--plot_type", type=click.Choice(["pie", "bar"]), default="pie", show_default=True,
              help="Plot type for distribution.")
def distribution(csv_path, label_col, language, dataset, split, plot_type):
    distribution_command(csv_path, label_col, language, dataset, split, plot_type)
    pass

@eda.command(epilog= """Example usage:


# Generate text length histogram (word count)
python main.py eda histogram --csv_path data.csv --text_col description --unit words

# Generate text length histogram (character count)
python main.py eda histogram --csv_path data.csv --text_col description --unit chars
""")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to CSV file.")
@click.option("--text_col", type=str, required=True, help="Text column name.")
@click.option("--unit", type=click.Choice(["words", "chars"]), required=True,
              help='Histogram unit: "words" or "chars".')
def histogram(csv_path, text_col, unit):
    histogram_command(csv_path, text_col, unit)
    pass




@cli.group()
def preprocess():
    pass


@preprocess.command(epilog= """Example usage:


# Remove Arabic-specific characters (tashkeel, tatweel, tarqeem, links, etc.)
python main.py preprocess remove --csv_path data.csv --text_col description --output cleaned.csv
                    
# Choose specific preprocessing
python main.py preprocess remove --csv_path data.csv --text_col description --remove "tashkeel,links" --output partial_clean.csv

""")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to CSV file.")
@click.option("--text_col", type=str, required=True, help="Text column name.")
@click.option("--remove", type=str, required=True,
              help='Comma-separated steps (e.g., "tashkeel,links").')
@click.option("--output", type=click.Path(dir_okay=False), required=True,
              help="Output CSV path.")
def remove(csv_path, text_col, remove, output):
    remove_command(csv_path, text_col, remove, output)
    pass

@preprocess.command(epilog= """Example usage:


# Remove stopwords using Arabic stopwords list
python main.py preprocess stopwords --csv_path cleaned.csv --text_col description --output no_stops.csv
                    
# Remove stopwords in any language
python main.py preprocess stopwords --csv_path data.csv --text_col description --language en --output cleaned.csv

python main.py preprocess stopwords --csv_path data.csv --text_col description --language fr --output cleaned.csv

python main.py preprocess stopwords --csv_path data.csv --text_col description --language auto --output cleaned.csv

""")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to CSV file.")
@click.option("--text_col", type=str, required=True, help="Text column name.")
@click.option("--output", type=click.Path(dir_okay=False), required=True,
              help="Output CSV path.")
@click.option("--language", type=click.Choice(["ar", "en", "fr", "auto", "both"]), required=True,
              help="Stopwords language selection.")
def stopwords(csv_path, text_col, output, language):
    stopwords_command(csv_path, text_col, output, language)
    pass

@preprocess.command(epilog= """Example usage:


# Normalize Arabic text (hamza, alef maqsoura, taa marbouta)
python main.py preprocess replace --csv_path no_stops.csv --text_col description --output normalized.csv
""")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to CSV file.")
@click.option("--text_col", type=str, required=True, help="Text column name.")
@click.option("--output", type=click.Path(dir_okay=False), required=True,
              help="Output CSV path.")
def replace(csv_path, text_col, output):
    replace_command(csv_path, text_col, output)
    pass

@preprocess.command(epilog= """Example usage:


# Chain all preprocessing steps
python main.py preprocess all --csv_path data.csv --text_col description --output final.csv

# Multilingual preprocessing
python main.py preprocess all --csv_path data.csv --text_col description --language auto --output final.csv
""")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to CSV file.")
@click.option("--text_col", type=str, required=True, help="Text column name.")
@click.option("--language", type=click.Choice(["ar", "en", "both", "auto"]), required=True,
              help="Language mode for preprocessing.")
@click.option("--output", type=click.Path(dir_okay=False), required=True,
              help="Output CSV path.")
@click.pass_context
def all(ctx, csv_path, text_col, language, output):
    all_command(ctx, csv_path, text_col, language, output)
    pass


@cli.group()
def embed():
    pass

@embed.command(epilog= """Example usage:


# TF-IDF Embedding (sklearn)
python main.py embed tfidf --csv_path cleaned.csv --text_col description --max_features 5000 --output tfidf_vectors.pkl
""")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to CSV file.")
@click.option("--text_col", type=str, required=True, help="Text column name.")
@click.option("--max_features", type=int, required=True, help="Max TF-IDF features.")
@click.option("--output", type=click.Path(dir_okay=False), required=True,
              help="Output pickle path.")
def tfidf(csv_path, text_col, max_features, output):
    tfidfcommand(csv_path, text_col, max_features, output)
    pass

@embed.command(epilog= """Example usage:


# Model2Vec Embedding (from HuggingFace: JadwalAlmaa/model2vec-ARBERTv2)
python main.py embed model2vec --csv_path cleaned.csv --text_col description --output model2vec_vectors.pkl
""")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to CSV file.")
@click.option("--text_col", type=str, required=True, help="Text column name.")
@click.option("--output", type=click.Path(dir_okay=False), required=True,
              help="Output pickle path.")
def model2vec(csv_path, text_col, output):
    model2vec_command(csv_path, text_col, output)
    pass

@embed.command(epilog= """Example usage:


# BERT Embedding (bonus)
python main.py embed bert --csv_path cleaned.csv --text_col description --model AraBERT --output bert_vectors.pkl
""")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to CSV file.")
@click.option("--text_col", type=str, required=True, help="Text column name.")
@click.option("--model", type=str, required=True, help="Model name/id.")
@click.option("--output", type=click.Path(dir_okay=False), required=True,
              help="Output pickle path.")
def bert(csv_path, text_col, model, output):
    bert_command(csv_path, text_col, model, output)
    pass

@embed.command(epilog= """Example usage:


# Sentence Transformers (bonus)
python main.py embed sentence-transformer --csv_path cleaned.csv --text_col description --model sentence-transformers/distiluse-base-multilingual-cased-v2 --output sent_vectors.pkl
""")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to CSV file.")
@click.option("--text_col", type=str, required=True, help="Text column name.")
@click.option("--model", type=str, required=True, help="SentenceTransformer model id.")
@click.option("--output", type=click.Path(dir_okay=False), required=True,
              help="Output pickle path.")
def sentence_transformer(csv_path, text_col, model, output):
    sentence_transformer_command(csv_path, text_col, model, output)
    pass



@cli.command(epilog= """Example usage:


# Train with default models (KNN, Logistic Regression, Random Forest)
python main.py train --csv_path final.csv --input_col embedding --output_col class --test_size 0.2 --models knn lr rf

# Train with all sklearn models
python main.py train --csv_path final.csv --input_col embedding --output_col class --models all

# Train with custom hyperparameters
python main.py train --csv_path final.csv --input_col embedding --output_col class --models "knn:n_neighbors=7" "lr:C=0.5"

# Save best model to disk
python main.py train --csv_path final.csv --input_col embedding --output_col class --save_model best_model.pkl

# Use dataset in full pipeline
python main.py train --dataset "ARBML/ArabiCorpus" --text_col text --output_col category
""")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False), required=False,
              help="Path to CSV file (mutually exclusive with --dataset).")
@click.option("--input_col", type=str, required=False,
              help="Input feature column name (e.g., embedding). Required with --csv_path.")
@click.option("--output_col", type=str, required=True, help="Target/label column.")
@click.option("--test_size", type=float, default=0.2, show_default=True,
              help="Test split size in [0,1].")
@click.option("--dataset", type=str, required=False,
              help='Hugging Face dataset name (mutually exclusive with --csv_path).')
@click.option("--models", type=str, multiple=True, required=True,
              help='Models to train (e.g., knn lr rf | "knn:n_neighbors=7"). Use "all" for all supported.')
@click.option("--save_model", type=click.Path(dir_okay=False), required=False,
              help="Optional path to save the best model.")
@click.option("--text_col", type=str, required=False,
              help="Text column name when using --dataset.")
def train(csv_path, input_col, output_col, test_size, dataset, models, save_model, text_col):
    train_command(csv_path, input_col, output_col, test_size, dataset, models, save_model, text_col)
    pass

#bonus-section-bonus-section-bonus-section-bonus-section-bonus-section-bonus-section

@cli.command(epilog= """Example usage:


# Generate synthetic Arabic text using a language model
python main.py generate --model gemini --class_name "positive" --count 100 --output synthetic.csv

# Generate using a downloaded model (optional)
python main.py generate --model local --count 100 --output synthetic.csv
""")
@click.option("--model", type=click.Choice(["gemini", "local"]), required=True,
              help="Generator backend.")
@click.option("--class_name", type=str, required=False,
              help="Optional class name/label for generated samples.")
@click.option("--count", type=int, required=True,
              help="Number of samples to generate.")
@click.option("--output", type=click.Path(dir_okay=False), required=True,
              help="Output CSV path.")
def generate(model, class_name, count, output):
    generate_command(model, class_name, count, output)
    pass


@eda.command(epilog= """Example usage:


# Remove statistical outliers from EDA/preprocessing
python main.py eda remove-outliers --csv_path data.csv --text_col description --method iqr --output clean_data.csv
""")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to CSV file.")
@click.option("--text_col", type=str, required=True, help="Text column name.")
@click.option("--method", type=click.Choice(["iqr", "zscore"]), required=True,
              help='Outlier removal method.')
@click.option("--output", type=click.Path(dir_okay=False), required=True,
              help="Output CSV path.")
def remove_outliers(csv_path, text_col, method, output):
    remove_outliers_command(csv_path, text_col, method, output)
    pass


@cli.command(epilog= """Example usage:


# Run all steps in sequence
python main.py pipeline --csv_path data.csv --text_col description --label_col class \
  --preprocessing "remove,stopwords,replace" \
  --embedding tfidf \
  --training "knn,lr,rf" \
  --output results/

# With all options
python main.py pipeline --csv_path data.csv --text_col description --label_col class \
  --preprocessing all --embedding model2vec --training all --save_report --save_models
""")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to CSV file.")
@click.option("--text_col", type=str, required=True, help="Text column name.")
@click.option("--label_col", type=str, required=True, help="Label column name.")
@click.option("--preprocessing", type=str, required=True,
              help='Preprocessing steps: "all" or comma-separated list.')
@click.option("--embedding", type=click.Choice(["tfidf", "model2vec", "word2vec", "fasttext", "bert", "sentence-transformer"]),
              required=True, help="Embedding method.")
@click.option("--training", type=str, required=True,
              help='Training models: "all" or comma-separated list.')
@click.option("--output", type=click.Path(file_okay=False), required=True,
              help="Output directory.")
@click.option("--save_model", is_flag=True, default=False, help="Save trained model(s).")
@click.option("--save_report", is_flag=True, default=False, help="Save evaluation report.")
def pipeline(csv_path, text_col, label_col, preprocessing, embedding, training, output, save_model, save_report):
    pipeline_command(csv_path, text_col, label_col, preprocessing, embedding, training, output, save_model, save_report)
    pass


@embed.command(epilog= """Example usage:


# Word2Vec
python main.py embed word2vec --csv_path cleaned.csv --text_col description --output word2vec_vectors.pkl
""")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to CSV file.")
@click.option("--text_col", type=str, required=True, help="Text column name.")
@click.option("--output", type=click.Path(dir_okay=False), required=True,
              help="Output pickle path.")
def word2vec(csv_path, text_col, output):
    word2vec_command(csv_path, text_col, output)
    pass

@embed.command(epilog= """Example usage:


# FastText
python main.py embed fasttext --csv_path cleaned.csv --text_col description --output fasttext_vectors.pkl
""")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to CSV file.")
@click.option("--text_col", type=str, required=True, help="Text column name.")
@click.option("--output", type=click.Path(dir_okay=False), required=True,
              help="Output pickle path.")
def fasttext(csv_path, text_col, output):
    fasttext_command(csv_path, text_col, output)
    pass

@preprocess.command(epilog= """Example usage:


# Apply Arabic stemming (Snowball)
python main.py preprocess stem --csv_path cleaned.csv --text_col description --language ar --stemmer snowball --output stemmed.csv
""")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to CSV file.")
@click.option("--text_col", type=str, required=True, help="Text column name.")
@click.option("--language", type=click.Choice(["ar", "en", "auto"]), required=True,
              help="Language for stemming.")
@click.option("--stemmer", type=click.Choice(["snowball"]), required=True,
              help="Stemmer to use.")
@click.option("--output", type=click.Path(dir_okay=False), required=True,
              help="Output CSV path.")
def stem(csv_path, text_col, language, stemmer, output):
    stem_command(csv_path, text_col, language, stemmer, output)
    pass

@preprocess.command(epilog= """Example usage:


# Arabic lemmatization (CAMeLTools)
python main.py preprocess lemmatize --csv_path cleaned.csv --text_col description --language ar --output lemmatized.csv
""")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to CSV file.")
@click.option("--text_col", type=str, required=True, help="Text column name.")
@click.option("--language", type=click.Choice(["ar", "en", "auto"]), required=True,
              help="Language for lemmatization.")
@click.option("--output", type=click.Path(dir_okay=False), required=True,
              help="Output CSV path.")
def lemmatize(csv_path, text_col, language, output):
    lemmatize_command(csv_path, text_col, language, output)
    pass

@eda.command(epilog= """Example usage:


# Generate word cloud for each class
python main.py eda wordcloud --csv_path cleaned.csv --text_col description --label_col class --output wordclouds/

# Combined word cloud
python main.py eda wordcloud --csv_path cleaned.csv --text_col description --output wordcloud.png
""")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to CSV file.")
@click.option("--text_col", type=str, required=True, help="Text column name.")
@click.option("--label_col", type=str, required=False,
              help="Optional label column (if provided, generate per-class wordclouds).")
@click.option("--output", type=click.Path(), required=True,
              help="Output path (directory for per-class, or file for combined).")
def wordcloud(csv_path, text_col, label_col, output):
    wordcloud_command(csv_path, text_col, label_col, output)
    pass

@cli.command(epilog= """Example usage:


# Build a semantic search system instead of classification
python main.py ir-setup --csv_path documents.csv --text_col content --index_type faiss --output index.faiss
""")
@click.option("--csv_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to CSV file.")
@click.option("--text_col", type=str, required=True, help="Text column name.")
@click.option("--index_type", type=click.Choice(["faiss", "bm25"]), required=True,
              help="Index type.")
@click.option("--output", type=click.Path(), required=True,
              help="Output index path.")
def ir_setup(csv_path, text_col, index_type, output):
    ir_setup_command(csv_path, text_col, index_type, output)
    pass

@cli.command(epilog= """Example usage:


# Query the index
python main.py ir-query --query "ما هو الذكاء الاصطناعي؟" --index_path index.faiss --top_k 5
""")
@click.option("--query", type=str, required=True, help="Query text.")
@click.option("--index_path", type=click.Path(exists=True), required=True,
              help="Path to the built index.")
@click.option("--top_k", type=int, default=5, show_default=True,
              help="Number of results to return.")
def ir_query(query, index_path, top_k):
    ir_query_command(query, index_path, top_k)
    pass

if __name__ == "__main__":
    cli()