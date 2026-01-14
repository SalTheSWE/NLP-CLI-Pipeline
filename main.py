import click

@click.group()
def cli():
    pass

@cli.group()
def eda():
    pass

@eda.command()#example usage python main.py eda distribution --csv_path data.csv --label_col class --plot_type bar
@click.option("--csv_path", required=True)
@click.option("--label_col", required=True)
@click.option("--language", required=True)#bonus
@click.option("--dataset", required=True)#bonus
@click.option("--split", required=True)#bonus
@click.option("--plot_type", required=False)#if not specified default to piechart, choose "pie" or "bar" 
def distribution():
    pass

@eda.command()#example usage python main.py eda distribution --csv_path data.csv --label_col class --plot_type bar
@click.option("--csv_path", required=True)
@click.option("--text_col", required=True)
@click.option("--unit", required=True)# could be "words" or "chars"
def histogram():
    pass




@cli.group()
def preprocess():
    pass


@preprocess.command()
@click.option("--csv_path", required=True)
@click.option("--text_col", required=True)
@click.option("--remove", required=True)
@click.option("--output", required=True)
def remove():
    pass

@preprocess.command()
@click.option("--csv_path", required=True)
@click.option("--text_col", required=True)
@click.option("--output", required=True)
@click.option("--language", required=True)#bonus
def stopwords():
    pass

@preprocess.command()
@click.option("--csv_path", required=True)
@click.option("--text_col", required=True)
@click.option("--output", required=True)
def replace():
    pass

@preprocess.command()
@click.option("--csv_path", required=True)
@click.option("--text_col", required=True)
@click.option("--language", required=True)#bonus
@click.option("--output", required=True)
def all():
    replace(remove(stopwords))# something like that 
    pass


@cli.group()
def embed():
    pass

@embed.command()
@click.option("--csv_path", required=True)
@click.option("--text_col", required=True)
@click.option("--max_features", required=True)
@click.option("--output", required=True)
def tfidf():
    pass

@embed.command()
@click.option("--csv_path", required=True)
@click.option("--text_col", required=True)
@click.option("--output", required=True)
def model2vec():
    pass

@embed.command()
@click.option("--csv_path", required=True)
@click.option("--text_col", required=True)
@click.option("--model", required=True)
@click.option("--output", required=True)
def bert():
    pass

@embed.command()
@click.option("--csv_path", required=True)
@click.option("--text_col", required=True)
@click.option("--model", required=True)
@click.option("--output", required=True)
def sentence_transformer():
    pass



@cli.command()
@click.option("--csv_path", required=True)
@click.option("--input_col", required=True)
@click.option("--output_col", required=True)
@click.option("--test_size", required=True)
@click.option("--dataset", required=True)#bonus
@click.option("--models", required=True)
@click.option("--save_model", required=True)
def train():
    pass

#bonus-section-bonus-section-bonus-section-bonus-section-bonus-section-bonus-section

@cli.command()
@click.option("--models", required=True)
@click.option("--class_name", required=True)
@click.option("--count", required=True)
@click.option("--output", required=True)
def generate():
    pass


@eda.command()
@click.option("--csv_path", required=True)
@click.option("--text_col", required=True)
@click.option("--method", required=True)
@click.option("--output", required=True)
def remove_outliers():
    pass


@cli.command()
@click.option("--csv_path", required=True)
@click.option("--text_col", required=True)
@click.option("--label_col", required=True)
@click.option("--preprocessing", required=True)
@click.option("--embedding", required=True)
@click.option("--training", required=True)
@click.option("--output", required=True)
@click.option("--save_model", required=True)
@click.option("--save_report", required=True)
def pipeline():
    pass


@embed.command()
@click.option("--csv_path", required=True)
@click.option("--text_col", required=True)
@click.option("--output", required=True)
def word2vec():
    pass

@embed.command()
@click.option("--csv_path", required=True)
@click.option("--text_col", required=True)
@click.option("--output", required=True)
def fasttext():
    pass

@preprocess.command()
@click.option("--csv_path", required=True)
@click.option("--text_col", required=True)
@click.option("--language", required=True)
@click.option("--stemmer", required=True)
@click.option("--output", required=True)
def stem():
    pass

@preprocess.command()
@click.option("--csv_path", required=True)
@click.option("--text_col", required=True)
@click.option("--language", required=True)
@click.option("--output", required=True)
def lemmatize():
    pass

@eda.command()
@click.option("--csv_path", required=True)
@click.option("--text_col", required=True)
@click.option("--label_col", required=True)
@click.option("--output", required=True)
def wordcloud():
    pass

@cli.command()
@click.option("--csv_path", required=True)
@click.option("--text_col", required=True)
@click.option("--index_type", required=True)
@click.option("--output", required=True)
def ir_setup():
    pass

@cli.command()
@click.option("--query", required=True, help = "WWWWWWWWWWWWWWW")
@click.option("--index_path", required=True)
@click.option("--top_k", required=True)
def ir_query():
    pass

if __name__ == "__main__":
    cli()