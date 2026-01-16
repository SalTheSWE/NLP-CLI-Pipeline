import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from utils.arabic_text import char_counts,word_counts, _fix_arabic
from utils.data_handler import save_current_figure, append_command_log




def distribution_command(csv_path, label_col, language="ar", dataset=None, split=None, plot_type = "pie"):
    df = pd.read_csv(csv_path)

    col = df[label_col]
    counts = col.value_counts()

    labels = counts.index.astype(str).tolist()
    if language and language.lower() in ["ar", "arabic"]:
        labels = [_fix_arabic(x) for x in labels]
        plt.rcParams["font.family"] = "Noto Naskh Arabic"  

    plt.figure(figsize=(7, 7))

    if plot_type == "pie":
        plt.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90)
        plt.axis("equal")
    elif plot_type == "bar":
        plt.bar(labels, counts.values)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
    else:
        raise ValueError("plot_type must be 'pie' or 'bar'")
    out_path = save_current_figure(None, bucket="visualizations", base_name=f"distribution_{label_col}_{plot_type}", add_timestamp=True)
    append_command_log("visualizations", f"eda distribution ({plot_type}) -> {out_path}")
    print("Saved:", out_path)
    plt.title(f"Distribution of {label_col}")
    plt.show()
    
    
def histogram_command(csv_path, text_col, unit):
    df = pd.read_csv(csv_path)

    col = df[text_col]

    if unit == "words":
        data = word_counts(col)
    elif unit == "chars":
        data = char_counts(col)
        
    data = pd.Series(data).dropna()
    print(f"Summary statistics ({unit})")
    print(f"Count:  {data.count()}")
    print(f"Mean:   {data.mean():.2f}")
    print(f"Median: {data.median():.2f}")
    print(f"Std:    {data.std():.2f}")
    print(f"Min:    {data.min()}")
    print(f"Max:    {data.max()}")

    min_v = int(data.min())
    max_v = int(data.max())

    bins = range(min_v, max_v + 2)

    plt.figure(figsize=(7, 5))
    sns.histplot(data, bins=bins, discrete=True, shrink=1)
    plt.xticks(range(min_v, max_v + 1))
    plt.title(f"{unit.capitalize()} histogram: {text_col}")

    out_path = save_current_figure(None, bucket="visualizations", base_name=f"histogram_{text_col}_{unit}", add_timestamp=True)
    append_command_log("visualizations", f"eda histogram ({unit}) -> {out_path}")
    print("Saved:", out_path)

    plt.show()
def remove_outliers_command(csv_path, text_col, method, output):
    return
def wordcloud_command(csv_path, text_col, label_col, output):

    return