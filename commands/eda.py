import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from utils.arabic_text import char_counts, word_counts, _fix_arabic
from utils.data_handler import save_current_figure, save_csv, append_command_log
import os



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
    df = pd.read_csv(csv_path)
    s = df[text_col].fillna("").astype(str)

    lengths = s.str.split().apply(len)  

    if method == "iqr":
        q1 = lengths.quantile(0.25)
        q3 = lengths.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        keep = (lengths >= lo) & (lengths <= hi)
    elif method == "zscore":
        mu = lengths.mean()
        sd = lengths.std() if lengths.std() != 0 else 1.0
        z = (lengths - mu) / sd
        keep = z.abs() <= 3.0
    else:
        raise ValueError("method must be 'iqr' or 'zscore'")

    before = len(df)
    df = df.loc[keep].reset_index(drop=True)
    after = len(df)

    out_path = save_csv(df, output, bucket="processed_csvs", base_name="no_outliers", add_timestamp=False)
    append_command_log("processed_csvs", f"remove-outliers ({method}) {before}->{after} -> {out_path}")
    print(f"Removed outliers: {before - after} rows. Saved CSV: {out_path}")
    return out_path

def wordcloud_command(csv_path, text_col, label_col, output):
    df = pd.read_csv(csv_path)
    df[text_col] = df[text_col].fillna("").astype(str)

    def maybe_arabic_shape(t):
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            return get_display(arabic_reshaper.reshape(t))
        except Exception:
            return t

    if label_col:
        out_dir = output
        os.makedirs(out_dir, exist_ok=True)

        for label, g in df.groupby(label_col):
            text = " ".join(g[text_col].tolist()).strip()
            if not text:
                continue
            wc = WordCloud(width=1200, height=600, background_color="white").generate(maybe_arabic_shape(text))
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"WordCloud: {label}")

            file_path = os.path.join(out_dir, f"wordcloud_{str(label)}.png")
            plt.tight_layout()
            plt.savefig(file_path, dpi=200, bbox_inches="tight")
            plt.close()

        print("Saved wordclouds to:", out_dir)
        return out_dir

    text = " ".join(df[text_col].tolist()).strip()
    wc = WordCloud(width=1200, height=600, background_color="white").generate(maybe_arabic_shape(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("WordCloud")

    out_path = save_current_figure(output, bucket="visualizations", base_name="wordcloud", add_timestamp=False)
    append_command_log("visualizations", f"eda wordcloud -> {out_path}")
    print("Saved:", out_path)
    plt.show()
    plt.close()
    return out_path
    