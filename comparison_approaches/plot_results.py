### extract result metrics for further processing

import pandas as pd
import ast
import re


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Check for NaN values and fill them if necessary
    if df.isna().any().any():
        print("DataFrame contains NaN values. Preprocessing steps might be needed.")
        df.fillna(0, inplace=True)
    else:
        print("DataFrame does not contain any NaN values. No preprocessing needed.")

    # Extract BERTScore
    df['bertscore_result'] = df['bertscore_result'].replace(0, "{}")
    df['bertscore_result'] = df['bertscore_result'].apply(safe_literal_eval)
    df[['bertscore_precision', 'bertscore_recall', 'bertscore_f1']] = df.apply(extract_bertscore_values, axis=1,
                                                                               result_type='expand')
    df.drop(columns=["bertscore_result"], inplace=True)

    # Extract ROUGE score
    df['rouge_score'] = df['rouge_score'].replace(0, "")
    df['rouge_score'] = df['rouge_score'].astype(str)
    df[['rougeL_f1', 'rougeL_precision', 'rougeL_recall']] = df.apply(extract_rouge_l_values, axis=1,
                                                                      result_type='expand')
    df.drop(columns=["rouge_score"], inplace=True)

    # Extract SacreBLEU score
    df[['Sacrebleu_score', 'Sacrebleu_counts', 'Sacrebleu_bp', 'Sacrebleu_sys_len', 'Sacrebleu_ref_len']] = df.apply(
        extract_sacrebleu_values, axis=1, result_type='expand')
    df.drop(columns=["sacrebleu_score"], inplace=True)

    df.to_csv("plots_baseline/baseline_scores_extracted.csv", index=False)
    return df


# extracting metrics
def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return {}  # Return an empty dictionary if there is a problem

def extract_bertscore_values(row):
    if isinstance(row['bertscore_result'], dict):
        bertscore_dict = row['bertscore_result']
        precision = bertscore_dict.get('precision', [None])[0]
        recall = bertscore_dict.get('recall', [None])[0]
        f1 = bertscore_dict.get('f1', [None])[0]
        return precision, recall, f1
    else:
        return None, None, None

def extract_rouge_l_values(row):
    rouge_score_str = row['rouge_score']
    pattern = re.compile(
        r"'rougeL': AggregateScore\(low=Score\(precision=(?P<low_prec>[^,]+), recall=(?P<low_rec>[^,]+), fmeasure=(?P<low_f1>[^,]+)\), "
        r"mid=Score\(precision=(?P<mid_prec>[^,]+), recall=(?P<mid_rec>[^,]+), fmeasure=(?P<mid_f1>[^,]+)\), "
        r"high=Score\(precision=(?P<high_prec>[^,]+), recall=(?P<high_rec>[^,]+), fmeasure=(?P<high_f1>[^,]+)\)\)"
    )
    match = pattern.search(rouge_score_str)
    if match:
        mid_f1 = float(match.group('mid_f1'))
        mid_precision = float(match.group('mid_prec'))
        mid_recall = float(match.group('mid_rec'))
        return mid_f1, mid_precision, mid_recall
    else:
        return None, None, None

def extract_sacrebleu_values(row):
    sacrebleu_score_str = row['sacrebleu_score']
    try:
        sacrebleu_dict = ast.literal_eval(sacrebleu_score_str)
        score = sacrebleu_dict.get('score', None)
        counts = sacrebleu_dict.get('counts', None)
        bp = sacrebleu_dict.get('bp', None)
        sys_len = sacrebleu_dict.get('sys_len', None)
        ref_len = sacrebleu_dict.get('ref_len', None)
        return score, counts, bp, sys_len, ref_len
    except (ValueError, SyntaxError) as e:
        return None, None, None, None, None


# plotting
import seaborn as sns
import matplotlib.pyplot as plt


def plot_boxplots(df, numeric_columns, output_dir='plots_baseline/plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for col in numeric_columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='category', y=col, data=df)
        plt.title(f'EVAL baseline: Distribution of {col} by Category')
        plt.xlabel('Category')
        plt.ylabel(col)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{col}_boxplot_baseline.png')
        plt.show()

# recall & precision ROUGE-L
def plot_recall_precision(approaches, recall, precision):
    import numpy as np
    import matplotlib.pyplot as plt

    categories = ['1K', '1A']
    x = np.arange(len(approaches))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))

    recall_1K = [recall[approach][0] for approach in approaches]
    recall_1A = [recall[approach][1] for approach in approaches]
    precision_1K = [precision[approach][0] for approach in approaches]
    precision_1A = [precision[approach][1] for approach in approaches]

    bars1 = ax.bar(x - width/2, recall_1K, width, label='Recall 1K', color='skyblue')
    bars2 = ax.bar(x - width/2, recall_1A, width, bottom=recall_1K, label='Recall 1A', color='lightgreen')
    bars3 = ax.bar(x + width/2, precision_1K, width, label='Precision 1K', color='salmon')
    bars4 = ax.bar(x + width/2, precision_1A, width, bottom=precision_1K, label='Precision 1A', color='orange')

    ax.set_xlabel('Approach')
    ax.set_ylabel('Values')
    ax.set_title('Mean Recall and Precision by Approach and Category')
    ax.set_xticks(x)
    ax.set_xticklabels(approaches)
    ax.legend()

    fig.tight_layout()
    plt.show()


def main():
    # Load and preprocess data
    df = load_and_preprocess_data("plots_baseline/eval_baseline_balanced_FINAL.csv")

    # Grouped statistics
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    grouped_df = df.groupby("category")
    grouped_stats = grouped_df[numeric_columns].agg(['mean', 'median']).reset_index()
    grouped_stats.to_csv("plots_baseline/baseline_res.csv", index=False)

    # Detect and mark outliers
    for col in numeric_columns:
        df[f"{col}_outlier"] = detect_outliers(df[col])

    # Plot boxplots
    plot_boxplots(df, numeric_columns)

    # Define recall and precision data
    approaches = ['Naive Llama 2', 'Baseline RAG', 'STB', 'Metadata Filtering', 'Re-ranking']
    recall = {
        'Naive Llama 2': [0.255, 0.322],
        'Baseline RAG': [0.446, 0.522],
        'STB': [0.634, 0.656],
        'Metadata Filtering': [0.643, 0.611],
        'Re-ranking': [0.704, 0.651]
    }
    precision = {
        'Naive Llama 2': [0.030, 0.103],
        'Baseline RAG': [0.226, 0.228],
        'STB': [0.245, 0.300],
        'Metadata Filtering': [0.299, 0.303],
        'Re-ranking': [0.276, 0.299]
    }

    # Plot recall and precision
    plot_recall_precision(approaches, recall, precision)


if __name__ == "__main__":
    main()
