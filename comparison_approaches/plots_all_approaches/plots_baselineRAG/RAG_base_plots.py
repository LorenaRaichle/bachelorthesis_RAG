# -*- coding: utf-8 -*-
"""
Analysis of results: Base RAG plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import re

# Step 1: Load the CSV file
file_path = "plots_baseline/eval_baseline_balanced_FINAL.csv"
df = pd.read_csv(file_path)

# Step 2: Data Preprocessing
print("Checking for NaN values...")
nan_values = df.isna().sum()
print("NaN Values:\n", nan_values)

if df.isna().any().any():
    print("DataFrame contains NaN values. Preprocessing steps might be needed.")
    df.fillna(0, inplace=True)
else:
    print("DataFrame does not contain any NaN values. No preprocessing needed.")

print("Shape of the DataFrame:", df.shape)
print("Columns in the DataFrame:", df.columns)
category_counts = df["category"].value_counts()
print("Number of values for each category:")
print(category_counts)
print("First Row:\n", df.loc[0])

# ---- BERT SCORE Extraction -------------
df['bertscore_result'] = df['bertscore_result'].replace(0, "{}")


def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return {}


df['bertscore_result'] = df['bertscore_result'].apply(safe_literal_eval)


def extract_bertscore_values(row):
    if isinstance(row['bertscore_result'], dict):
        bertscore_dict = row['bertscore_result']
        precision = bertscore_dict.get('precision', [None])[0]
        recall = bertscore_dict.get('recall', [None])[0]
        f1 = bertscore_dict.get('f1', [None])[0]
        return precision, recall, f1
    else:
        return None, None, None


df['bertscore_precision'], df['bertscore_recall'], df['bertscore_f1'] = zip(*df.apply(extract_bertscore_values, axis=1))
df.drop(columns=["bertscore_result"], inplace=True)

# -------------------ROUGE SCORE Extraction------------------
df['rouge_score'] = df['rouge_score'].replace(0, "")


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


df['rougeL_f1'], df['rougeL_precision'], df['rougeL_recall'] = zip(*df.apply(extract_rouge_l_values, axis=1))
df.drop(columns=["rouge_score"], inplace=True)


# -------------------SACRE BLEU SCORE Extraction------------------
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
    except (ValueError, SyntaxError):
        return None, None, None, None, None


df['Sacrebleu_score'], df['Sacrebleu_counts'], df['Sacrebleu_bp'], df['Sacrebleu_sys_len'], df[
    'Sacrebleu_ref_len'] = zip(*df.apply(extract_sacrebleu_values, axis=1))
df.drop(columns=["sacrebleu_score"], inplace=True)

# Save the extracted scores to CSV
df.to_csv("plots_baseline/baseline_scores_extracted.csv", index=False)

# Add the group name to each row
df['category_group'] = df['category']
grouped_df = df.groupby("category")
group_counts = grouped_df.size()
print(group_counts)

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

grouped_stats = grouped_df[numeric_columns].agg(['mean', 'median'])
print("Grouped stats:", grouped_stats)
grouped_stats = grouped_stats.reset_index()
grouped_stats.to_csv("plots_baseline/baseline_res.csv", index=False)


# Outlier Detection
def detect_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    return (series < lower_fence) | (series > upper_fence)


for col in numeric_columns:
    outlier_col_name = f"{col}_outlier"
    df[outlier_col_name] = detect_outliers(df[col])

print(df.head())

# Save DataFrame with outliers to CSV
# df.to_csv("Llama_res_with_outliers.csv", index=False)

# Data Visualization
import matplotlib.pyplot as plt
import numpy as np

# Data for Visualization
approaches = ['Naive Llama 2', 'Baseline RAG', 'STB', 'Metadata Filtering', 'Re-ranking']
categories = ['1K', '1A']
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

# Set positions and width
x = np.arange(len(approaches))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 8))

# Plot Recall
recall_1K = [recall[approach][0] for approach in approaches]
recall_1A = [recall[approach][1] for approach in approaches]
bars1 = ax.bar(x - width / 2, recall_1K, width, label='Recall 1K', color='skyblue')
bars2 = ax.bar(x - width / 2, recall_1A, width, bottom=recall_1K, label='Recall 1A', color='lightgreen')

# Plot Precision
precision_1K = [precision[approach][0] for approach in approaches]
precision_1A = [precision[approach][1] for approach in approaches]
bars3 = ax.bar(x + width / 2, precision_1K, width, label='Precision 1K', color='salmon')
bars4 = ax.bar(x + width / 2, precision_1A, width, bottom=precision_1K, label='Precision 1A', color='orange')

# Add some text for labels, title, and custom x-axis tick labels
ax.set_xlabel('Approach')
ax.set_ylabel('Values')
ax.set_title('Mean Recall and Precision by Approach and Category')
ax.set_xticks(x)
ax.set_xticklabels(approaches)
ax.legend()

fig.tight_layout()

plt.show()
