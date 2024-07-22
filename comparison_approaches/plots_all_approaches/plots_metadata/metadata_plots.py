# -*- coding: utf-8 -*-
"""
Analysis of results, metadata plots
"""
import os
import pandas as pd
import ast
import re
import seaborn as sns
import matplotlib.pyplot as plt

# Define file paths
input_file_path = "plots_metadata/eval-metadata-VALUES2.csv"
output_file_path_extracted = "plots_metadata/metadata_scores_extracted.csv"
output_file_path_summary = "plots_metadata/metadata_res.csv"
output_plots_dir = "plots_metadata/plots/"

# Create output directory if it doesn't exist
os.makedirs(output_plots_dir, exist_ok=True)

# Step 1: Load the CSV file
df = pd.read_csv(input_file_path)

# Step 2: Data Preprocessing
print("Columns in DataFrame:")
print(df.columns)

# Drop irrelevant columns
df.drop(columns=["sacrebleu_score", "rouge_score", "bertscore_result"], axis=1, inplace=True)

# Check for NaN values
nan_values = df.isna().sum()
print("NaN Values:\n", nan_values)

if df.isna().any().any():
    print("DataFrame contains NaN values. Filling NaNs with 0.")
    df.fillna(0, inplace=True)
else:
    print("DataFrame does not contain any NaN values.")

# Check zero values
zero_value_count = (df["SacreBLEU"] == 0).sum()
print("Total number of zero values in 'SacreBLEU':", zero_value_count)

# Print shape and categories
print("Shape of the DataFrame:", df.shape)
print("Columns in the DataFrame:", df.columns)
category_counts = df["category"].value_counts()
print("Number of values for each category:\n", category_counts)
print("First Row:\n", df.loc[0])

# ---- BERT SCORE Extraction ----
df['BERTScore'] = df['BERTScore'].replace(0, "{}")

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return {}

df['BERTScore'] = df['BERTScore'].apply(safe_literal_eval)

def extract_bertscore_values(row):
    if isinstance(row['BERTScore'], dict):
        bertscore_dict = row['BERTScore']
        precision = bertscore_dict.get('precision', [None])[0]
        recall = bertscore_dict.get('recall', [None])[0]
        f1 = bertscore_dict.get('f1', [None])[0]
        return precision, recall, f1
    return None, None, None

df['bertscore_precision'], df['bertscore_recall'], df['bertscore_f1'] = zip(*df.apply(extract_bertscore_values, axis=1))
df.drop(columns=["BERTScore"], inplace=True)

# ---- ROUGE SCORE Extraction ----
df['ROUGE'] = df['ROUGE'].replace(0, "")

def extract_rouge_l_values(row):
    rouge_score_str = row['ROUGE']
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
    return None, None, None

df['rougeL_f1'], df['rougeL_precision'], df['rougeL_recall'] = zip(*df.apply(extract_rouge_l_values, axis=1))
df.drop(columns=["ROUGE"], inplace=True)

# ---- SACRE BLEU SCORE Extraction ----
def extract_sacrebleu_values(row):
    sacrebleu_score_str = row['SacreBLEU']
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

df['Sacrebleu_score'], df['Sacrebleu_counts'], df['Sacrebleu_bp'], df['Sacrebleu_sys_len'], df['Sacrebleu_ref_len'] = zip(*df.apply(extract_sacrebleu_values, axis=1))
df.drop(columns=["SacreBLEU"], inplace=True)

# Save the extracted scores to a CSV file
df.to_csv(output_file_path_extracted, index=False)

# Add the group name to each row and calculate grouped statistics
df['category_group'] = df['category']
grouped_df = df.groupby("category")
group_counts = grouped_df.size()
print(group_counts)

numeric_columns = ['EM', 'F1', 'correct_context', 'cnt_unanswCorr', 'cnt_unanswIncorr', 'cnt_answNotGiven', 'Sacrebleu_score', 'Sacrebleu_counts', 'Sacrebleu_bp', 'Sacrebleu_sys_len', 'Sacrebleu_ref_len', 'rougeL_f1', 'rougeL_precision', 'rougeL_recall', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

grouped_stats = grouped_df[numeric_columns].agg(['mean', 'median'])
print("Grouped stats:", grouped_stats)
grouped_stats = grouped_stats.reset_index()
grouped_stats.to_csv(output_file_path_summary, index=False)

# ---- Outlier Detection ----
def detect_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    return (series < lower_fence) | (series > upper_fence)

# Add outlier columns to DataFrame
for col in numeric_columns:
    outlier_col_name = f"{col}_outlier"
    df[outlier_col_name] = detect_outliers(df[col])

# Save the DataFrame with outlier columns
# df.to_csv("plots_metadata/metadata_res_with_outliers.csv", index=False)

# ---- Data Visualization ----
# Plot boxplots for each numeric column
for col in numeric_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='category', y=col, data=df)
    plt.title(f'EVAL Metadata: Distribution of {col} by Category')
    plt.xlabel('Category')
    plt.ylabel(col)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(True)  # Show grid
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig(f'{output_plots_dir}{col}_boxplot_metadata.png')
    plt.show()

print("Analysis completed. Outputs saved to:")
print(output_file_path_extracted)
print(output_file_path_summary)
print(f'Plots saved to {output_plots_dir}')
