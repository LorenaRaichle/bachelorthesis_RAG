



## Using stratified sampling to create a evaluation dataset of SQuAD2.0 dataset of questions and expected answers in order to evaluate different RAG systems.

# The script performs stratified sampling on the SQuAD 2.0 training dataset to generate two subsets: one that maintains the original category distribution and another with a custom distribution emphasizing abstract questions. The output of this script will be three CSV files:

# Three different question categories depending on the expected answer will be created
#   - 1K = keyword based answer
#   - 1 A = abstract answer
#   - 0 = unanswerable question
#should be maintained.

# output files:
# 1. squad_v2_train_with_category.csv: This file contains the original SQuAD 2.0 training dataset with an additional 'Category' column that classifies questions into '1K' (keyword-based answers), '1A' (abstract answers), and '0' (unanswerable questions).
# 2. sampled_balanced_df.csv: This file contains a subset of the dataset (1% of the total) that maintains the original distribution of categories. It ensures that the proportions of '1K', '1A', and '0' categories in the subset are the same as in the original dataset.
# 3. sampled_custom_df.csv: This file contains a custom-distributed subset of the dataset with a specified distribution: 50% abstract questions ('1A'), 25% keyword-based questions ('1K'), and 25% unanswerable questions ('0'). This custom sampling is based on the desired_total number of samples.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset


def determine_category(row):
    if not row['answers']['text']:
        return '0'
    else:
        num_tokens = len(row['answers']['text'][0].split())
        if num_tokens <= 2:
            return '1K'
        else:
            return '1A'


def load_and_preprocess_squad():
    squad_v2_train = load_dataset('squad_v2', split="train")
    squad_v2_train = squad_v2_train.to_pandas()
    squad_v2_train['Category'] = squad_v2_train.apply(determine_category, axis=1)
    squad_v2_train.to_csv('squad_v2_train_with_category.csv', index=False)
    print("squad_v2_train_with_category.csv saved.")
    return squad_v2_train


def stratified_sampling(df, test_size=0.01):
    X = df.drop(columns=['Category'])
    y = df['Category']
    X_train, X_sample, y_train, y_sample = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    sampled_balanced_df = pd.concat([X_sample, y_sample], axis=1)
    sampled_balanced_df.to_csv('sampled_balanced_df.csv', index=False)
    print("sampled_balanced_df.csv saved.")
    return sampled_balanced_df


def custom_sampling(df, proportions, total_samples):
    X = df.drop(columns=['Category'])
    y = df['Category']
    samples = []

    for category, proportion in proportions.items():
        num_samples = int(proportion * total_samples)
        X_category = X[y == category]
        y_category = y[y == category]
        X_sample, _, y_sample, _ = train_test_split(X_category, y_category, train_size=num_samples, random_state=42,
                                                    stratify=y_category)
        samples.append(pd.concat([X_sample, y_sample], axis=1))

    sampled_custom_df = pd.concat(samples)
    sampled_custom_df.to_csv('sampled_custom_df.csv', index=False)
    print("sampled_custom_df.csv saved.")
    return sampled_custom_df


def main():
    squad_v2_train = load_and_preprocess_squad()
    print(squad_v2_train.head())

    category_counts = squad_v2_train['Category'].value_counts()
    print("Original Distribution of Categories:\n", category_counts)

    sampled_balanced_df = stratified_sampling(squad_v2_train)
    sampled_category_counts = sampled_balanced_df['Category'].value_counts()
    print("Distribution of Categories in Sampled Balanced DataFrame:\n", sampled_category_counts)

    custom_proportions = {'1A': 0.5, '1K': 0.25, '0': 0.25}
    desired_total = 1300
    sampled_custom_df = custom_sampling(squad_v2_train, custom_proportions, desired_total)
    sampled_custom_df_counts = sampled_custom_df['Category'].value_counts()
    print("Distribution of Categories in Custom Sampled DataFrame:\n", sampled_custom_df_counts)


if __name__ == "__main__":
    main()
