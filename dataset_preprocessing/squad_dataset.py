
import pandas as pd
from datasets import load_dataset


def check_statistics(df):
    stats = {}

    for column in df.columns:
        if column == 'answers':
            # Convert 'answers' column to string for statistical purposes
            df[column] = df[column].apply(lambda x: str(x))

        if df[column].dtype == 'object':
            # Calculate length of each string item in the column
            length_series = df[column].apply(lambda x: len(str(x)))
            stats[column] = {
                'unique_count': df[column].nunique(),
                'length_mean': length_series.mean(),
                'length_std': length_series.std(),
                'length_min': length_series.min(),
                'length_25%': length_series.quantile(0.25),
                'length_50%': length_series.median(),
                'length_75%': length_series.quantile(0.75),
                'length_max': length_series.max()
            }
        elif pd.api.types.is_numeric_dtype(df[column]):
            # Generate descriptive statistics for numeric columns
            stats[column] = df[column].describe().to_dict()

    return stats


def preprocess_dataset():
    squad_v2_train = load_dataset('squad_v2', split="train")
    print("Dataset loaded:", type(squad_v2_train))
    squad_v2_train = squad_v2_train.to_pandas()[['id', 'context', 'title', 'question', 'answers']]
    unique_questions = squad_v2_train["question"].nunique()
    print("Total questions:", len(squad_v2_train["question"]))
    print("Unique questions:", unique_questions)
    squad_v2_train_unique_contexts = squad_v2_train.drop_duplicates(subset='context', keep='first')

    squad_v2_train['context_len_words'] = squad_v2_train['context'].apply(lambda x: len(str(x).split(' ')))
    squad_v2_train['answer_len_words'] = squad_v2_train['answers'].apply(lambda x: len(str(x).split(' ')))

    print("Context length stats:", squad_v2_train['context_len_words'].describe())
    print("Answer length stats:", squad_v2_train['answer_len_words'].describe())

    stats = check_statistics(squad_v2_train)
    for column, column_stats in stats.items():
        print(f"\nStatistics for column '{column}':")
        for stat_name, stat_value in column_stats.items():
            print(f"  {stat_name}: {stat_value}")

    return squad_v2_train

preprocess_dataset()
