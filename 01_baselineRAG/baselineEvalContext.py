# -*- coding: utf-8 -*-
"""
Continuation of RAG_baseline.py to evaluate retrieved context specifically using normalized contexts.
--> final output of finished analysis: eval_baseline_balanced_FINAL.csv
"""


import pandas as pd
import ast
from transformers import BertTokenizer


def normalize_text(s):

    import string, re
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    s = str(s)
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, truth):
    print("predic", prediction)
    print("truth", truth)
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    prediction = normalize_text(prediction)
    prediction_tokens = tokenizer.tokenize(prediction)

    if isinstance(truth, list) and len(truth) == 0:
        truth = []
    else:
        truth = normalize_text(truth)
    truth_tokens = tokenizer.tokenize(truth)
    # If truths is empty and prediction is "no answer in the index", return 1
    if "please select" in prediction:
        return 0
    if not truth and "!no answer in the index!" in prediction:
        return 1
    if truth and "!no answer in the index!" in prediction:
        return 0
    # Check if all tokens of the gold answer are in the prediction
    if all(token in prediction_tokens for token in truth_tokens):
        return 1
    return 0


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    common_tokens = set(pred_tokens) & set(truth_tokens)
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    return 2 * (prec * rec) / (prec + rec)


def compute_metrics(response, gold_answers, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven):
    print("resp", response)
    print("gold answ", gold_answers)
    if isinstance(gold_answers, str):
        # Safely evaluate the string to convert it into a list
        try:
            gold_answers = ast.literal_eval(gold_answers)
        except (ValueError, SyntaxError):
            gold_answers = [gold_answers]  # If it fails, treat it as a single-element list

    # Convert each element to lowercase
    if isinstance(gold_answers, list):
        gold_answers = [answer.lower() for answer in gold_answers]

    print("gold answ after:", gold_answers)

    if isinstance(response, list):
        response = ' '.join(str(r) for r in response)
    elif isinstance(response, str):
        pass
    if hasattr(response, 'response'):
        response = str(response.response)
    try:
        print("compute metrics gold + aresp", gold_answers, "+", response)
        if "!no answer in the index!" in response and not gold_answers:
            cnt_unanswCorr = 1
            print("LLM correctly identified unanswerable question. cnt_unanswCorr:", cnt_unanswCorr)
            return None, None, None, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven
        elif len(response) > 0 and all(len(ref) == 0 for ref in gold_answers):
            cnt_unanswIncorr = 1
            print("LLM incorrectly answered an unanswerable question. cnt_unanswIncorr:", cnt_unanswIncorr)
            return None, None, None, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven
        elif "!no answer in the index!" in response and any(len(ref) > 0 for ref in gold_answers):
            cnt_answNotGiven = 1
            print("LLM did not provide any answer, gold answer available. cnt_answNotGiven:", cnt_answNotGiven)
            return None, None, None, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven
        else:
            sacrebleu_score = sacrebleu.compute(predictions=[response], references=[[ref] for ref in gold_answers])
            rouge_score = rouge.compute(predictions=[response], references=gold_answers)
            bertscore_result = bertscore.compute(predictions=[response], references=gold_answers, lang="en")
            return sacrebleu_score, rouge_score, bertscore_result, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven
    except Exception as e:
        print("Error occurred during score computation:", e)
        return None, None, None, None, None, None


if __name__ == "__main__":
    csv_file = "eval_baseline_balanced.csv"
    resp_df = pd.read_csv(csv_file)
    resp_df = resp_df.drop(columns=["context", "EM", "F1"])
    updated_results = []
    print("resp df cols", resp_df.columns)
    # Evaluate each question in the CSV file
    for index, row in resp_df.iterrows():

        question = row["question"]
        expected_context = row["expected_context"]
        retrieved_context = row["retrieved_context"]
        response = row["LLMresponse"]
        gold_answers = row['gold_answer']
        try:
            gold_answers_list = ast.literal_eval(gold_answers)
        except (ValueError, SyntaxError):
            gold_answers_list = [gold_answers]
        gold_answer = gold_answers_list[0] if gold_answers_list else ""
        em_score = compute_exact_match(response, gold_answer)
        f1_score = compute_f1(response, gold_answer)
        print("em", em_score, "f1", f1_score)
        normalized_expected_context = normalize_text(expected_context)
        normalized_retrieved_context = normalize_text(retrieved_context)
        if normalized_expected_context in normalized_retrieved_context:
            correct_context = 1
        else:
            correct_context = 0
        print("correct context", correct_context)
        updated_results.append({
            "EM": em_score,
            "F1": f1_score,
            "correct_context": correct_context
        })
    updated_df = pd.DataFrame(updated_results)
    final_df = pd.concat([resp_df, updated_df[['EM', 'F1', 'correct_context']]], axis=1)
    final_df.to_csv("eval_baseline_balanced_FINAL.csv", index=False)
    print("Evaluation of all questions from the CSV file completed, saved in 'eval_baseline_balanced_FINAL.csv'.")

