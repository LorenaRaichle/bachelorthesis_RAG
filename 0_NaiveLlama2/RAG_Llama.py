# -*- coding: utf-8 -*-
"""

## RAG System implementing Naive Llama2 With Hugging Face.
- load and set up the LLaMA 2 model, answer questions from the SQuAD2.0 dataset, evaluate the answers, and store the results in a final DataFrame.
"""



import subprocess
import pandas as pd
import datasets
from datasets import load_dataset
from transformers import BertTokenizer, AutoModelForCausalLM, AutoTokenizer
import torch


# loading SQUAD dataset, converts it to a DataFrame and performs various preprocessing tasks such as calculating the length of contexts and answers.
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

    return squad_v2_train


#Loads the LLaMA 2 model and tokenizer from the Hugging Face model hub.
def setup_llama_model():
    system_prompt = """
    You are a Q&A assistant. Please answer the questions in one sentence.
    """

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        torch_dtype=torch.float16,  # Use mixed precision
        device_map="auto")
    return tokenizer, model, system_prompt


# Define functions for evaluation
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
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
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    prediction = normalize_text(prediction)
    prediction_tokens = tokenizer.tokenize(prediction)
    truth = normalize_text(truth)
    truth_tokens = tokenizer.tokenize(truth)
    if "please select" in prediction:
        return 0
    if not truth and prediction == "no answer in the index":
        return 1
    if truth and prediction == "no answer in the index":
        return 0
    if all(token in prediction_tokens for token in truth_tokens):
        return 1
    return 0


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    common_tokens = set(pred_tokens) & set(truth_tokens)
    if len(common_tokens) == 0:
        return 0
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    return 2 * (prec * rec) / (prec + rec)


# Computes various metrics (Exact Match, F1 score, BLEU score, ROUGE score, BERTScore) to evaluate the model's answers against the ground truth answers.
def compute_metrics(response, gold_answers):
    references = [str(answer).lower() for answer in gold_answers]
    if isinstance(response, str):
        response_list = [response]
    elif isinstance(response, list):
        response_list = [str(r) for r in response]
    elif hasattr(response, 'response'):
        response_list = [response.response]
    else:
        response_list = [str(response)]
    try:
        response_bleu = response_list
        reference_bleu = [references]
        em_score = max(compute_exact_match(response, gold_answer) for gold_answer in gold_answers)
        f1_score = max(compute_f1(response, gold_answer) for gold_answer in gold_answers)
        sacrebleu_score = sacrebleu.compute(predictions=response_list, references=[[ref] for ref in references])
        rouge_score = rouge.compute(predictions=response_list, references=references)
        bertscore_result = bertscore.compute(predictions=response_list, references=references, lang="en")
        return em_score, f1_score, sacrebleu_score, rouge_score, bertscore_result
    except Exception as e:
        print("Error occurred during score computation:", e)
        return None, None, None, None, None


# Takes a question, uses the Naive Llama 2 model to generate an answer, and extracts the answer from the model's response.
def generate_answer(question, tokenizer, model, system_prompt):
    input_text = system_prompt + "\nQuestion: " + question + "\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(inputs["input_ids"], max_length=256, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the answer part
    answer = response.split("Answer:")[-1].strip()
    return answer


# Calls the generate_answer function to get a response from the model.
# Computes the evaluation metrics and stores the results along with the question and ground truth.
def retrieve_and_evaluate(question, gold_answers, row, category, tokenizer, model, system_prompt):
    print(f"Retrieving and evaluating question: {question}")
    response = generate_answer(question, tokenizer, model, system_prompt)
    em_score, f1_score, sacrebleu_score, rouge_score, bertscore_result = compute_metrics(response, gold_answers)
    result = {
        "id": row["id"],
        "category": category,
        "question": question,
        "LLMresponse": response,
        "gold_answer": gold_answers,
        "EM": em_score,
        "F1": f1_score,
        "sacrebleu_score": sacrebleu_score,
        "rouge_score": rouge_score,
        "bertscore_result": bertscore_result
    }
    return result


if __name__ == "__main__":
    squad_v2_train = preprocess_dataset()
    tokenizer, model, system_prompt = setup_llama_model()
    csv_file = "sampled_balanced_df.csv"
    sampled_df = pd.read_csv(csv_file)
    results = []

    sacrebleu = datasets.load_metric("sacrebleu")
    rouge = datasets.load_metric("rouge")
    bertscore = datasets.load_metric("bertscore")
    # sampled_df = sampled_df[:10]
    print(sampled_df.shape)

    # Iterating through each row of the evaluation dataset, retrieves the ground truth answers from the preprocessed SQuAD dataset, generates answers, evaluates them, and collects the results.
    for index, row in sampled_df.iterrows():
        question = row["question"]
        category = row["Category"]
        # excluding the evaluation of unanswerable questions of Category 0 for the Naive Llama 2 approach sine there is no access to external knowledge to detect, whether question is answerable from the index
        if category not in ["1K", "1A"]:
            continue

        id = row["id"]
        squad_row = squad_v2_train[squad_v2_train["id"] == id]
        if squad_row.empty:
            print(f"No matching question found in the preprocessed Squad dataset for: {question}")
            continue
        gold_answers = squad_row['answers'].iloc[0]['text']
        result = retrieve_and_evaluate(question, gold_answers, row, category, tokenizer, model, system_prompt)
        if result:
            results.append(result)

    result_df = pd.DataFrame(results)
    print("result printed")
    print(result_df)
    # final results containing all columns are stored in a csv
    result_df.to_csv("eval_Llama.csv", index=False)
    print("Evaluation of all questions from the CSV file completed. Saved as 'eval_Llama.csv'.")

