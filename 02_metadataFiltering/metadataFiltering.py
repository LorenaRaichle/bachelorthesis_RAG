# -*- coding: utf-8 -*-
"""
Implementing Metadata Filtering Approach
"""

import datasets
import pandas as pd
import re
import spacy
import numpy as np
import ast
import subprocess
from datasets import load_dataset
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext, Document
from llama_index.core.vector_stores import PineconeVectorStore
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.pinecone import Pinecone
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.schema import TextNode
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from transformers import BertTokenizer


#  Load and preprocess the SQuAD v2.0 dataset.
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

    return squad_v2_train, squad_v2_train_unique_contexts


#  Set up the Llama Index with a Hugging Face LLM, embeddings, and Pinecone vector store.
def setup_llama_index(squad_v2_train):

    # Hugging Face login
    api_token = "xxx"
    subprocess.run(['huggingface-cli', 'login', '--token', api_token])

    system_prompt = """
    You are a Q&A assistant. Please answer the questions based on the context information provided: {additional_info}. If the answer is not given in the index, indicate "!no answer in the index!".
    """
    query_wrapper_prompt = SimpleInputPrompt("{query_str}")

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        device_map="auto",
    )

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    pinecone_instance = Pinecone(api_key="xxx")

    # Create Pinecone index if it doesn't exist
    existing_index = pinecone_instance.list_indexes().names()
    if "rag-metadata" not in existing_index:
        pinecone_instance.create_index(
            name='rag-metadata',
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    pinecone_index = pinecone_instance.Index("rag-metadata")

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    total_contexts = squad_v2_train["context"].tolist()
    combined_text = "\n\n".join(total_contexts)
    combined_document = Document(text=combined_text)

    contains_all_contexts = all(context in combined_document.text for context in total_contexts)
    if contains_all_contexts:
        print("Combined document contains all contexts.")
    else:
        print("Combined document does not contain all contexts.")

    from llama_index.core.node_parser import SentenceWindowNodeParser
    node_parser = SentenceWindowNodeParser.from_defaults(window_size=2, window_metadata_key="window",
                                                         original_text_metadata_key="original_text")

    sentence_nodes = node_parser.get_nodes_from_documents([combined_document])
    for idx, node in enumerate(sentence_nodes):
        node.id_ = f"node-{idx}"

    sentence_index = VectorStoreIndex(sentence_nodes, service_context=service_context)
    return system_prompt, llm, sentence_index


# Extract named entities from a question using spaCy, returning a list of named entities
def extract_named_entities(question, nlp_spacy):
    doc = nlp_spacy(question)
    return [ent.text for ent in doc.ents]


# Normalize text by removing articles, punctuation, and standardizing whitespace
def normalize_text(s):
    import string
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text, flags=re.UNICODE)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    s = str(s)
    return white_space_fix(remove_articles(remove_punc(lower(s))))


# Compute the exact match score between the prediction and the truth
def compute_exact_match(prediction, truth):
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    prediction = normalize_text(prediction)
    prediction_tokens = tokenizer.tokenize(prediction)

    if isinstance(truth, list) and len(truth) == 0:
        truth = []
    else:
        truth = normalize_text(truth)
    truth_tokens = tokenizer.tokenize(truth)

    if "please select" in prediction:
        return 0
    if not truth and "!no answer in the index!" in prediction:
        return 1
    if truth and "!no answer in the index!" in prediction:
        return 0
    if all(token in prediction_tokens for token in truth_tokens):
        return 1
    return 0


# Compute the F1 score between the prediction and the truth.
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


def compute_metrics(response, gold_answers, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven):
    """
    Compute various evaluation metrics for the response.
    Args:
        response (str): The predicted answer.
        gold_answers (list or str): The ground truth answers.
        cnt_unanswCorr (int): Counter for correctly identified unanswerable questions.
        cnt_unanswIncorr (int): Counter for incorrectly answered unanswerable questions.
        cnt_answNotGiven (int): Counter for cases where the answer was not given but expected.
    """
    if isinstance(gold_answers, str):
        try:
            gold_answers = ast.literal_eval(gold_answers)
        except (ValueError, SyntaxError):
            gold_answers = [gold_answers]

    if isinstance(gold_answers, list):
        gold_answers = [answer.lower() for answer in gold_answers]
    if isinstance(response, list):
        response = ' '.join(str(r) for r in response)
    elif isinstance(response, str):
        pass
    if hasattr(response, 'response'):
        response = str(response.response)

    try:
        if "!no answer in the index!" in response and not gold_answers:
            cnt_unanswCorr += 1
            return None, None, None, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven
        elif len(response) > 0 and all(len(ref) == 0 for ref in gold_answers):
            cnt_unanswIncorr += 1
            return None, None, None, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven
        elif "!no answer in the index!" in response and any(len(ref) > 0 for ref in gold_answers):
            cnt_answNotGiven += 1
            return None, None, None, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven
        else:
            sacrebleu_score = datasets.load_metric("sacrebleu").compute(predictions=[response],
                                                                        references=[[ref] for ref in gold_answers])
            rouge_score = datasets.load_metric("rouge").compute(predictions=[response], references=gold_answers)
            bertscore_result = datasets.load_metric("bertscore").compute(predictions=[response],
                                                                         references=gold_answers, lang="en")
            return sacrebleu_score, rouge_score, bertscore_result, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven
    except Exception as e:
        print("Error occurred during score computation:", e)
        return None, None, None, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven


# Main function to preprocess the dataset, set up the Llama Index, and evaluate responses.
def main():
    squad_v2_train, squad_v2_train_unique_contexts = preprocess_dataset()
    system_prompt, llm, sentence_index = setup_llama_index(squad_v2_train_unique_contexts)

    query_engine = sentence_index.as_query_engine(
        similarity_top_k=2,
        node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")],
    )

    csv_file = "sampled_balanced_df.csv"
    resp_df = pd.read_csv(csv_file)
    results = []
    counter = 0

    for index, row in resp_df.iterrows():
        question = row["question"]
        print("question:", question)

        response = query_engine.query(question)
        print("response: ", response)

        window = response.source_nodes[0].node.metadata["window"]
        sentence = response.source_nodes[0].node.metadata["original_text"]
        print(f"Window: {window}")
        print("------------------")
        print(f"Original Sentence: {sentence}")

        additional_info = window
        cnt_unanswCorr = 0
        cnt_unanswIncorr = 0
        cnt_answNotGiven = 0
        expected_context = row["context"]
        retrieved_context = additional_info
        id = row["id"]

        squad_row = squad_v2_train[squad_v2_train["id"] == id]
        if squad_row.empty:
            print(f"No matching question found in the preprocessed Squad dataset for: {question}")
            continue

        gold_answers = squad_row['answers'].iloc[0]['text']
        print("here GOLD ANSWER", gold_answers)

        em_score = compute_exact_match(response, gold_answers)
        f1_score = compute_f1(response, gold_answers)
        print("em", em_score, "f1", f1_score)

        sacrebleu_score, rouge_score, bertscore_result, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven = compute_metrics(
            response, gold_answers, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven
        )

        print("sacrebleu_score", sacrebleu_score)
        print("rouge_score", rouge_score)
        print("bertscore_result", bertscore_result)
        print("cnt_unanswCorr", cnt_unanswCorr)
        print("cnt_unanswIncorr", cnt_unanswIncorr)
        print("cnt_answNotGiven", cnt_answNotGiven)

        normalized_expected_context = normalize_text(expected_context)
        normalized_retrieved_context = normalize_text(retrieved_context)
        correct_context = int(normalized_expected_context in normalized_retrieved_context)
        print("correct context", correct_context)

        result = {
            "id": row["id"],
            "category": row["Category"],
            "title": row["title"],
            "question": question,
            "LLMresponse": response,
            "gold_answer": gold_answers,
            "retrieved_context": retrieved_context,
            "expected_context": expected_context,
            "correct_context": correct_context,
            "EM": em_score,
            "F1": f1_score,
            "sacrebleu_score": sacrebleu_score,
            "rouge_score": rouge_score,
            "bertscore_result": bertscore_result,
            "cnt_unanswCorr": cnt_unanswCorr,
            "cnt_unanswIncorr": cnt_unanswIncorr,
            "cnt_answNotGiven": cnt_answNotGiven,
        }
        if result:
            results.append(result)
            print("length of results:", len(results))
        print(
            "---------------------------------------------------------------------------------------------------------------------------------------")

        counter += 1
        if counter % 30 == 0:
            filename = f"eval_META_NR_{counter}.csv"
            intermediate_df = pd.DataFrame(results)
            intermediate_df.to_csv(filename, index=False)
            print("df saved to ", counter)

    result_df = pd.DataFrame(results)
    result_df.to_csv("eval_METADATA_FINAL.csv", index=False)
    print("Evaluation of all questions from the CSV file completed, saved in 'eval_METADATA_FINAL.csv'.")


if __name__ == "__main__":
    main()
