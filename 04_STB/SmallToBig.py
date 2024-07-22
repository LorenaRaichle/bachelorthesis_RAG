import pandas as pd
from datasets import load_dataset
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import IndexNode
from transformers import BertTokenizer
import ast
import string
import re
import subprocess


# Define functions
def preprocess_dataset():
    squad_v2_train = load_dataset('squad_v2', split="train")
    squad_v2_train = squad_v2_train.to_pandas()[['id', 'context', 'title', 'question', 'answers']]
    squad_v2_train_unique_contexts = squad_v2_train.drop_duplicates(subset='context', keep='first')
    squad_v2_train['context_len_words'] = squad_v2_train['context'].apply(lambda x: len(str(x).split(' ')))
    squad_v2_train['answer_len_words'] = squad_v2_train['answers'].apply(lambda x: len(str(x).split(' ')))
    return squad_v2_train


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def compute_exact_match(prediction, truth):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    prediction_tokens = tokenizer.tokenize(normalize_text(prediction))
    truth_tokens = tokenizer.tokenize(normalize_text(truth))
    if "please select" in prediction:
        return 0
    if not truth and "!no answer in the index!" in prediction:
        return 1
    if truth and "!no answer in the index!" in prediction:
        return 0
    if all(token in prediction_tokens for token in truth_tokens):
        return 1
    return 0


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    if not pred_tokens or not truth_tokens:
        return int(pred_tokens == truth_tokens)
    common_tokens = set(pred_tokens) & set(truth_tokens)
    if not common_tokens:
        return 0
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)


def compute_metrics(response, gold_answers, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven):
    if isinstance(gold_answers, str):
        try:
            gold_answers = ast.literal_eval(gold_answers)
        except (ValueError, SyntaxError):
            gold_answers = [gold_answers]
    gold_answers = [answer.lower() for answer in gold_answers]
    if isinstance(response, list):
        response = ' '.join(str(r) for r in response)
    elif hasattr(response, 'response'):
        response = str(response.response)
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
        # Assume sacrebleu, rouge, and bertscore are loaded elsewhere or update to use alternative methods
        # Example metric calculation
        sacrebleu_score = None
        rouge_score = None
        bertscore_result = None
        return sacrebleu_score, rouge_score, bertscore_result, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven


def create_vector_index(base_nodes, service_context):
    sub_chunk_sizes = [128, 256, 512]
    sub_node_parsers = [
        SimpleNodeParser.from_defaults(chunk_size=c, chunk_overlap=c // 4) for c in sub_chunk_sizes
    ]
    all_nodes = []
    for base_node in base_nodes:
        for parser in sub_node_parsers:
            sub_nodes = parser.get_nodes_from_documents([base_node])
            sub_inodes = [IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes]
            all_nodes.extend(sub_inodes)
        original_node = IndexNode.from_text_node(base_node, base_node.node_id)
        all_nodes.append(original_node)
    vector_index = VectorStoreIndex(all_nodes, service_context=service_context)
    vector_retriever = vector_index.as_retriever(similarity_top_k=2)
    retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever},
        node_dict={n.node_id: n for n in all_nodes},
        verbose=True
    )
    query_engine = RetrieverQueryEngine.from_args(retriever, service_context=service_context)
    return retriever, query_engine


# Main function
def main():
    # Load dataset and setup models
    squad_v2_train = preprocess_dataset()
    total_contexts = squad_v2_train["context"].tolist()
    combined_text = "\n\n".join(total_contexts)
    combined_document = Document(text=combined_text)
    node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)
    base_nodes = node_parser.get_nodes_from_documents([combined_document])
    for idx, node in enumerate(base_nodes):
        node.id_ = f"node-{idx}"

    api_token = "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"
    subprocess.run(['huggingface-cli', 'login', '--token', api_token])
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

    retriever, query_engine = create_vector_index(base_nodes, service_context)

    # Load and process CSV
    csv_file = "sampled_balanced_df.csv"
    resp_df = pd.read_csv(csv_file)
    results = []
    counter = 0

    for index, row in resp_df.iterrows():
        question = row["question"]
        response, retrieved_context = small_to_big_retrieval(retriever, query_engine, question)

        id = row["id"]
        result = {
            "id": id,
            "category": row["Category"],
            "title": row["title"],
            "question": question,
            "LLMresponse": str(response),
            "retrieved_context": retrieved_context
        }
        results.append(result)
        counter += 1
        if counter % 30 == 0:
            filename = f"eval_STB_NR_{counter}.csv"
            pd.DataFrame(results).to_csv(filename, index=False)

    pd.DataFrame(results).to_csv("eval_STB_1.csv", index=False)

    # Metrics Computation
    resp_df = pd.read_csv("eval_STB_1.csv")
    updated_results = []
    for index, row in resp_df.iterrows():
        cnt_unanswCorr = 0
        cnt_unanswIncorr = 0
        cnt_answNotGiven = 0
        question = row["question"]
        id = row["id"]
        retrieved_context = row["retrieved_context"]
        response = row["LLMresponse"]
        squad_row = squad_v2_train.loc[squad_v2_train["id"] == id]
        if not squad_row.empty:
            gold_answers_dict = squad_row.iloc[0]['answers']
            gold_answers = gold_answers_dict.get('text', [])
            expected_context = squad_row.iloc[0]['context']
            em_score = compute_exact_match(response, gold_answers)
            f1_score = compute_f1(response, gold_answers)
            normalized_expected_context = normalize_text(expected_context)
            normalized_retrieved_context = normalize_text(retrieved_context)
            correct_context = int(normalized_expected_context in normalized_retrieved_context)
            sacrebleu_score, rouge_score, bertscore_result, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven = compute_metrics(
                response, gold_answers, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven)
            updated_results.append({
                "EM": em_score,
                "F1": f1_score,
                "correct_context": correct_context,
                "gold_answer": gold_answers,
                "expected_context": expected_context,
                "sacrebleu_score": sacrebleu_score,
                "rouge_score": rouge_score,
                "bertscore_result": bertscore_result,
                "cnt_unanswCorr": cnt_unanswCorr,
                "cnt_unanswIncorr": cnt_unanswIncorr,
                "cnt_answNotGiven": cnt_answNotGiven,
            })

    final_df = pd.concat([resp_df, pd.DataFrame(updated_results)], axis=1)
    final_df.to_csv("STB_final_eval.csv", index=False)
    print("Evaluation completed, results saved in 'STB_final_eval.csv'.")


if __name__ == "__main__":
    main()
