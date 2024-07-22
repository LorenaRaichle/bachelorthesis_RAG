
"""
###R AG baseline implementation
- using the Llama-2 model along with Pinecone for vector storage and retrieval
- system retrieves relevant context from a dataset and uses it to generate answers to questions
- model performance evaluation using different metrics
"""

import subprocess
from collections import defaultdict
import pandas as pd

from datasets import load_dataset


from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext, Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt


# Load and preprocess the SQuAD dataset
def preprocess_dataset():
    squad_v2_train = load_dataset('squad_v2', split="train")
    print("Dataset loaded:", type(squad_v2_train))
    squad_v2_train = squad_v2_train.to_pandas()[['id', 'context', 'title', 'question', 'answers']]
    unique_questions = squad_v2_train["question"].nunique()
    print("Total questions:", len(squad_v2_train["question"]))
    print("Unique questions:", unique_questions)

    squad_v2_train_unique_contexts = squad_v2_train.drop_duplicates(subset='context', keep='first')

    # squad_v2_train = squad_v2_train[:50]  # Subset for faster processing
    squad_v2_train['context_len_words'] = squad_v2_train['context'].apply(lambda x: len(str(x).split(' ')))
    squad_v2_train['answer_len_words'] = squad_v2_train['answers'].apply(lambda x: len(str(x).split(' ')))

    print("Context length stats:", squad_v2_train['context_len_words'].describe())
    print("Answer length stats:", squad_v2_train['answer_len_words'].describe())

    return squad_v2_train, squad_v2_train_unique_contexts


# Set up the Llama-2 model, insert SQuAD context data into Pinecone
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
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model) # chunk_size=1024,

    pinecone_instance = Pinecone(api_key="xxx")

    existing_index = pinecone_instance.list_indexes().names()
    if "rag-baseline" not in existing_index:
        pinecone_instance.create_index(
            name='rag-baseline',
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    pinecone_index = pinecone_instance.Index("rag-baseline")

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index) # host=pinecone_host)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    contexts_and_ids = list(zip(squad_v2_train["context"].tolist(), squad_v2_train["id"].tolist()))
    combined_text = "\n\n".join([context for context, _ in contexts_and_ids])
    combined_document = Document(text=combined_text)

    node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)
    base_nodes = node_parser.get_nodes_from_documents([combined_document])

    node_id_to_node = {}
    vectors_to_upsert = []
    for idx, node in enumerate(base_nodes):
        node.id_ = f"node-{idx}"
        node_id_to_node[node.id_] = node
        # Generate embedding for the node text
        embedding = embed_model.get_text_embedding(node.text)
        # Prepare vector data for upserting
        vector_data = (node.id_, embedding)
        vectors_to_upsert.append(vector_data)

    # Upsert vectors to Pinecone
    batch_size = 100  # Adjust the batch size as needed
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        pinecone_index.upsert(vectors=batch)

    original_id_to_node_ids = defaultdict(list)
    for idx, node in enumerate(base_nodes):
        for context, original_doc_id in contexts_and_ids:
            if context in node.text:
                original_id_to_node_ids[original_doc_id].append(node.id_)

    index = VectorStoreIndex(base_nodes, service_context=service_context)
    base_retriever = index.as_retriever(similarity_top_k=1)

    index_stats = pinecone_index.describe_index_stats()
    num_vectors = index_stats['total_vector_count']
    print(f"Number of vectors in the index: {num_vectors}")
    return base_retriever, service_context, original_id_to_node_ids, node_id_to_node, llm ,system_prompt

# define functions for evaluation
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


from transformers import BertTokenizer
def compute_exact_match(prediction, truth):
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    prediction = normalize_text(prediction)
    prediction_tokens = tokenizer.tokenize(prediction)
    truth = normalize_text(truth)
    truth_tokens = tokenizer.tokenize(truth)
    # If truths is empty and prediction is "no answer in the index", return 1
    if "please select" in prediction:
        return 0
    if not truth and prediction == "no answer in the index":
        return 1
    if truth and prediction == "no answer in the index":
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



# EVALUATION

import datasets
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.evaluation import RetrieverEvaluator, FaithfulnessEvaluator, RelevancyEvaluator

# Compute various evaluation metrics for the model's response.
def compute_metrics(response, gold_answers, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven):
    print("resp", response)
    print("gold answ", gold_answers)
    gold_answers= [str(answer).lower() for answer in gold_answers]
    # references = [str(answer).lower() for answer in gold_answers]
    # if isinstance(response, str):
    # response = [response]
    if isinstance(response, list):
        response = [str(r) for r in response]
    if hasattr(response, 'response'):
        response = str(response.response)
    # else:
    #   response_list = [str(response)]
    try:
        #  response_bleu = response_list
        #  reference_bleu = [references]
        # print("resp bleu, :", response_bleu)
        # print("ref belu:", reference_bleu)
        # if any("!no answer in the index!" in resp for resp in response_bleu) and all(len(ref) == 0 for ref in reference_bleu):
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


# Retrieve context, generate answer, and evaluate the results.
def retrieve_and_evaluate(system_prompt, question, gold_answers, row, category, squad_v2_train, base_retriever, service_context, original_id_to_node_ids, node_id_to_node, llm, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven):

    retrievals = base_retriever.retrieve(question)
    contexts = [n.text for n in retrievals]
    retrieved_context = "\n\n".join(contexts)

    # Replace the placeholder in the system prompt with the retrieved context
    system_prompt = system_prompt.format(additional_info=retrieved_context)
    llm.system_prompt = system_prompt

    query_engine = RetrieverQueryEngine.from_args(base_retriever, service_context=service_context)
    response = query_engine.query(question)
    if gold_answers:
        em_score = max(compute_exact_match(response, gold_answer) for gold_answer in gold_answers)
        f1_score = max(compute_f1(response, gold_answer) for gold_answer in gold_answers)
    else:
        em_scor e =0
        f1_scor e =0
    retriever_evaluator = RetrieverEvaluator.from_metric_names(["mrr", "hit_rate"], retriever=base_retriever)
    squad_row = squad_v2_train[squad_v2_train["question"] == question]
    original_ids = squad_row["id"].tolist()

    try:
        expected_ids = [node_id for original_id in original_ids for node_id in original_id_to_node_ids[original_id]]
    except KeyError as e:
        print(f"Skipping question due to missing ID mapping: {e}")
        return
    retrieved_ids = [retrieval.id_ for retrieval in retrievals]
    eval_result = retriever_evaluator.evaluate(question ,retrieved_ids, expected_ids)
    mrr = getattr(eval_result, 'mrr', 0.0)
    hit_rate = getattr(eval_result, 'hit_rate', 0.0)

    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
    relevancy_evaluator = RelevancyEvaluator(llm=llm)
    eval_result_faithfulness = faithfulness_evaluator.evaluate_response(query=question, response=response)
    eval_result_relevancy = relevancy_evaluator.evaluate_response(query=question, response=response)

    sacrebleu_score, rouge_score, bertscore_result, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven = compute_metrics \
        (response, gold_answers, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven)
    result = {
        "id": row["id"],
        "category": category,
        "context": row["context"],
        "title": row["title"],
        "question": question,
        "LLMresponse": response,
        "gold_answer": gold_answers,
        "retrieved_context": retrieved_context,
        "expected_context": row["context"] if not row.empty else None,
        "EM": em_score,
        "F1": f1_score,
        "MRR": mrr,
        "HitRate": hit_rate,
        "faithfulness": eval_result_faithfulness.passing,
        "sacrebleu_score": sacrebleu_score,
        "rouge_score": rouge_score,
        "bertscore_result": bertscore_result,
        "relevancy": eval_result_relevancy.passing,
        "cnt_unanswCorr": cnt_unanswCorr,
        "cnt_unanswIncorr": cnt_unanswIncorr,
        "cnt_answNotGiven": cnt_answNotGiven,
    }
    return result, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven

if __name__ == "__main__":
    squad_v2_train, squad_v2_train_unique_contexts = preprocess_dataset()
    base_retriever, service_context, original_id_to_node_ids, node_id_to_node, llm, system_prompt = setup_llama_index \
        (squad_v2_train_unique_contexts)
    csv_file = "sampled_balanced_df.csv"
    sampled_df = pd.read_csv(csv_file)

    results =[]
    sacrebleu = datasets.load_metric("sacrebleu")
    rouge = datasets.load_metric("rouge")
    bertscore = datasets.load_metric("bertscore")

    # sampled_df = sampled_df[:5]
    # Evaluate each question in the CSV file
    for index, row in sampled_df.iterrows():
        cnt_unanswCorr = 0
        cnt_unanswIncorr = 0
        cnt_answNotGiven = 0
        question = row["question"]
        print("QUESTION:" ,question)
        category = row["Category"]
        id =  row["id"]
        # Retrieve the gold answers from the preprocessed Squad dataset based on the question
        squad_row = squad_v2_train[squad_v2_train["id"] == id]
        if squad_row.empty:
            print(f"No matching question found in the preprocessed Squad dataset for: {question}")
            continue
        gold_answers = squad_row['answers'].iloc[0]['text']
        # Evaluate the retrieved answers using LLAMA index and other evaluation metrics
        result, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGive n= retrieve_and_evaluate(system_prompt, question, gold_answers, row, category, squad_v2_train, base_retriever, service_context, original_id_to_node_ids, node_id_to_node, llm, cnt_unanswCorr, cnt_unanswIncorr, cnt_answNotGiven)
        if result:
            results.append(result)
            print("length of results:", len(results))
    result_df = pd.DataFrame(results)
    # Save the results to a CSV file
    result_df.to_csv("eval_baseline_balanced.csv", index=False)
    print("Evaluation of all questions from the CSV file completed, saved in 'eval_baseline_balanced.csv'.")
