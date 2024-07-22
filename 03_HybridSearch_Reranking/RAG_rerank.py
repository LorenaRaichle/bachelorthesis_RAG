# -*- coding: utf-8 -*-

"""RAG Hybrid Search & Re-ranking implementation with Llama2 Baseline Evaluation
"""

import numpy as np
import pandas as pd
import time
import subprocess
import string
import ast
from datasets import load_dataset
from cohere.errors import TooManyRequestsError
import cohere
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext, Document
from llama_index.llms.huggingface import HuggingFaceInferenceAPI, HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from transformers import BertTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.evaluation import RetrieverEvaluator, FaithfulnessEvaluator, RelevancyEvaluator
import datasets
import sacrebleu
import rouge
import bertscore

# Define constants
MAX_RETRIES = 5  # Define a constant for maximum retries
API_TOKEN_HUGGINGFACE = "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"  # Replace with your actual token
API_TOKEN_COHERE = "xxx"  # Replace with your actual Cohere token
PINECONE_API_KEY = "xxx"  # Replace with your actual Pinecone API key


# Helper function for querying with retry mechanism
def query_with_retry(query_engine, query, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            response = query_engine.query(query)
            return response
        except TooManyRequestsError as e:
            print(f"TooManyRequestsError: {e}. Retrying in 60 seconds...")
            time.sleep(60)


# Helper function for reranking with retry mechanism
def rerank_with_retry(co, query, documents, top_n, model, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            rerank_results = co.rerank(query=query, documents=documents, top_n=top_n, model=model)
            return rerank_results
        except TooManyRequestsError as e:
            print(f"TooManyRequestsError: {e}. Retrying in 60 seconds...")
            time.sleep(60)
    raise


# Function to preprocess data and create the necessary indexes
def preprocess_data():
    """Load and preprocess data for the RAG system."""
    # Load dataset and preprocess
    squad_v2_train = load_dataset('squad_v2', split="train")
    squad_v2_train = squad_v2_train.to_pandas()[['id', 'context', 'title', 'question', 'answers']]
    print("Number of contexts before dropping duplicates:", squad_v2_train["context"].shape[0])
    squad_v2_train_unique_contexts = squad_v2_train.drop_duplicates(subset='context', keep='first')
    print("Number of contexts after dropping duplicates:", squad_v2_train_unique_contexts["context"].shape[0])
    total_contexts = squad_v2_train_unique_contexts["context"].tolist()

    # Define prompts for system and query wrapper
    system_prompt = """
    You are a Q&A assistant. Please answer the questions in one sentence, if the answer is available in the index. Otherwise indicate "!no answer in the index!". Please consider the following additional information: {additional_info}.
    """
    query_wrapper_prompt = SimpleInputPrompt("{query_str}")

    # Log in to Hugging Face
    subprocess.run(['huggingface-cli', 'login', '--token', API_TOKEN_HUGGINGFACE])
    print("1.2 Hugging Face login done")

    # Initialize LLM and embedding model
    llm = HuggingFaceInferenceAPI(
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

    # Compute embeddings for contexts
    context_embeddings = [embed_model.get_text_embedding(doc) for doc in total_contexts]
    print("Number of context embeddings:", len(context_embeddings))
    context_embeddings = np.array(context_embeddings)
    print("Shape of context embeddings:", context_embeddings.shape)

    # Create service context
    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embed_model
    )
    print("2. Service context created")

    # Tokenization and preprocessing
    def remove_punctuation(tokens):
        return [token for token in tokens if token not in string.punctuation]

    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    total_tokens = [remove_punctuation(tokenizer.tokenize(context.lower())) for context in total_contexts]

    print("Tokens for one context:", total_tokens[0])

    # Initialize Pinecone and create indexes
    pinecone_instance = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_host = "https://rag-project-lr-vwolzqa.svc.apw5-4e34-81fa.pinecone.io"

    existing_index_names = pinecone_instance.list_indexes().names()
    if "rag-rerank" not in existing_index_names:
        pinecone_instance.create_index(
            name='rag-rerank',
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    pinecone_index = pinecone_instance.Index("rag-rerank")
    print("Pinecone index stats:", pinecone_index.describe_index_stats())
    dimensionality_index = pinecone_index.describe_index_stats()["dimension"]
    print("Dimensionality expected by the Pinecone index:", dimensionality_index)

    vectors_to_upsert = [(str(i), embedding.tolist(), {"tokens": tokens}) for i, (embedding, tokens) in
                         enumerate(zip(context_embeddings, total_tokens))]

    # Upsert vectors into Pinecone index
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        pinecone_index.upsert(vectors=batch)
    print("Upsert in Pinecone done")

    if "rag-rerank-query" not in existing_index_names:
        pinecone_instance.create_index(
            name='rag-rerank-query',
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    pinecone_index_query = pinecone_instance.Index("rag-rerank-query")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index_query, host=pinecone_host)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Convert dataset to documents
    squad_v2_train_dict_list = squad_v2_train.to_dict(orient='records')

    def create_document_from_dict(data_dict):
        metadata = data_dict.pop("metadata", {})
        doc = Document(text=data_dict["text"], metadata=metadata, doc_id=data_dict["id"],
                       question=data_dict["question"])
        return doc

    documents = [create_document_from_dict(data_dict) for data_dict in squad_v2_train_dict_list]

    print("Type of documents:", type(documents))
    print("Number of documents:", len(documents))

    # Create the index from documents
    index = VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context,
                                            service_context=service_context)
    print("Type of index:", type(index))
    print("Value of index:", index)
    print("3. Index created")

    return index, embed_model, pinecone_index, total_contexts


# Function to retrieve and rerank based on the query
def retrieve_and_rerank(query: str, embedding_model, index, top_k: int, rerank_top_k: int):
    """Retrieve and rerank documents based on the query."""
    question_embedding = embedding_model.get_text_embedding(query)
    if not isinstance(question_embedding, list):
        question_embedding = question_embedding.tolist()
    print("Query embedding done")
    print("Query:", query)

    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    total_tokens_query = [token for token in remove_punctuation(tokenizer.tokenize(query.lower()))]
    print("Current tokens of query:", total_tokens_query)

    # Retrieve top-k documents
    response_vecs = index.query(
        vector=question_embedding,
        top_k=top_k,
        includeMetadata=True,
        filter={'tokens': {'$in': total_tokens_query}},
        namespace=""
    )
    top_ids = [match["id"] for match in response_vecs["matches"]]
    print("Returned top_k ids:", top_ids)
    for id in top_ids:
        print(f'{id}: {documents[int(id)]["question"]}')

    # Rerank the retrieved documents
    top_contexts = [total_contexts[int(id)] for id in top_ids]
    rerank_results = rerank_with_retry(cohere, query, top_contexts, top_n=rerank_top_k, model="rerank-model")

    print("Rerank results:", rerank_results)
    return rerank_results


# Main execution
def main():
    """Main function to execute the workflow."""
    index, embed_model, pinecone_index, total_contexts = preprocess_data()

    csv_file = "sampled_balanced_df.csv"
    sampled_df = pd.read_csv(csv_file)
    print(sampled_df.shape)
    sampled_df = sampled_df[720:]
    results = []
    counter = 900

    for idx, row in sampled_df.iterrows():
        question = row["question"]
        print("QUESTION:", question)
        category = row["Category"]
        id = row["id"]
        squad_row = squad_v2_train[squad_v2_train["id"] == id]
        if squad_row.empty:
            print(f"No matching question found in the preprocessed Squad dataset for: {question}, {id}")
            continue
        gold_answers = squad_row['answers'].iloc[0]['text']
        additional_info, top_ids, reranked_order = retrieve_and_rerank(question, embed_model, pinecone_index, top_k=7,
                                                                       rerank_top_k=3)
        print("additional info: ", additional_info)
        replaced_prompt = system_prompt.format(additional_info=additional_info)
        print("System Prompt with Additional Info:")
        print(replaced_prompt)
        query_engine = index.as_query_engine()
        final_response = query_with_retry(query_engine, question)
        print("Final LLM Response including reranked context: ", final_response)
        result = {
            "id": row["id"],
            "category": category,
            "title": row["title"],
            "question": question,
            "LLMresponse": final_response,
            "gold_answer": gold_answers,
            "expected_context": row["context"] if not row.empty else None,
            "top_ids": top_ids,
            "reranked_order": reranked_order,
            "additional_info/retrieved context": additional_info,
        }
        if result:
            results.append(result)
            print("length of results:", len(results))
        print(".....................................NEXT........................")
        counter += 1
        if counter % 30 == 0:
            filename = f"eval_rerank_nr_{counter}.csv"
            intermediate_df = pd.DataFrame(results)
            intermediate_df.to_csv(filename, index=False)
            print("df saved to ", counter)

    result_df = pd.DataFrame(results)
    result_df.to_csv("eval_rerank_balanced_FINAL.csv", index=False)
    print("Evaluation of all questions from the CSV file completed, saved in 'eval_rerank_balanced_FINAL.csv'.")


if __name__ == "__main__":
    main()
