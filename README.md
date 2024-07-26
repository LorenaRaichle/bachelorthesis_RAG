# Bachelorthesis_RAG

## Investigating hallucinations in LLMs: A comparative analysis of Retrieval Augmented Generation enhancement approaches
### Author
Lorena Raichle

This repository contains the code, data, and documentation related to the practical part of my bachelor thesis on investigating hallucinations in Large Language Models. The thesis provides a comparative analysis of various RAG enhancement approaches.

For the implementation, the computational resources provided by the state of Baden-Württemberg through bwHPC and the German Research Foundation (DFG) through grant INST 35/1597-1 FUGG have been used.

### Table of Contents
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Directory Structure](#directory-structure)

### Dependencies
- Python 3.9
- Required Python libraries are listed in `requirements.txt`

### Setup
To set up the project environment, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/LorenaRaichle/bachelorthesis_RAG.git
    ```
2. Navigate to the project directory:
    ```sh
    cd bachelorthesis_RAG
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Directory Structure
A brief overview of the repository's directory structure:

- **01_baselineRAG**  
  Contains Python implementation files and detailed CSV files for every question, expected and retrieved context, and answers, as well as metrics per question.

- **02_metadataFiltering**  
  Contains Python implementation files and detailed CSV files related to Metadata Filtering.

- **03_HybridSearch_Reranking**  
  Contains Python implementation files and detailed CSV files related to Hybrid Search Reranking.

- **04_STB**  
  Contains Python implementation files and detailed CSV files related to the Small-to-Big (STB) approach.

- **0_NaiveLlama2**  
  Contains Python implementation files and detailed CSV files related to the Naive Llama2 approach.

- **dataset_preprocessing**  
  Contains scripts and SQuAD2.0 dataset access for preprocessing datasets.

- **evaluation_StratifiedSampling**  
  Contains scripts and final CSV output for evaluation subset using stratified sampling.

- **run_job.sh**  = script for running job in Cluster

- **comparison_approaches**  
  Contains results comparing different approaches.
  - **mean_scores_per_approach**  
    Contains final CSV files of all metrics (mean and median) scores for all approaches.
  - **plots_all_approaches**  
    Contains plots for all evaluated approaches.
    - **plots_COMPARED_APPROACHES**  
      Contains comparative plots for different approaches.
    - **plots_STB**  
      Contains plots specific to the STB approach.
    - **plots_baselineRAG**  
      Contains plots specific to the Baseline RAG approach.
    - **plots_llama2**  
      Contains plots specific to the Naive Llama2 approach.
    - **plots_metadata**  
      Contains plots specific to Metadata Filtering.
    - **plots_rerank**  
      Contains plots specific to the Hybrid Search Reranking approach.

### Files
- **Llama_res.csv**  
  Results file for the Naive Llama2 approach.

- **STB_res.csv**  
  Results file for the Small-to-Big (STB) approach.

- **baseline_res.csv**  
  Results file for the Baseline RAG approach.

- **metadata_res.csv**  
  Results file for Metadata Filtering.

- **rerank_res.csv**  
  Results file for Hybrid Search Reranking.

- **baseline_scores_extracted.csv**  
  Extracted scores from Baseline RAG results.

- **plot_results.py**  
  Script for generating and saving plots.

- **.gitattributes**  
  Git attributes file for configuring repository settings.

- **README.md**  
  This file, containing information about the project and directory structure.
