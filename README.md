# Information Retrieval Evaluation with LLMs

## Project Overview

This project investigates alternative methods for evaluating Information Retrieval (IR) systems by directly prompting Large Language Models (LLMs) to assign relevance labels to passages associated with queries. The goal is to address the limitations of traditional human-annotated relevance labels, which can be costly and biased. This project explores two main hypotheses:

- **Hypothesis 1:** Breaking down relevance into criteria like exactness, coverage, topicality, and contextual fit might improve LLM evaluation. Various approaches are tested to prompt LLMs for criteria-level grades, which are then aggregated into a final relevance label.
- **Hypothesis 2:** Differences in linguistic style between queries and passages may affect automatic relevance label prediction. This hypothesis is tested by synthesizing query-style summaries of passages and using those summaries to assess relevance.

This project includes an empirical evaluation using data from the **LLMJudge Challenge (Summer 2024)**, where our "Four Prompts" approach achieved the highest score in **Kendall's tau**.

## Repository Structure

- **data/**: Contains the query, document, and relevance files.
  - `llm4eval_dev_qrel_2024.txt`: Development relevance labels.
  - `llm4eval_query_2024.txt`: List of queries.
  - `llm4eval_document_2024.jsonl`: List of documents to evaluate.
  
- **results/**: Output folder for the results of different evaluation strategies.
  - Contains various result files based on different prompting and evaluation strategies.
  
- **analysis/**: Folder for the analysis of the evaluation results.
  - Includes scripts to analyze relevance scores and calculate agreement metrics like Cohen's Kappa.

## Dependencies

- Python 3.x
- Meta's Llama LLMs
- Required Python packages: `numpy`, `scikit-learn`, etc. (see `requirements.txt`)

## How to Run

### Baseline Evaluation with `3210` Scoring

```bash
python3 main.py --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
  --test_qrel_path "./data/llm4eval_dev_qrel_2024.txt" \
  --queries_path "./data/llm4eval_query_2024.txt" \
  --docs_path "./data/llm4eval_document_2024.jsonl" \
  --chunk_size 1000 \
  --result_file_path "./results/dev_baseline.run" \
  --problematic_passages_path "./results/problematic_passages.txt" \
  --generative_error_file_path "./results/generative_error_queries_and_responses_dev_baseline.txt" \
  --score_order_in_prompt "3210" \
  --store_top_k_doc_scores 100 \
  --exam
