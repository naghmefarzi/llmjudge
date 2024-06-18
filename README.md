# llm4judge



command on all doc curpos:
python3 main.py --model_id "meta-llama/Meta-Llama-3-8B-Instruct" --test_qrel_path "./data/llm4eval_dev_qrel_2024.txt" --queries_path "./data/llm4eval_query_2024.txt" --docs_path "./data/llm4eval_document_2024.jsonl" --chunk_size 1000 --result_file_path "./results/dev_baseline.run" --problematic_passages_path "./results/problematic_passages.txt" --generative_error_file_path "./results/generative_error_queries_and_responses_dev_baseline.txt" --score_order_in_prompt "3210" --store_top_k_doc_scores 100 --exam


command on qrel only:
python3 main.py --model_id "meta-llama/Meta-Llama-3-8B-Instruct" --test_qrel_path "./data/llm4eval_dev_qrel_2024.txt" --queries_path "./data/llm4eval_query_2024.txt" --docs_path "./data/llm4eval_document_2024.jsonl" --result_file_path "./results/dev_baseline_qrel.run" --score_order_in_prompt "3210" 




analysis command:
python3 analysis_qrels.py "./data/llm4eval_dev_qrel_2024.txt" "./results/llm4eval_dev_qrel_results_baseline.txt" "./analysis/dev_baseline_analysis.txt"
