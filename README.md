# llm4judge



command on all doc curpos:

python3 main.py --model_id "meta-llama/Meta-Llama-3-8B-Instruct" --test_qrel_path "./data/llm4eval_dev_qrel_2024.txt" --queries_path "./data/llm4eval_query_2024.txt" --docs_path "./data/llm4eval_document_2024.jsonl" --chunk_size 1000 --result_file_path "./results/dev_baseline.run" --problematic_passages_path "./results/problematic_passages.txt" --generative_error_file_path "./results/generative_error_queries_and_responses_dev_baseline.txt" --score_order_in_prompt "3210" --store_top_k_doc_scores 100 --exam


command on qrel only:

python3 main.py --model_id "meta-llama/Meta-Llama-3-8B-Instruct" --test_qrel_path "./data/llm4eval_dev_qrel_2024.txt" --queries_path "./data/llm4eval_query_2024.txt" --docs_path "./data/llm4eval_document_2024.jsonl" --result_file_path "./results/dev_baseline_qrel.txt" --score_order_in_prompt "0123" 

python3 cohens_kappa_multiclass.py "./data/llm4eval_dev_qrel_2024.txt" "./results/dev_baseline_qrel_prompt order: 0123.txt" "./analysis/dev_baseline_qrel_prompt order: 0123.txt"


command on iterative prompting with 3210 as order:

python3 main.py --model_id "meta-llama/Meta-Llama-3-8B-Instruct" --test_qrel_path "./data/llm4eval_dev_qrel_2024.txt" --queries_path "./data/llm4eval_query_2024.txt" --docs_path "./data/llm4eval_document_2024.jsonl" --result_file_path "./results/dev_iterative_prompting_qrel.txt" --score_order_in_prompt "3210" --iterative_prompts True


python3 cohens_kappa_multiclass.py "./data/llm4eval_dev_qrel_2024.txt" "./results/dev_iterative_prompting_qrel.txt" "./analysis/dev_iterative_prompting_qrel.txt"



command on decompsing prompt:
python3 main.py --model_id "meta-llama/Meta-Llama-3-8B-Instruct" --test_qrel_path "./data/llm4eval_dev_qrel_2024.txt" --queries_path "./data/llm4eval_query_2024.txt" --docs_path "./data/llm4eval_document_2024.jsonl" --result_file_path "./results/dev_decomposed_relavance_qrel.txt" --score_order_in_prompt "3210" --decomposed_relavance True

python3 cohens_kappa_multiclass.py "./data/llm4eval_dev_qrel_2024.txt" "./results/dev_decomposed_relavance_qrel.txt" "./analysis/dev_decomposed_relavance_qrel.txt"



sun then decomposed:
python3 main.py --model_id "meta-llama/Meta-Llama-3-8B-Instruct" --test_qrel_path "./data/llm4eval_dev_qrel_2024.txt" --queries_path "./data/llm4eval_query_2024.txt" --docs_path "./data/llm4eval_document_2024.jsonl" --result_file_path "./results/dev_sun_then_decomposed_relavance_qrel.txt" --sunprompt_then_decomposed True


analysis command:
python3 analysis_qrels.py "./data/llm4eval_dev_qrel_2024.txt" "./results/llm4eval_dev_qrel_results_baseline.txt" "./analysis/dev_baseline_analysis.txt"

python3 analysis_qrels.py "./data/llm4eval_dev_qrel_2024.txt" "./results/dev_baseline_qrel_prompt order: 3210.txt" "./analysis/dev_baseline_qrel_prompt order: 3210.txt"







python3 main.py --model_id "meta-llama/Meta-Llama-3-8B-Instruct" --test_qrel_path "./data/llm4eval_test_qrel_2024.txt" --queries_path "./data/llm4eval_query_2024.txt" --docs_path "./data/llm4eval_document_2024.jsonl" --result_file_path "./results/test_decomposed_relavance_qrel.txt" --score_order_in_prompt "3210" --decomposed_relavance True
