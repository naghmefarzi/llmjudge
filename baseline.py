import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os
import gc
import jsonlines
import logging
import transformers
import torch
from heapq import nlargest
from pathlib import Path
from collections import defaultdict
import json

from typing import Tuple, List, Dict, Callable, NewType, Optional, Iterable
from huggingface_hub import login
from transformers import (AutoTokenizer,AutoModelForCausalLM,
                          TextStreamer,pipeline,BitsAndBytesConfig)
import sys
sys.path.append('.')

from model_utils import *
from relevance_scoring import get_relevance_score_baseline, write_top_k_results, process_documents, get_relevance_score_iterative_prompts, get_relevance_score_decomposed_prompts, sun_prompt_then_decomposed,make_query_out_of_passage_relevance
from data_processing import load_data_files, clean_files, process_documents_in_chunks
from prompts import *
from exam_question_generation import generate_question_set



def process_test_qrel_baseline(test_qrel, docid_to_doc, qid_to_query, result_path, pipeline,chunk_size ,generative_error_file_path: Optional[str],problematic_passages_path: Optional[str],system_message:str, k: Optional[int]):
    with open(result_path, 'w') as result_file_top_k, open(result_path.replace(".run","_full.run"), 'w') as result_file_full:
        relevance_scores_agg = defaultdict(dict)
        for documents in process_documents_in_chunks(docid_to_doc, chunk_size):
            processed_documents = process_documents(documents, test_qrel, qid_to_query, pipeline, k, generative_error_file_path, problematic_passages_path, system_message)
            for qidx, doc_scores in processed_documents.items():
                relevance_scores_agg[qidx].update(doc_scores)
        write_top_k_results(relevance_scores_agg, result_file_full , result_file_top_k, k)
    torch.cuda.empty_cache()  # Clear GPU cache if using GPU


def process_test_qrel_baseline_only_qrel(test_qrel, docid_to_doc, qid_to_query, result_path, pipeline ,system_message:str):
    generation_path = result_path.replace("results","generation_errors")
    with open(result_path, 'w') as result_file, open(generation_path, 'w') as generation_errors_file:
        for eachline in tqdm(test_qrel.itertuples(index=True)):
            qidx = eachline.qid
            docidx = eachline.docid
            
            # Generate prompt
            prompt = get_prompt(query=qid_to_query[qidx], passage=docid_to_doc[docidx], pipeline=pipeline)
            
            try:
                # Get relevance score
                pred_score = get_relevance_score_baseline(prompt, pipeline, system_message)
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"CUDA out of memory error for docid {docidx}. Skipping this document.")
                    continue
                else:
                    raise e
            
            # Debugging: Print prompt and score once
            if not hasattr(process_test_qrel_baseline_only_qrel, "called"):
                process_test_qrel_baseline_only_qrel.called = True
                print(prompt)
                print()
                print(f"{qidx} 0 {docidx} {pred_score}\n")
            try:
                int_pred_score = int(pred_score)

                # Write result to file
                result_file.write(f"{qidx} 0 {docidx} {pred_score}\n")
            except:
                generation_errors_file.write(f"{qidx} 0 {docidx} {pred_score}\n")
                
                print(f"pred score for {qidx} and {docidx} is not a number it is: {pred_score}")
                continue
                

def process_passages_based_on_gen_query(test_qrel, docid_to_doc, qid_to_query, result_path, pipeline):
    generation_path = result_path.replace("results","generation_errors")
    logs_path = result_path.replace("results","logs").replace(".txt",".json") 
    cuda_errors_path = result_path.replace("results","cuda_errors")
    passage_to_predicted_query = {}
    with open(result_path, 'w') as result_file, open(generation_path, 'w') as generation_errors_file, open(cuda_errors_path,"w") as cuda_errors_file:
        for eachline in tqdm(test_qrel.itertuples(index=True)):
            qidx = eachline.qid
            docidx = eachline.docid

            try:
                # Get relevance score
                pred_score, passage_to_predicted_query = make_query_out_of_passage_relevance(query=qid_to_query[qidx], passage=docid_to_doc[docidx],pipeline=pipeline,log_file_path=logs_path,qidx=qidx,docidx=docidx, passage_to_predicted_query=passage_to_predicted_query)
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    error = f"CUDA out of memory error for docid {docidx}. Skipping this document."
                    print(error)
                    cuda_errors_file.write(error)
                    
                    result_file.write(f"{qidx} 0 {docidx} 0\n")
                    
                    continue
                else:
                    raise e
            
            # Debugging: Print prompt and score once
            if not hasattr(process_passages_based_on_gen_query, "called"):
                process_passages_based_on_gen_query.called = True
                print(f"{qidx} 0 {docidx} {pred_score}\n")
            try:
                int_pred_score = int(pred_score)
                if 0<=int_pred_score<=3:
                    # Write result to file
                    result_file.write(f"{qidx} 0 {docidx} {pred_score}\n")
                else:
                    result_file.write(f"{qidx} 0 {docidx} 0\n")
                
            except:
                generation_errors_file.write(f"{qidx} 0 {docidx} {pred_score}\n")
                
                print(f"pred score for {qidx} and {docidx} is not a number it is: {pred_score}")
                result_file.write(f"{qidx} 0 {docidx} 0\n")
                
                continue



def process_test_iterative_prompts_only_qrel(test_qrel, docid_to_doc, qid_to_query, result_path:str, pipeline ,system_message:str):
    generation_path = result_path.replace("results","generation_errors")
    logs_path = result_path.replace("results","logs").replace(".txt",".json") 
    decomposed_path = result_path.replace("results","decomposed_scores").replace(".txt",".json") 
    cuda_errors_path = result_path.replace("results","cuda_errors")
    decomposed_scores = {}
    
    with open(result_path, 'w') as result_file, open(generation_path, 'w') as generation_errors_file, open(cuda_errors_path,"w") as cuda_errors_file:
        for eachline in tqdm(test_qrel.itertuples(index=True)):
            qidx = eachline.qid
            docidx = eachline.docid

            try:
                # Get relevance score
                pred_score, decomposed_scores_list_for_one_query = get_relevance_score_iterative_prompts(query=qid_to_query[qidx], passage=docid_to_doc[docidx],pipeline=pipeline,log_file_path=logs_path,system_message=system_message,qidx=qidx,docidx=docidx)
                decomposed_scores[(qidx,docidx)] = decomposed_scores_list_for_one_query
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    error = f"CUDA out of memory error for docid {docidx}. Skipping this document."
                    print(error)
                    cuda_errors_file.write(error)
                    
                    result_file.write(f"{qidx} 0 {docidx} 0\n")
                    
                    continue
                else:
                    raise e
            
            # Debugging: Print prompt and score once
            if not hasattr(process_test_qrel_baseline_only_qrel, "called"):
                process_test_qrel_baseline_only_qrel.called = True
                print(f"{qidx} 0 {docidx} {pred_score}\n")
            try:
                int_pred_score = int(pred_score)
                if 0<=int_pred_score<=3:
                    # Write result to file
                    result_file.write(f"{qidx} 0 {docidx} {pred_score}\n")
                else:
                    result_file.write(f"{qidx} 0 {docidx} 0\n")
                
            except:
                generation_errors_file.write(f"{qidx} 0 {docidx} {pred_score}\n")
                
                print(f"pred score for {qidx} and {docidx} is not a number it is: {pred_score}")
                result_file.write(f"{qidx} 0 {docidx} 0\n")
                
                continue
        # Convert tuples to strings
        decomposed_scores_str_keys = {str(k): v for k, v in decomposed_scores.items()}

        # Save to a JSON file
        with open(decomposed_path, 'w') as file:
            json.dump(decomposed_scores_str_keys, file, indent=4)
                

def process_test_decomposed_prompts_only_qrel(test_qrel, docid_to_doc, qid_to_query, result_path:str, pipeline ,system_message:str):
    generation_path = result_path.replace("results","generation_errors")
    logs_path = result_path.replace("results","logs").replace(".txt",".json") 
    decomposed_path = result_path.replace("results","decomposed_scores").replace(".txt",".json") 
    cuda_errors_path = result_path.replace("results","cuda_errors")
    decomposed_scores = {}
    
    with open(result_path, 'w') as result_file, open(generation_path, 'w') as generation_errors_file, open(cuda_errors_path,"w") as cuda_errors_file:
        for eachline in tqdm(test_qrel.itertuples(index=True)):
            qidx = eachline.qid
            docidx = eachline.docid

            try:
                # Get relevance score
                # print(type(list(docid_to_doc.keys())[0]))
                # print(type(docidx))
                # print(docidx in list(docid_to_doc.keys()))
                # print(qid_to_query[qidx])
                # print(docid_to_doc[docidx])
                # print(qidx)
                # print(docidx)
                
                try:
                    pred_score, decomposed_scores_list_for_one_query = get_relevance_score_decomposed_prompts(query=qid_to_query[qidx], passage=docid_to_doc[docidx],pipeline=pipeline,log_file_path=logs_path,system_message=system_message,qidx=qidx,docidx=docidx)
                except:
                    docidx = str(docidx)
                    pred_score, decomposed_scores_list_for_one_query = get_relevance_score_decomposed_prompts(query=qid_to_query[qidx], passage=docid_to_doc[docidx],pipeline=pipeline,log_file_path=logs_path,system_message=system_message,qidx=qidx,docidx=docidx)
                    
                    
                decomposed_scores[(qidx,docidx)] = decomposed_scores_list_for_one_query
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    error = f"CUDA out of memory error for docid {docidx}. Skipping this document."
                    print(error)
                    cuda_errors_file.write(error)
                    
                    result_file.write(f"{qidx} 0 {docidx} 0\n")
                    
                    continue
                else:
                    raise e
            
            # Debugging: Print prompt and score once
            if not hasattr(process_test_qrel_baseline_only_qrel, "called"):
                process_test_qrel_baseline_only_qrel.called = True
                print(f"{qidx} 0 {docidx} {pred_score}\n")
            try:
                int_pred_score = int(pred_score)
                if 0<=int_pred_score<=3:
                    # Write result to file
                    result_file.write(f"{qidx} 0 {docidx} {pred_score}\n")
                else:
                    result_file.write(f"{qidx} 0 {docidx} 0\n")
                
            except:
                generation_errors_file.write(f"{qidx} 0 {docidx} {pred_score}\n")
                
                print(f"pred score for {qidx} and {docidx} is not a number it is: {pred_score}")
                result_file.write(f"{qidx} 0 {docidx} 0\n")
                
                continue
        # Convert tuples to strings
        decomposed_scores_str_keys = {str(k): v for k, v in decomposed_scores.items()}

        # Save to a JSON file
        with open(decomposed_path, 'w') as file:
            json.dump(decomposed_scores_str_keys, file, indent=4)



def process_test_sunprompt_then_decomposed_only_qrel(test_qrel, docid_to_doc, qid_to_query, result_path:str, pipeline ,system_message:str):
    generation_path = result_path.replace("results","generation_errors")
    logs_path = result_path.replace("results","logs").replace(".txt",".json") 
    decomposed_path = result_path.replace("results","decomposed_scores").replace(".txt",".json") 
    cuda_errors_path = result_path.replace("results","cuda_errors")
    decomposed_scores = {}
    
    with open(result_path, 'w') as result_file, open(generation_path, 'w') as generation_errors_file, open(cuda_errors_path,"w") as cuda_errors_file:
        for eachline in tqdm(test_qrel.itertuples(index=True)):
            qidx = eachline.qid
            docidx = eachline.docid

            try:
                # Get relevance score
                try:
                    pred_score, decomposed_scores_list_for_one_query = sun_prompt_then_decomposed(query=qid_to_query[qidx], passage=docid_to_doc[docidx],pipeline=pipeline,log_file_path=logs_path,system_message=system_message,qidx=qidx,docidx=docidx)
                except:
                    docidx = str(docidx)
                    pred_score, decomposed_scores_list_for_one_query = sun_prompt_then_decomposed(query=qid_to_query[qidx], passage=docid_to_doc[docidx],pipeline=pipeline,log_file_path=logs_path,system_message=system_message,qidx=qidx,docidx=docidx)
                  
                
                # pred_score, decomposed_scores_list_for_one_query = sun_prompt_then_decomposed(query=qid_to_query[qidx], passage=docid_to_doc[docidx],pipeline=pipeline,log_file_path=logs_path,system_message=system_message,qidx=qidx,docidx=docidx)
                decomposed_scores[(qidx,docidx)] = decomposed_scores_list_for_one_query
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    error = f"CUDA out of memory error for docid {docidx}. Skipping this document."
                    print(error)
                    cuda_errors_file.write(error)
                    
                    result_file.write(f"{qidx} 0 {docidx} 0\n")
                    
                    continue
                else:
                    raise e
            
            # Debugging: Print prompt and score once
            if not hasattr(process_test_sunprompt_then_decomposed_only_qrel, "called"):
                process_test_sunprompt_then_decomposed_only_qrel.called = True
                print(f"{qidx} 0 {docidx} {pred_score}\n")
            try:
                int_pred_score = int(pred_score)
                if 0<=int_pred_score<=3:
                    # Write result to file
                    result_file.write(f"{qidx} 0 {docidx} {pred_score}\n")
                else:
                    result_file.write(f"{qidx} 0 {docidx} 0\n")
                
            except:
                generation_errors_file.write(f"{qidx} 0 {docidx} {pred_score}\n")
                
                print(f"pred score for {qidx} and {docidx} is not a number it is: {pred_score}")
                result_file.write(f"{qidx} 0 {docidx} 0\n")
                
                continue
        # Convert tuples to strings
        decomposed_scores_str_keys = {str(k): v for k, v in decomposed_scores.items()}

        # Save to a JSON file
        with open(decomposed_path, 'w') as file:
            json.dump(decomposed_scores_str_keys, file, indent=4)