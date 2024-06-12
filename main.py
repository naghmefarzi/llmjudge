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

from typing import Tuple, List, Dict, Callable, NewType, Optional, Iterable
from huggingface_hub import login
from transformers import (AutoTokenizer,AutoModelForCausalLM,
                          TextStreamer,pipeline,BitsAndBytesConfig)
import sys
sys.path.append('.')

from model_utils import *
from prompts import create_system_message
from exam_question_generation import generate_question_set
from data_processing import load_data_files, clean_files
from baseline import process_test_qrel_baseline
from examline import process_exam_qrel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print("device: {d}".format(d = device))


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)




def main():
    parser = argparse.ArgumentParser(description="Process QREL data with a specified model.")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID or path to the model.")
    parser.add_argument("--test_qrel_path", type=str, required=True, help="Path to the test QREL file.")
    parser.add_argument("--result_file_path", type=str, required=True, help="Path to the result file.")
    parser.add_argument("--queries_path", type=str, required=True, help="Path to the queries file.")
    parser.add_argument("--docs_path", type=str, required=True, help="Path to the documents file.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Size of chunks to process at a time.")
    parser.add_argument("--exam", action="store_true", help="Use exam model.")
    parser.add_argument("--problematic_passages_path", type=str, help="Path to the file for problematic passages (CUDA memory problem).")
    parser.add_argument("--generative_error_file_path", type=str, help="Path to the file for problematic passages (CUDA memory problem).")
    parser.add_argument("--score_order_in_prompt", type=str, default="3210", help="order of scores in baseline prompt.")
    parser.add_argument("--store_top_k_doc_scores", type=int, default=20, help="write top k documents in qrel file")
    
    
    args = parser.parse_args()

    docid_to_doc, qid_to_query, test_qrel = load_data_files(args.docs_path, args.queries_path, args.test_qrel_path)
    clean_files(args.result_file_path, args.problematic_passages_path, args.generative_error_file_path)
     
        
        
    system_message = create_system_message(args.score_order_in_prompt)
    if not args.exam:
        pipe = get_model_baseline(args.model_id)
        # result_path =  args.result_file_path.replace(".txt",f"_prompt order: {args.score_order_in_prompt}.txt")
        result_path =  args.result_file_path.replace(".run",f"_prompt order: {args.score_order_in_prompt}.run")
        
        generative_error_file_path = args.generative_error_file_path.replace(".txt",f"_prompt order: {args.score_order_in_prompt}.txt")
        process_test_qrel_baseline(test_qrel, docid_to_doc, qid_to_query, result_path, pipe, args.chunk_size, generative_error_file_path, args.problematic_passages_path, system_message, args.store_top_k_doc_scores)
    else:
        pipe = get_model_baseline(args.model_id)
        result_path =  args.result_file_path.replace(".txt",f"_prompt order: {args.score_order_in_prompt}.txt")
        result_path =  args.result_file_path.replace(".run",f"_prompt order: {args.score_order_in_prompt}.run")
        generative_error_file_path = args.generative_error_file_path.replace(".txt",f"_prompt order: {args.score_order_in_prompt}.txt")
        question_set_file_path = './exam_question_set/' + args.test_qrel_path.replace(".txt","_exam.jsonl")
        if not os.path.exists(question_set_file_path):
            generate_question_set(args.test_qrel_path, test_qrel, qid_to_query, pipe)
            
            
            
        process_exam_qrel(test_qrel, docid_to_doc, qid_to_query, result_path, pipe, args.chunk_size, generative_error_file_path, args.problematic_passages_path, system_message, args.store_top_k_doc_scores)
        







if __name__=="__main__":
    main()
    