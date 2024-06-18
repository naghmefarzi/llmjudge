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
from relevance_scoring import get_relevance_score_baseline, write_top_k_results, process_documents
from data_processing import load_data_files, clean_files, process_documents_in_chunks
from prompts import create_system_message
from exam_question_generation import generate_question_set



def process_exam_qrel(test_qrel, docid_to_doc, qid_to_query, result_path, pipeline,chunk_size ,generative_error_file_path: Optional[str],problematic_passages_path: Optional[str],system_message:str, k: Optional[int]):
    with open(result_path, 'w') as result_file:
        relevance_scores_agg = defaultdict(dict)
        for documents in process_documents_in_chunks(docid_to_doc, chunk_size):
            
            processed_documents = process_documents(documents, test_qrel, qid_to_query, pipeline, k, generative_error_file_path, problematic_passages_path, system_message)
            for qidx, doc_scores in processed_documents.items():
                relevance_scores_agg[qidx].update(doc_scores)
        write_top_k_results(relevance_scores_agg, result_file, k)
    torch.cuda.empty_cache()  # Clear GPU cache if using GPU


