import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import gc
import jsonlines
import logging
import transformers
import torch
from pathlib import Path
from typing import Tuple, List, Dict, Callable, NewType, Optional, Iterable
from huggingface_hub import login
from transformers import (AutoTokenizer,AutoModelForCausalLM,
                          TextStreamer,pipeline,BitsAndBytesConfig)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("device: {d}".format(d = device))
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
MAX_LENGTH = 8000
problamatic_passages_path = "./passages_length_exceed.txt"


# Authenticate using the token
from huggingface_hub import login

def get_all_docid_to_doc(docs_path: str = './data/llm4eval_document_2024.jsonl'):
    docid_to_doc = dict()
    with jsonlines.open(docs_path, 'r') as document_file:
        for obj in document_file:
            docid_to_doc[obj['docid']] = obj['doc']
    return docid_to_doc

def get_all_query_id_to_query(query_path:str):
    query_data = pd.read_csv(query_path, sep="\t", header=None, names=['qid', 'qtext'])
    qid_to_query = dict(zip(query_data.qid, query_data.qtext))
    return qid_to_query

def load_documents_in_chunks(docs_path, chunk_size=100):
    docid_to_doc = {}
    with jsonlines.open(docs_path, 'r') as document_file:
        for obj in document_file:
            docid = obj['docid']
            docid_to_doc[docid] = obj['doc']
            if len(docid_to_doc) >= chunk_size:
                yield docid_to_doc
                docid_to_doc = {}
        if docid_to_doc:  # Yield the remaining documents if any
            yield docid_to_doc



def get_model_q(name_or_path_to_model: str):


    tokenizer = AutoTokenizer.from_pretrained(name_or_path_to_model)
    bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path_to_model,
        #torch_dtype=torch.bfloat16,
        quantization_config = bnb_config,
        device_map="auto",
    )
    return model,tokenizer




def truncate_prompt_based_on_passage(prompt:str,pipeline):
    # Truncate passage part of the prompt
    prompt_tokens = pipeline.tokenizer.tokenize(prompt)
    passage_start_index = prompt.find("Passage:")
    passage_end_index = prompt.find("Score:")
    passage = prompt[passage_start_index:passage_end_index]
    
    passage_tokens = pipeline.tokenizer.tokenize(passage)
    limit = MAX_LENGTH-(prompt_tokens-passage_tokens)
    passage_tokens = passage_tokens[:limit]
        
    truncated_passage = pipeline.tokenizer.decode(passage_tokens)
    print(f"truncated_passage: {truncated_passage}")
    return truncated_passage




def get_model_baseline(name_or_path_to_model : str):
    

    pipeline = transformers.pipeline(
        "text-generation",
        model=name_or_path_to_model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    return pipeline

def get_relevance_score_baseline(prompt,pipeline):
  messages = [
      {"role": "system", "content": system_message},
      {"role": "user", "content": prompt},
  ]

  prompt = pipeline.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
  )

  terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = pipeline(
      prompt,
      max_new_tokens=100,
      eos_token_id=terminators,
      pad_token_id=128009,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )

  return outputs[0]["generated_text"][len(prompt):]

def get_prompt(query, passage,pipeline):

    prompt = f"""Please rate how the given passage is relevant to the query. The output must be only a score that indicate how relevant they are.

    Query: {query}
    Passage: {passage}

    Score:"""
    # Tokenize prompt using Llama tokenizer
    # tokens = pipeline.tokenizer.tokenize(prompt)
    
    # # Check if prompt exceeds maximum sequence length token
    # if len(tokens) > MAX_LENGTH:
    #     passage = truncate_prompt_based_on_passage(prompt,pipeline,MAX_LENGTH)
    #     prompt = f"""Please rate how the given passage is relevant to the query. The output must be only a score that indicate how relevant they are.

    #     Query: {query}
    #     Passage: {passage}

    #     Score:"""
    return prompt

    
# Define chunk size
system_message = """You are a search quality rater evaluating the relevance of passages. Given a query and passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:

3 = Perfectly relevant: The passage is dedicated to the query and contains the exact answer.
2 = Highly relevant: The passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.
1 = Related: The passage seems related to the query but does not answer it.
0 = Irrelevant: The passage has nothing to do with the query

Assume that you are writing an answer to the query. If the passage seems to be related to the query but does not include any answer to the query, mark it 1. If you would use any of the information contained in the passage in such an asnwer, mark it 2. If the passage is primarily about the query, or contains vital information about the topic, mark it 3. Otherwise, mark it 0."""

def process_batch_q(batch, qid_to_query, docid_to_doc,  result_file, model, tokenizer):


    for eachline in batch.itertuples(index=True):
        try:
            qidx = eachline.qid
            docidx = eachline.docid
            prompt = get_prompt(query=qid_to_query[qidx], passage=docid_to_doc[docidx],pipeline=model)
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
            
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("")
            ]
            
            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators[0],
                pad_token_id=128009,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            
            response = outputs[0][input_ids.shape[-1]:]
            response_text = tokenizer.decode(response, skip_special_tokens=True)
            
            result_file.write(f"{qidx} 0 {docidx} {response_text}\n")
        
        except Exception as e:
            err = f"Error processing QID {qidx}, DOCID {docidx}: {e}\n"
            # print(err)
            with open(problamatic_passages_path,"a") as f:
                f.write(err)
        
        torch.cuda.empty_cache()  # Clear GPU cache if using GPU

  
    
def process_test_qrel_q(test_qrel, docid_to_doc, qid_to_query, result_path, model, tokenizer,chunk_size = 100):
    # Open file to write results
    with open(result_path, 'w') as result_file:
        for start_idx in tqdm(range(0, len(test_qrel), chunk_size)):
            # print(start_idx)
            batch = test_qrel.iloc[start_idx:start_idx + chunk_size]
            process_batch_q(batch, qid_to_query, docid_to_doc,  result_file, model, tokenizer)
            torch.cuda.empty_cache()  # Clear GPU cache if using GPU  
            del batch
            
def process_test_qrel_baseline(test_qrel, docid_to_doc, qid_to_query, result_path, pipeline,chunk_size = 100,errors_file_path="./errors.txt",):
    # Open file to write results
    with open(result_path, 'w') as result_file, open(errors_file_path,'w') as errors_file:
        for start_idx in tqdm(range(0, len(test_qrel), chunk_size)):
            # print(start_idx)
            batch = test_qrel.iloc[start_idx:start_idx + chunk_size]
            process_batch_baseline(batch, qid_to_query, docid_to_doc,  result_file, pipeline,errors_file)
            torch.cuda.empty_cache()  # Clear GPU cache if using GPU  
            del batch



def process_batch_baseline(batch, qid_to_query, docid_to_doc,  result_file, pipeline,errors_file):


    for eachline in batch.itertuples(index=True):
        try:
            qidx = eachline.qid
            docidx = eachline.docid
            prompt = get_prompt(query=qid_to_query[qidx], passage=docid_to_doc[docidx],pipeline=pipeline)
            response_text = get_relevance_score_baseline(prompt,pipeline)
            try:
                response_text_int = int(response_text)
                result_file.write(f"{qidx} 0 {docidx} {response_text}\n")
            except:
                errors_file.write(f"{qidx} 0 {docidx} {response_text}\n")
        
        except Exception as e:
            print(f"Error processing QID {qidx}, DOCID {docidx}: {e}")
        
        torch.cuda.empty_cache()  # Clear GPU cache if using GPU

  

def main():

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    test_qrel_path = "./data/llm4eval_test_qrel_2024.txt"
    result_file_path = 'llm4eval_test_qrel_results_baseline.txt'
    query_path = "./data/llm4eval_query_2024.txt"
    docs_path = './data/llm4eval_document_2024.jsonl'
    chunk_size = 10
    
    
    docid_to_doc = get_all_docid_to_doc(docs_path)
    qid_to_query = get_all_query_id_to_query(query_path)
    pipeline = get_model_baseline(model_id)

    if os.path.exists(result_file_path):
        os.remove(result_file_path)  # Ensure the result file is empty before starting

    test_qrel = pd.read_csv(test_qrel_path, sep=" ", header=None, names=['qid', 'Q0', 'docid'])
    # process_test_qrel_q(test_qrel, docid_to_doc, qid_to_query, result_file_path, model, tokenizer)
    process_test_qrel_baseline(test_qrel, docid_to_doc, qid_to_query, result_file_path, pipeline,chunk_size)
    



if __name__=="__main__":
    main()
    