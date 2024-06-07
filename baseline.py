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
from pathlib import Path
from typing import Tuple, List, Dict, Callable, NewType, Optional, Iterable
from huggingface_hub import login
from transformers import (AutoTokenizer,AutoModelForCausalLM,
                          TextStreamer,pipeline,BitsAndBytesConfig)
import sys
sys.path.append('.')
from data_processing import load_data_files, clean_files
from prompts import create_system_message


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print("device: {d}".format(d = device))


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


## Constants
MAX_LENGTH = 8000
system_message = """You are a search quality rater evaluating the relevance of passages. Given a query and passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:

3 = Perfectly relevant: The passage is dedicated to the query and contains the exact answer.
2 = Highly relevant: The passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.
1 = Related: The passage seems related to the query but does not answer it.
0 = Irrelevant: The passage has nothing to do with the query

Assume that you are writing an answer to the query. If the passage seems to be related to the query but does not include any answer to the query, mark it 1. If you would use any of the information contained in the passage in such an asnwer, mark it 2. If the passage is primarily about the query, or contains vital information about the topic, mark it 3. Otherwise, mark it 0."""




def get_prompt(query, passage,pipeline):

    prompt = f"""Please rate how the given passage is relevant to the query. The output must be only a score that indicate how relevant they are.

    Query: {query}
    Passage: {passage}

    Score:"""
    return prompt
    # return truncate_prompt_based_on_passage(prompt, pipeline, MAX_LENGTH)

   


def truncate_prompt_based_on_passage(prompt:str,pipeline, max_length: int) -> str:
    # Truncate passage part of the prompt
    """Truncate passage in the prompt if it exceeds the maximum token length."""
    tokens = pipeline.tokenizer.tokenize(prompt)
    if len(tokens) <= max_length:
        return prompt

    passage_start_index = prompt.find("Passage:") + len("Passage:")
    passage_end_index = prompt.find("Score:")
    truncated_passage = prompt[passage_start_index:passage_end_index]

    passage_tokens = pipeline.tokenizer.tokenize(truncated_passage)
    prompt_tokens = pipeline.tokenizer.tokenize(prompt[:passage_start_index]) + pipeline.tokenizer.tokenize(prompt[passage_end_index:])
    available_length = max_length - len(prompt_tokens)

    truncated_passage_tokens = passage_tokens[:available_length]
    print("here")
    print(truncated_passage_tokens)
    truncated_passage = pipeline.tokenizer.decode(truncated_passage_tokens[1])
    print(f"{prompt[:passage_start_index]} {truncated_passage} {prompt[passage_end_index:]}")
    return f"{prompt[:passage_start_index]} {truncated_passage} {prompt[passage_end_index:]}"





def get_model_baseline(name_or_path_to_model : str):
    

    pipeline = transformers.pipeline(
        "text-generation",
        model=name_or_path_to_model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    return pipeline

def get_relevance_score_baseline(prompt: str,pipeline,system_message:str):
  messages = [
      {"role": "system", "content": system_message},
      {"role": "user", "content": prompt},
  ]
  
  
  # Check if the function has been called before
  if not hasattr(get_relevance_score_baseline, "called"):
  # Set the attribute to indicate the function has been called
    get_relevance_score_baseline.called = True
    print(messages)

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


def process_test_qrel_baseline(test_qrel, docid_to_doc, qid_to_query, result_path, pipeline,chunk_size ,generative_error_file_path: Optional[str],problematic_passages_path: Optional[str],system_message:str):
    # Open file to write results
    with open(result_path, 'w') as result_file:
        for start_idx in tqdm(range(0, len(test_qrel), chunk_size)):
            # print(start_idx)
            batch = test_qrel.iloc[start_idx:start_idx + chunk_size]
            
            process_batch_baseline(batch, qid_to_query, docid_to_doc,  result_file, pipeline,generative_error_file_path,problematic_passages_path, system_message)
                
            torch.cuda.empty_cache()  # Clear GPU cache if using GPU  
            del batch



def process_batch_baseline(batch, qid_to_query, docid_to_doc,  result_file, pipeline,generative_error_file_path:Optional[str], problematic_passages_path: Optional[str], system_message: str):


    for eachline in batch.itertuples(index=True):
        try:
            qidx = eachline.qid
            docidx = eachline.docid
            prompt = get_prompt(query=qid_to_query[qidx], passage=docid_to_doc[docidx],pipeline=pipeline)
            # print(prompt)
            response_text = get_relevance_score_baseline(prompt,pipeline,system_message)
            try:
                response_text_int = int(response_text)
                if response_text_int in [0,1,2,3,"0","1","2","3"]:
                    result_file.write(f"{qidx} 0 {docidx} {response_text}\n")
                else:
                    if generative_error_file_path:
                        with open(generative_error_file_path,"a") as errors_file:
                            errors_file.write(f"{qidx} 0 {docidx} {response_text}\n")
                    
            except:
                if generative_error_file_path:
                    with open(generative_error_file_path,"a") as errors_file:
                        errors_file.write(f"{qidx} 0 {docidx} {response_text}\n")
        
        except Exception as e:
            if problematic_passages_path:
                    with open(problematic_passages_path,"a") as p:
                        p.write(f"Error processing QID {qidx}, DOCID {docidx}: {e}\n")
                        # print(f"Error processing QID {qidx}, DOCID {docidx}: {e}")
        
        torch.cuda.empty_cache()  # Clear GPU cache if using GPU


 
def get_model_quantized(name_or_path_to_model: str) -> Tuple:


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


def process_batch_quantized(batch, qid_to_query, docid_to_doc,  result_file, model, tokenizer,problematic_passages_path: Optional[str]):

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
            if problematic_passages_path:
                with open(problematic_passages_path,"a") as f:
                    f.write(err)
        
        torch.cuda.empty_cache()  # Clear GPU cache if using GPU

  
    
def process_test_qrel_quantized(test_qrel, docid_to_doc, qid_to_query, result_path, model, tokenizer,chunk_size = 100):
    # Open file to write results
    with open(result_path, 'w') as result_file:
        for start_idx in tqdm(range(0, len(test_qrel), chunk_size)):
            # print(start_idx)
            batch = test_qrel.iloc[start_idx:start_idx + chunk_size]
            process_batch_quantized(batch, qid_to_query, docid_to_doc,  result_file, model, tokenizer)
            torch.cuda.empty_cache()  # Clear GPU cache if using GPU  
            del batch
            



def main():
    parser = argparse.ArgumentParser(description="Process QREL data with a specified model.")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID or path to the model.")
    parser.add_argument("--test_qrel_path", type=str, required=True, help="Path to the test QREL file.")
    parser.add_argument("--result_file_path", type=str, required=True, help="Path to the result file.")
    parser.add_argument("--queries_path", type=str, required=True, help="Path to the queries file.")
    parser.add_argument("--docs_path", type=str, required=True, help="Path to the documents file.")
    parser.add_argument("--chunk_size", type=int, default=100, help="Size of chunks to process at a time.")
    parser.add_argument("--quantized", action="store_true", help="Use quantized model.")
    parser.add_argument("--problematic_passages_path", type=str, help="Path to the file for problematic passages (CUDA memory problem).")
    parser.add_argument("--generative_error_file_path", type=str, help="Path to the file for problematic passages (CUDA memory problem).")
    parser.add_argument("--score_order_in_prompt", type=str,default="3210", help="order of scores in prompt.")
    
    
    args = parser.parse_args()

    docid_to_doc, qid_to_query, test_qrel = load_data_files(args.docs_path, args.queries_path, args.test_qrel_path)
    clean_files(args.result_file_path, args.problematic_passages_path, args.generative_error_file_path)
     
        
        
    system_message = create_system_message(args.score_order_in_prompt)
    if not args.quantized:
        pipe = get_model_baseline(args.model_id)
        result_path =  args.result_file_path.replace(".txt",f"_prompt order: {args.score_order_in_prompt}.txt")
        process_test_qrel_baseline(test_qrel, docid_to_doc, qid_to_query, result_path, pipe, args.chunk_size, args.generative_error_file_path, args.problematic_passages_path, system_message)
    else:
        model, tokenizer = get_model_quantized(args.model_id)
        process_test_qrel_quantized(test_qrel, docid_to_doc, qid_to_query, args.result_file_path, model, tokenizer,args.problematic_passages_path)









if __name__=="__main__":
    main()
    