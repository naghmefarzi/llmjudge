import numpy as np
import pandas as pd
import jsonlines
from tqdm import tqdm
import os
import gc
import logging
import transformers
import torch
from pathlib import Path
from typing import Tuple, List, Dict, Callable, NewType, Optional, Iterable




if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("device: {d}".format(d = device))
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

hf_api_token = "yourhftoken"

# Authenticate using the token
from huggingface_hub import login
login(token=hf_api_token)

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

def get_model(name_or_path_to_model : str):
    

    pipeline = transformers.pipeline(
        "text-generation",
        model=name_or_path_to_model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    return pipeline



def get_prompt(query, passage):
    return f"""Please rate how the given passage is relevant to the query. The output must be only a score that indicate how relevant they are.

    Query: {query}
    Passage: {passage}

    Score:"""



def get_relevance_score(prompt:str, pipeline):
    system_message = """You are a search quality rater evaluating the relevance of passages. Given a query and passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:

    3 = Perfectly relevant: The passage is dedicated to the query and contains the exact answer.
    2 = Highly relevant: The passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.
    1 = Related: The passage seems related to the query but does not answer it.
    0 = Irrelevant: The passage has nothing to do with the query

    Assume that you are writing an answer to the query. If the passage seems to be related to the query but does not include any answer to the query, mark it 1. If you would use any of the information contained in the passage in such an asnwer, mark it 2. If the passage is primarily about the query, or contains vital information about the topic, mark it 3. Otherwise, mark it 0."""

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

    with torch.no_grad():  # Disable gradient calculation
        outputs = pipeline(
            prompt,
            max_new_tokens=200,
            eos_token_id=terminators,
            pad_token_id=128009,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

    return outputs[0]["generated_text"][len(prompt):]
    
def process_test_qrel(test_qrel, docid_to_doc, qid_to_query, pipeline, result_file):
    for eachline in tqdm(test_qrel.itertuples(index=True)):
        qidx = eachline.qid
        docidx = eachline.docid
        if docidx in docid_to_doc:
            print(docidx)
            prompt = get_prompt(query=qid_to_query[qidx], passage=docid_to_doc[docidx])
            print("prompt is ready")
            pred_score = get_relevance_score(prompt, pipeline)
            line = f"{qidx} 0 {docidx} {pred_score}\n"
            print(line)
            result_file.write(line)
            torch.cuda.empty_cache()
            gc.collect()


def main():


    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    test_qrel_path = "./data/llm4eval_test_qrel_2024.txt"
    result_file_path = 'llm4eval_test_qrel_results_baseline.txt'
    query_path = "./data/llm4eval_query_2024.txt"
    docs_path = './data/llm4eval_document_2024.jsonl'
    chunk_size = 100
    
    
    # docid_to_doc = get_all_docid_to_doc(docs_path)
    qid_to_query = get_all_query_id_to_query(query_path)
    pipeline = get_model(model_id)

    if os.path.exists(result_file_path):
        os.remove(result_file_path)  # Ensure the result file is empty before starting

    test_qrel = pd.read_csv(test_qrel_path, sep=" ", header=None, names=['qid', 'Q0', 'docid'])
    

    with open(result_file_path, 'a') as result_file:
        for docid_to_doc in load_documents_in_chunks(docs_path, chunk_size):
            print(len(docid_to_doc))
            
            process_test_qrel(test_qrel, docid_to_doc, qid_to_query, pipeline, result_file)
            # Clear cache and force garbage collection after each chunk
            torch.cuda.empty_cache()
            gc.collect()

if __name__=="__main__":
    main()
    