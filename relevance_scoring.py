from collections import defaultdict
from heapq import nlargest
from prompts import get_prompt
from typing import Tuple, List, Dict, Callable, NewType, Optional, Iterable
import torch
from prompts import create_system_message
import re
import json

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



def find_first_number(text):
    # Regular expression to find standalone numbers
    match = re.search(r'\b\d+\b', text)
    if match:
        if 0<=int(match.group())<=3:
            return int(match.group())
    else:
        return None

def get_relevance_score_iterative_prompts(query,passage,pipeline,log_file_path,system_message,qidx,docidx):
    decomposed_scores_dict = {}
    # Define the initial prompt
    system_message_decomposition = """You are evaluating the relevance of a passage to a query. Please provide a score on an integer scale of 0 to 3 for each dimension of relevance.

3 = Excellent: The passage fully meets the criteria.
2 = Good: The passage partially meets the criteria.
1 = Fair: The passage has minor relevance but lacks in certain aspects.
0 = Poor: The passage does not meet the criteria at all.

Proceed with the evaluation.\n"""

    # Define the hierarchical order of prompts
    decomposed_criterias = {
        "Exactness":" How precisely does the passage answer the query",
        "Topicality": "Is the passage about the same subject as the query",
        "Depth": "How much detail does the passage provide about the topic",
        "Contextual Fit": "Does the passage provide relevant background or context",
    }
    prompt = f'''

        Query: {query}
        Passage: {passage}\n'''
    
    for i in range(len(decomposed_criterias)):
        
        criteria_prompt = f"Please rate how the given passage in case of {list(decomposed_criterias.keys())[i]} to the query. The output must be only a score (0-3) that indicate {list(decomposed_criterias.values())[i]}."         
        
        to_ask_prompt = criteria_prompt + prompt+'''\nScore:'''
            
        score = get_relevance_score_baseline(to_ask_prompt,pipeline,system_message_decomposition)
        num_score = find_first_number(score)
        prompt+=f"\n{list(decomposed_criterias.keys())[i]}: {num_score}"
        decomposed_scores_dict[list(decomposed_criterias.keys())[i]]=num_score
        if not hasattr(get_relevance_score_iterative_prompts,"called"):
            # get_relevance_score_iterative_prompts.called =True
            print(system_message_decomposition)
            print(to_ask_prompt)
            print(score)
    inf = {"qidx":qidx,
                   "docidx":docidx,
                   "query": query,
                   "passage": passage,
                   "decomposed_scores_dict":decomposed_scores_dict,
                   "prompt":prompt,
                   
            
        }
    
            
    criteria_prompt = '''Please rate how the given passage is relevant to the query. The output must be only a score that indicate how relevant they are.\n'''
    to_ask_prompt = criteria_prompt + prompt + '''\nScore:'''
    score = get_relevance_score_baseline(prompt,pipeline,system_message)
    num_score = find_first_number(score)
    if not hasattr(get_relevance_score_iterative_prompts,"called"):
            get_relevance_score_iterative_prompts.called =True
            print("******")
            print(system_message)
            print(to_ask_prompt)
            print(score)
            print("*"*20)
            print(num_score)
    inf['relevance_prompt']=to_ask_prompt
    inf['llm_response'] = score
    inf['final_relevance_score'] = num_score
    with open(log_file_path,"a") as f:
        json.dump(inf, f)
    return num_score , decomposed_scores_dict
        


 
 
 
def get_relevance_score_decomposed_prompts(query,passage,pipeline,log_file_path,system_message,qidx,docidx):
    decomposed_scores_dict = {}
    # Define the initial prompt
    system_message_decomposition = """You are evaluating the relevance of a passage to a query. Please provide a score on an integer scale of 0 to 3 for each dimension of relevance.

3 = Excellent: The passage fully meets the criteria.
2 = Good: The passage partially meets the criteria.
1 = Fair: The passage has minor relevance but lacks in certain aspects.
0 = Poor: The passage does not meet the criteria at all.

Proceed with the evaluation.\n"""

    # Define the hierarchical order of prompts
    decomposed_criterias = {
        "Exactness":" How precisely does the passage answer the query",
        "Topicality": "Is the passage about the same subject as the query",
        "Depth": "How much detail does the passage provide about the topic",
        "Contextual Fit": "Does the passage provide relevant background or context",
    }
    prompt = f'''
        Query: {query}
        Passage: {passage}\n'''
    prompt_ = prompt
    for i in range(len(decomposed_criterias)):
        
        criteria_prompt = f"Please rate how the given passage in case of {list(decomposed_criterias.keys())[i]} to the query. The output must be only a score (0-3) that indicate {list(decomposed_criterias.values())[i]}."         
        
        to_ask_prompt = criteria_prompt + prompt+'''\nScore:'''
            
        score = get_relevance_score_baseline(to_ask_prompt,pipeline,system_message_decomposition)
        num_score = find_first_number(score)
        prompt_+=f"\n{list(decomposed_criterias.keys())[i]}: {num_score}"
        decomposed_scores_dict[list(decomposed_criterias.keys())[i]]=num_score
        if not hasattr(get_relevance_score_iterative_prompts,"called"):
            # get_relevance_score_iterative_prompts.called =True
            print(system_message_decomposition)
            print(to_ask_prompt)
            print(score)
    inf = {"qidx":qidx,
                   "docidx":docidx,
                   "query": query,
                   "passage": passage,
                   "decomposed_scores_dict":decomposed_scores_dict,
                   "prompt":prompt_,
                   
            
        }
    
            
    criteria_prompt = '''Please rate how the given passage is relevant to the query. The output must be only a score that indicate how relevant they are.\n'''
    to_ask_prompt = criteria_prompt + prompt_ + '''\nScore:'''
    score = get_relevance_score_baseline(prompt,pipeline,system_message)
    num_score = find_first_number(score)
    if not hasattr(get_relevance_score_iterative_prompts,"called"):
            get_relevance_score_iterative_prompts.called =True
            print("******")
            print(system_message)
            print(to_ask_prompt)
            print(score)
            print("*"*20)
            print(num_score)
    inf['relevance_prompt']=to_ask_prompt
    inf['llm_response'] = score
    inf['final_relevance_score'] = num_score
    with open(log_file_path,"a") as f:
        json.dump(inf, f)
    return num_score , decomposed_scores_dict
        


 
 
 

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




def write_top_k_results(relevance_scores_agg, result_file_full , result_file_top_k, k: Optional[int]):
    for qidx, doc_scores in relevance_scores_agg.items():
        top_k_docs = nlargest(k, doc_scores.items(), key=lambda x: x[1])
        all_top = nlargest(len(doc_scores.items()), doc_scores.items(), key=lambda x: x[1])
            
        for docidx, relevance_score in top_k_docs:
            result_file_top_k.write(f"{qidx} 0 {docidx} {relevance_score}\n")
            
        for docidx, relevance_score in all_top:
            result_file_full.write(f"{qidx} 0 {docidx} {relevance_score}\n")


def process_documents(documents, test_qrel, qid_to_query, pipeline, k, generative_error_file_path, problematic_passages_path, system_message):
    # print(len(test_qrel['qid'].unique()))
    # print(len(documents))
    relevance_scores_agg = defaultdict(dict)
    # print(f"len documents {len(documents)}")
    # print(documents)
    for docidx, doc_content in documents.items():
        
        relevance_scores = {}
        # print(f"len queries {len(test_qrel['qid'].unique())}")
        for qidx in test_qrel['qid'].unique():
            
            try:
                prompt = get_prompt(query=qid_to_query[qidx], passage=doc_content, pipeline=pipeline)
                response_text = get_relevance_score_baseline(prompt, pipeline, system_message)
                try:
                    response_text_int = int(response_text)
                    relevance_scores[qidx] = response_text_int
                except:
                    if generative_error_file_path: #GENERATION ERROR-RELEVANCE SCORE IS NOT A NUMBER
                        with open(generative_error_file_path, "a") as errors_file:
                            errors_file.write(f"{qidx} 0 {docidx} {response_text}\n")
            except Exception as e: #CUDA MEMORY
                if problematic_passages_path:
                    with open(problematic_passages_path, "a") as p:
                        p.write(f"Error processing QID {qidx}, DOCID {docidx}: {e}\n")
        
        for qidx, relevance_score in relevance_scores.items():
            relevance_scores_agg[qidx][docidx] = relevance_score

    return relevance_scores_agg
        # yield docidx, relevance_scores

# def process_documents_exam(documents, test_qrel, qid_to_query, pipeline, k, generative_error_file_path, problematic_passages_path, system_message):
    
#     relevance_scores_agg = defaultdict(dict)
#     for docidx, doc_content in documents.items():
#         relevance_scores = {}
#         for qidx in test_qrel['qid'].unique():
            