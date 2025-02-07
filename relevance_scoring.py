from collections import defaultdict
from heapq import nlargest
from prompts import get_prompt
from typing import Tuple, List, Dict, Callable, NewType, Optional, Iterable
import torch
from prompts import create_system_message,create_system_message_for_non_rel,create_system_message_for_rel
import re
import json
from together_model import TogetherPipeline
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
        # "Depth": "How much detail does the passage provide about the topic",
        "Coverage": "how much of the passage is dedicated to discussing the query and its related topics.",
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
    
            
    criteria_prompt = "Please rate how the given passage is relevant to the query by considering the provided grades for exactness, topicality, coverage, and contextual fit. The output should be only a score indicating the relevance.\n" 
    to_ask_prompt = criteria_prompt + prompt + '''\nScore:'''
    score = get_relevance_score_baseline(to_ask_prompt,pipeline,system_message)
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
        
#######


def sun_prompt_then_decomposed(query,passage,pipeline,log_file_path,system_message,qidx,docidx):
    
    decomposed_scores_dict = {}
    system_message_decomposition = """Please assess how well the provided passage meets specific criteria in relation to the query. Use the following scoring scale (0-3) for evaluation:
3: Highly relevant / Fully satisfies the criterion.
2: Fairly relevant / Adequately addresses the criterion.
1: Marginally relevant / Partially addresses the criterion.
0: Not relevant at all / No information provided."""
    
    prompt = f'''Instruction: Given a passage and a query, predict
whether the passage includes an answer to the query by producing either ”Yes” or ”No”.
Question: {query} Passage: {passage} Answer:'''


    rel_or_not = get_relevance_score_baseline(prompt,pipeline,"")

    # print(rel_or_not)
    if "yes" in rel_or_not.lower() and "no" not in rel_or_not.lower():
        decomposed_criterias = {
        "Exactness":" How precisely does the passage answer the query.",
        "Coverage": "how much of the passage is dedicated to discussing the query and its related topics.",
        }
        
        prompt = f'''Query: {query}\nPassage: {passage}\n'''
        prompt_ = prompt
        for i in range(len(decomposed_criterias)):
            criteria = list(decomposed_criterias.keys())[i]
            criteria_definition = list(decomposed_criterias.values())[i]
            criteria_prompt = f'''Please rate how well the given passage meets the {criteria} criteria in relation to the query. The output should be a single score (0-3) indicating {criteria_definition}.'''
            to_ask_prompt = criteria_prompt + prompt+'''\nScore:'''
                
            score = get_relevance_score_baseline(to_ask_prompt,pipeline,system_message_decomposition)
            num_score = find_first_number(score)
            prompt_+=f"\n{list(decomposed_criterias.keys())[i]}: {num_score}"
            decomposed_scores_dict[list(decomposed_criterias.keys())[i]]=num_score
            if not hasattr(sun_prompt_then_decomposed,"called"):
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
        
                
        rel_prompt = '''The given passage is relevant to the query, please rate how relevant it is to the query. The output must be only a score (2 or 3) that indicate how relevant they are.\n'''
        to_ask_prompt = rel_prompt + prompt_ + '''\nScore:'''
        score = get_relevance_score_baseline(to_ask_prompt,pipeline,create_system_message_for_rel)
        num_score = find_first_number(score)
        if not hasattr(sun_prompt_then_decomposed,"called"):
                sun_prompt_then_decomposed.called =True
                print("******")
                print(create_system_message_for_rel)
                print(to_ask_prompt)
                print(score)
                print("*"*20)
                print(num_score)
        inf['binary_rel'] = rel_or_not
                
        inf['relevance_prompt'] = to_ask_prompt
        inf['llm_response'] = score
        inf['final_relevance_score'] = num_score
        with open(log_file_path,"a") as f:
            json.dump(inf, f)
    
    elif "no" in rel_or_not.lower() and "yes" not in rel_or_not.lower():
    
        decomposed_criterias = {
        "Topicality": "Is the passage about the same subject as the whole query (not only a single word of it).",
        "Contextual Fit": "Does the passage provide relevant background or context.",
        }
        
        prompt = f'''Query: {query}\nPassage: {passage}\n'''
        prompt_ = prompt
        for i in range(len(decomposed_criterias)):
            criteria = list(decomposed_criterias.keys())[i]
            criteria_definition = list(decomposed_criterias.values())[i]
            criteria_prompt = f'''Please rate how well the given passage meets the {criteria} criteria in relation to the query. The output should be a single score (0-3) indicating "{criteria_definition}".'''
            to_ask_prompt = criteria_prompt + prompt+'''\nScore:'''
                
            score = get_relevance_score_baseline(to_ask_prompt,pipeline,system_message_decomposition)
            num_score = find_first_number(score)
            prompt_+=f"\n{list(decomposed_criterias.keys())[i]}: {num_score}"
            decomposed_scores_dict[list(decomposed_criterias.keys())[i]]=num_score
            if not hasattr(sun_prompt_then_decomposed,"called"):
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
        
                
        rel_prompt = '''The given passage is irrelevant to the query, please rate how irrelevant it is to the query. The output must be only a score (0 or 1) that indicate how irrelevant they are.\n'''
        to_ask_prompt = rel_prompt + prompt_ + '''\nScore:'''
        score = get_relevance_score_baseline(to_ask_prompt, pipeline,create_system_message_for_non_rel)
        num_score = find_first_number(score)
        if not hasattr(sun_prompt_then_decomposed,"called"):
                sun_prompt_then_decomposed.called =True
                print("******")
                print(create_system_message_for_non_rel)
                print(to_ask_prompt)
                print(score)
                print("*"*20)
                print(num_score)
        inf['binary_rel'] = rel_or_not
        inf['relevance_prompt'] = to_ask_prompt
        inf['llm_response'] = score
        inf['final_relevance_score'] = num_score
        with open(log_file_path,"a") as f:
            json.dump(inf, f)
    else:
        print(rel_or_not)
        num_score = 0
    return num_score , decomposed_scores_dict
            

def make_query_out_of_passage_relevance(query,passage,pipeline,log_file_path,qidx,docidx,passage_to_predicted_query):
    if docidx not in passage_to_predicted_query:
        query_generator_prompt = f'''Please identify the search query that best corresponds to the following passage. Keep your response concise.\n Passage: {passage}'''
        generated_query = get_relevance_score_baseline(query_generator_prompt,pipeline,"you are a query generator. for example having this document:'Categories: Dogs. Article Summary X. If your puppy is starting to get teeth, it's probably between 3 and 4 weeks old. At 8 weeks of age, your puppy will have 28 baby teeth. For an adult dog, expect 1 or 2-year-olds to have white teeth, while 3-year-olds may have signs of tooth decay, such as yellow and brown tartar.'\n you should generate a query such as : 'dog age by teeth'. ")
        passage_to_predicted_query[docidx] = generated_query
    else:
        generated_query = passage_to_predicted_query[docidx]
    
    
    
    rel_prompt = f'''Please rate the similarity between the following queries:

                        '{generated_query}'

                        and

                        '{query}'

                        3: Highest similarity

                        2: Fairly similar

                        1: Minor similarity

                        0: Not similar'''
    
    
    score = get_relevance_score_baseline(rel_prompt,pipeline,"You are a similarity evaluator agent. Please rate the similarity between the two items on a scale from 0 to 3.")
    num_score = find_first_number(score)
    inf = {"qidx":qidx,
                   "docidx":docidx,
                   "query": query,
                   "passage": passage,
                   "generated_query":generated_query,
                   "prompt":rel_prompt,
                   "llm_response":score,
                   "final_relevance_score":num_score
                   
            
        }
    with open(log_file_path,"a") as f:
        json.dump(inf, f)
    return num_score,passage_to_predicted_query
 
 
 
def get_relevance_score_decomposed_prompts(query,passage,pipeline,log_file_path,system_message,qidx,docidx):
    decomposed_scores_dict = {}
    # Define the initial prompt
    system_message_decomposition = """Please assess how well the provided passage meets specific criteria in relation to the query. Use the following scoring scale (0-3) for evaluation:

0: Not relevant at all / No information provided.
1: Marginally relevant / Partially addresses the criterion.
2: Fairly relevant / Adequately addresses the criterion.
3: Highly relevant / Fully satisfies the criterion."""

    # Define the hierarchical order of prompts
    decomposed_criterias = {
        "Exactness":" How precisely does the passage answer the query.",
        "Topicality": "Is the passage about the same subject as the whole query (not only a single word of it).",
        # "Depth": "How much detail does the passage provide about the topic",
        "Coverage": "How much of the passage is dedicated to discussing the query and its related topics.",
        "Contextual Fit": "Does the passage provide relevant background or context.",
    }
    prompt = f'''Query: {query}\nPassage: {passage}\n'''
    prompt_ = prompt
    for i in range(len(decomposed_criterias)):
        criteria = list(decomposed_criterias.keys())[i]
        criteria_definition = list(decomposed_criterias.values())[i]
        criteria_prompt = f'''Please rate how well the given passage meets the {criteria} criterion in relation to the query. The output should be a single score (0-3) indicating {criteria_definition}.'''
        to_ask_prompt = criteria_prompt + prompt+'''\nScore:'''
            
        score = get_relevance_score_baseline(to_ask_prompt,pipeline,system_message_decomposition)
        num_score = find_first_number(score)
        prompt_+=f"\n{list(decomposed_criterias.keys())[i]}: {num_score}"
        decomposed_scores_dict[list(decomposed_criterias.keys())[i]]=num_score
        if not hasattr(get_relevance_score_decomposed_prompts,"called"):
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
    
            
    criteria_prompt = '''Please rate how the given passage is relevant to the query based on the given scores. The output must be only a score that indicates how relevant they are.\n'''
    to_ask_prompt = criteria_prompt + prompt_ + '''\nScore:'''
    score = get_relevance_score_baseline(to_ask_prompt, pipeline, system_message)
    num_score = find_first_number(score)
    if not hasattr(get_relevance_score_decomposed_prompts,"called"):
            get_relevance_score_decomposed_prompts.called =True
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
        


 


 
def five_prompt(query,passage,pipeline,log_file_path,system_message,qidx,docidx,baseline_prompt=None,baseline_system_message=None):
    # baseline_relevance = get_relevance_score_baseline(baseline_prompt,pipeline,baseline_system_message)
    decomposed_scores_dict = {}
    # Define the initial prompt
    system_message_decomposition = """Please assess how well the provided passage meets specific criteria in relation to the query. Use the following scoring scale (0-3) for evaluation:

0: Not relevant at all / No information provided.
1: Marginally relevant / Partially addresses the criterion.
2: Fairly relevant / Adequately addresses the criterion.
3: Highly relevant / Fully satisfies the criterion."""

    # Define the hierarchical order of prompts
    decomposed_criterias = {
        "Exactness":" How precisely does the passage answer the query.",
        "Topicality": "Is the passage about the same subject as the whole query (not only a single word of it).",
        "Depth": "How much detail does the passage provide about the topic",
        "Coverage": "How much of the passage is dedicated to discussing the query and its related topics.",
        "Contextual Fit": "Does the passage provide relevant background or context.",
    }
    prompt = f'''Query: {query}\nPassage: {passage}\n'''
    prompt_ = prompt
    for i in range(len(decomposed_criterias)):
        criteria = list(decomposed_criterias.keys())[i]
        criteria_definition = list(decomposed_criterias.values())[i]
        criteria_prompt = f'''Please rate how well the given passage meets the {criteria} criterion in relation to the query. The output should be a single score (0-3) indicating {criteria_definition}.'''
        to_ask_prompt = criteria_prompt + prompt+'''\nScore:'''
            
        score = get_relevance_score_baseline(to_ask_prompt,pipeline,system_message_decomposition)
        num_score = find_first_number(score)
        prompt_+=f"\n{list(decomposed_criterias.keys())[i]}: {num_score}"
        decomposed_scores_dict[list(decomposed_criterias.keys())[i]]=num_score
        if not hasattr(get_relevance_score_decomposed_prompts,"called"):
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
    
    prompt_ += f"relevance score"        
    criteria_prompt = '''Please rate how the given passage is relevant to the query based on the given scores. The output must be only a score that indicates how relevant they are.\n'''
    to_ask_prompt = criteria_prompt + prompt_ + '''\nScore:'''
    score = get_relevance_score_baseline(to_ask_prompt, pipeline, system_message)
    num_score = find_first_number(score)
    if not hasattr(get_relevance_score_decomposed_prompts,"called"):
            get_relevance_score_decomposed_prompts.called =True
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
        


 




def get_relevance_score_baseline(prompt: str, pipeline, system_message: str):
    # Prepare messages for chat template
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    
    # Check if the function has been called before and log messages once
    if not hasattr(get_relevance_score_baseline, "called"):
        get_relevance_score_baseline.called = True
        print(messages)

    
    if isinstance(pipeline, TogetherPipeline):
        # Directly call Together API
        if not hasattr(get_relevance_score_baseline, "output_from_together"):
            get_relevance_score_baseline.output_from_together = True
            print("output is from a model loaded on together ai")
        outputs = pipeline(messages)
        output = outputs[0]["generated_text"]
        
    else:
        # Define terminators once for reuse
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        # Check if pipeline tokenizer supports chat templates and process accordingly
        if hasattr(pipeline.tokenizer, "apply_chat_template"):
            if hasattr(pipeline.tokenizer, 'chat_template') and pipeline.tokenizer.chat_template is not None:
                # Use chat template if supported and set
                prompt = pipeline.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback when chat template is not set
                if not hasattr(get_relevance_score_baseline, "warning"):
                    get_relevance_score_baseline.warning = True
                    print("Warning: Chat template not set, falling back to simple concatenation.")

                prompt = f"{system_message}\n{prompt}"
        else:
            # Fallback for models without chat template support
            prompt = f"{system_message}\n{prompt}"

        # Generate output from the model
        outputs = pipeline(
            prompt,
            max_new_tokens=100,
            eos_token_id=terminators,
            pad_token_id=128009,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )

        # Return generated text without the prompt if chat template was used, otherwise return full text
        if hasattr(pipeline.tokenizer, 'chat_template') and pipeline.tokenizer.chat_template is not None:
            output =  outputs[0]["generated_text"][len(prompt):]
        else:
            output = outputs[0]["generated_text"]
    if not hasattr(get_relevance_score_baseline, "print_one_output"):
        get_relevance_score_baseline.print_one_output = True
        print(f"sample output: {output}")    
    return output


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
            