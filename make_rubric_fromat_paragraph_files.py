import json
import os
import gzip
import re
def generate_json_line(query_id, paragraph_id, text, exactness_score, topicality_score, coverage_score, contextual_fit_score, relevance_label, final_relevance_label, threshold_on_sum, query_text, ground_truth_relevance_label, binary_rel=None, passage_to_msmarco=None, qidtomsmarcoqids=None, relevance_score_from_generated_qrel=False, generated_qrel_dict=None, sum_flag = False):
    # Define the structure of the JSON line
    json_line = [
        str(qidtomsmarcoqids[query_id]) if qidtomsmarcoqids else str(query_id),  # The query ID (e.g., q49)
        [
            {
                "paragraph_id": str(passage_to_msmarco[paragraph_id]) if passage_to_msmarco else str(paragraph_id),  # The paragraph ID (e.g., p3659)
                "text": text,  # The paragraph text (e.g., passage)
                "paragraph": "",  # Empty or additional paragraph-related field (you can customize it)
                "paragraph_data": {
                    "judgments": [
                        {
                            "paragraphId": str(passage_to_msmarco[paragraph_id]) if passage_to_msmarco else str(paragraph_id),  # The paragraph ID (e.g., p3659)
                            "query": str(qidtomsmarcoqids[query_id]) if qidtomsmarcoqids else str(query_id),
                            "relevance": ground_truth_relevance_label,  # The relevance score
                            "titleQuery": query_text  # The title or name for the query
                        }
                    ],
                    "rankings": []  # Can be used for any rankings you might want to include
                },
                "exam_grades": [{
                    "correctAnswered": [
                        "Exactness",
                        "Topicality",
                        "Coverage",
                        "Contextual Fit"
                    ],
                    "wrongAnswered": [],
                    "answers": [
                        ["Exactness", str(exactness_score)],
                        ["Topicality", str(topicality_score)],
                        ["Coverage", str(coverage_score)],
                        ["Contextual Fit", str(contextual_fit_score)]
                    ],
                    "llm_response_errors": {},
                    "llm": "meta-llama/Meta-Llama-3-8B-Instruct",  # Example LLM model
                    "llm_options": {},
                    "exam_ratio": 1,
                    "prompt_info": {
                        "prompt_class": "FourPrompts",
                        "prompt_style": "Relevance Criteria",
                        "context_first": False,
                        "check_unanswerable": False,
                        "check_answer_key": True,
                        "is_self_rated": True,
                        "rating_extractor": "SelfRaterStrict",
                        "before_conversion_ids": (paragraph_id,query_id) if passage_to_msmarco else None,
                        "relevance_label_info": "the max of criterion",
                        "binary_relevance": binary_rel 
                    },
                    "self_ratings": [
                        {"nugget_id": "Exactness", "self_rating": exactness_score},
                        {"nugget_id": "Coverage", "self_rating": coverage_score},
                        {"nugget_id": "Topicality", "self_rating": topicality_score},
                        {"nugget_id": "Contextual Fit", "self_rating": contextual_fit_score}
                    ],
                    "prompt_type": "nugget",
                    "relevance_label": relevance_label
                },
                {
          "correctAnswered": ["aggregate"],
          "wrongAnswered": [],
          "answers": [
            ["aggregate", threshold_on_sum if sum_flag 
                            else (
                                final_relevance_label if not relevance_score_from_generated_qrel 
                                else generated_qrel_dict[(query_id, paragraph_id)]
                            )]
          ],
          "llm_response_errors": {},
          "llm": "meta-llama/Meta-Llama-3-8B-Instruct",
          "llm_options": {},
          "exam_ratio": 1,
          "prompt_info": {
            "prompt_class": "FourAggregationPrompt",
            "prompt_style": "Prompt to aggregate scores",
            "context_first": False,
            "check_unanswerable": False,
            "check_answer_key": True,
            "is_self_rated": True
          },
          "self_ratings": [
            { "nugget_id": "aggregate", "self_rating": threshold_on_sum if sum_flag 
                                                        else (
                                                            final_relevance_label if not relevance_score_from_generated_qrel 
                                                            else generated_qrel_dict[(query_id, paragraph_id)]
                                                        )}
          ],
          "prompt_type": "nugget",
          "relevance_label": threshold_on_sum if sum_flag 
                            else (
                                final_relevance_label if not relevance_score_from_generated_qrel 
                                else generated_qrel_dict[(query_id, paragraph_id)]
                            )
        }
      ]
    }
  ]
]
    # print(json_line)
    return json.dumps(json_line)





def make_mapping_dict(doc_mapping_path):
    passage_to_msmarco = {}
    with open(doc_mapping_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                msmarcoid, passaageid = parts
                passage_to_msmarco[passaageid] = msmarcoid
    return passage_to_msmarco
    

           
def make_qrel_dic(qre_file_path):
    d = {}
    with open(qre_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                query_id = parts[0]
                document_id = parts[2]
                relevance_score = int(parts[3])
                try:
                    d[(int(query_id),int(document_id))]=relevance_score
                except:
                    # print((query_id,document_id))
                    d[(query_id,document_id)]=relevance_score
                    
    # print(d)
    return d



def threshold_for_sum(overal_sum):
    if 10<=overal_sum <=12:
        overal_score = 3
    elif 7<=overal_sum <=9:
        overal_score = 2
    if 5<=overal_sum <=6:
        overal_score = 1
    if 0<=overal_sum <=5: 
        overal_score = 0 
    return overal_score

def process_json_line(json_line, qrel_dic): #qrel_path = the path to qrel of collection ,json_line:one entry of an input file in the llm4judge format
    # Extract the necessary fields from the JSON line
    qidx = json_line["qidx"]
    docidx = json_line["docidx"]
    query = json_line["query"]
    passage = json_line["passage"]
    try:
        binary_rel = json_line["binary_rel"]
    except:
        binary_rel = None
    
    
    exactness_score, coverage_score, topicality_score, contextual_fit_score = 0, 0, 0, 0
    # Extract the decomposed scores
    if binary_rel:
        if binary_rel.lower() == "yes":
            decomposed_scores = json_line["decomposed_scores_dict"]
            coverage_score = decomposed_scores["Coverage"]
            exactness_score = decomposed_scores["Exactness"]
        elif binary_rel.lower() == "no":
            decomposed_scores = json_line["decomposed_scores_dict"]
            topicality_score = decomposed_scores["Topicality"]
            contextual_fit_score = decomposed_scores["Contextual Fit"]
    else:
        decomposed_scores = json_line["decomposed_scores_dict"]
        coverage_score = decomposed_scores["Coverage"]
        exactness_score = decomposed_scores["Exactness"]
        topicality_score = decomposed_scores["Topicality"]
        contextual_fit_score = decomposed_scores["Contextual Fit"]
    # Extract the final relevance score
    relevance_label = json_line["final_relevance_score"]
    
    # Build the data structure for the entry
    try:
        groundtruth = qrel_dic[(qidx,int(docidx))]
    except:
        groundtruth = qrel_dic[(qidx,docidx)]
        
    exactness_score = exactness_score if exactness_score is not None else 0
    topicality_score = topicality_score if topicality_score is not None else 0
    coverage_score = coverage_score if coverage_score is not None else 0
    contextual_fit_score = contextual_fit_score if contextual_fit_score is not None else 0
    
    return {
        "query_id": qidx,
        "paragraph_id": docidx,
        "text": passage,
        "exactness_score": exactness_score,
        "topicality_score": topicality_score,
        "coverage_score": coverage_score,
        "contextual_fit_score": contextual_fit_score,
        "relevance_label": max(exactness_score,topicality_score,coverage_score, contextual_fit_score),
        "final_relevance_label": relevance_label if relevance_label is not None else max(exactness_score,topicality_score,coverage_score, contextual_fit_score),
        "query_text":query,
        "ground_truth_relevance_label":groundtruth,
        "binary_rel": binary_rel,
        "threshold_on_sum": threshold_for_sum(sum([exactness_score, topicality_score, coverage_score, contextual_fit_score]))
            
    }

# def generate_data_to_write(input_file, qrel_file_path): #input is my llm4judge format of logs .json file
    # List to hold all the processed data entries
    data_to_write = []

    # Open the file and process it
    with open(input_file, 'r', encoding='utf-8') as f:
        # Read the entire content of the file as a single string
        content = f.read()

        # Use regex to find all valid JSON objects (by splitting on closing and opening curly braces)
        json_objects = re.findall(r'\{.*?\}', content, re.DOTALL)
        
        # Process each JSON object
        for line_number, json_str in enumerate(json_objects, start=1):
            try:
                # Parse the individual JSON object
                entry = json.loads(json_str)
                processed_data = process_json_line(entry, qrel_file_path)
                data_to_write.append(processed_data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line {line_number}: {json_str[:80]}...")
                print(f"Error details: {e}")
    
    return data_to_write


def generate_data_to_write(filename,qrel_file_path):
    qrel_dic = make_qrel_dic(qrel_file_path)
    
    entries = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
        
        # Keep track of where we are in the string
        pos = 0
        while pos < len(content):
            try:
                # Parse one JSON object at a time
                decoder = json.JSONDecoder()
                obj, idx = decoder.raw_decode(content[pos:])
                processed_data = process_json_line(obj, qrel_dic)
                entries.append(processed_data)
                # Move to the end of the current object
                pos += idx
                
            except json.JSONDecodeError as e:
                # Skip any whitespace
                if content[pos].isspace():
                    pos += 1
                    continue
                else:
                    print(f"Error parsing at position {pos}: {e}")
                    break
    
    return entries

output_name = None



passage_to_msmarco = make_mapping_dict("./private_data/gold_data/docid_to_docidx.txt")
qid_to_qidx = make_mapping_dict("./private_data/gold_data/qid_to_qidx.txt")
qrel_file_path = "./private_data/gold_data/llm4eval_test_qrel_2024_withRel.txt"
input_file = "./logs/test_decomposed_relavance_qrel_llama70b.json"
# input_file = "./logs/llmjudge_test_4prompts_qrel_flant5large.json"
# input_file = "./logs/llmjudge_test_qrel_sun_then_decomposed_flant5large.json"
# input_file = "./logs/test_decomposed_relavance_qrel.json"
# input_file = "./logs/test_sun_then_decomposed_relavance_qrel.json"
# generated_qrel_dict = make_qrel_dic("./results/test_NaiveB_on_decomposed.txt")
# output_name = "test_sum_of_decomposed_flant5large.json"


# qrel_file_path = "./data/dl2019/2019qrels-pass.txt"
# input_file = "./logs/dl2019_test_sun_then_decomposed_flant5large.json"
# input_file = "./logs/dl2019_test_4prompts_flant5large.json"
# input_file = "./logs/4_prompts_dl2019.json"
# input_file = "./logs/dl2019_sun_then_decomposed_relavance_qrel.json"
# output_name = "dl2019_sum_of_decomposed_flant5large.json"

# qrel_file_path = "./data/dl2020/2020qrels-pass.txt"
# input_file = "./logs/dl2020_test_sun_then_decomposed_flant5large.json"
# input_file = "./logs/dl2020_test_4prompts_flant5large.json"
# input_file = "./logs/4_prompts_dl2020.json"
# input_file = "./logs/dl2020_sun_then_decomposed_relavance_qrel.json"
# output_name = "dl2020_sum_of_decomposed_flant5large.json"

# input_files = ["./logs/4_prompts_dl2019.json","./logs/4_prompts_dl2020.json", "./logs/test_sun_then_decomposed_relavance_qrel.json","./logs/test_gen_query_similarity_qrel.json","./logs/dl2019_gen_query_similarity_qrel.json","./logs/dl2019_sun_then_decomposed_relavance_qrel.json","./logs/dl2020_sun_then_decomposed_relavance_qrel.json"]
# for input_file in input_files:
data_to_write = generate_data_to_write(input_file,qrel_file_path)

output_dir = "./rubric_format_inputs/"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# output_name = "test_sum_of_decomposed_prompts.json"
output_full_name = f"./rubric_format_inputs/{os.path.basename(input_file).replace('.json', '.jsonl.gz')}" if output_name is None else f"./rubric_format_inputs/{os.path.basename(output_name).replace('.json', '.jsonl.gz')}"
# Open the file to write the JSON lines in compressed text mode
with gzip.open(output_full_name, 'wt', encoding='utf-8') as f:
    visited = []
    for entry in data_to_write:
        if (entry["query_id"], entry["paragraph_id"]) not in visited:
            visited.append((entry["query_id"], entry["paragraph_id"]))
            
            binary_rel = entry["binary_rel"] if entry["binary_rel"] else None
            
            json_line = generate_json_line(
                entry["query_id"], entry["paragraph_id"], entry["text"], 
                entry["exactness_score"], entry["topicality_score"], entry["coverage_score"],
                entry["contextual_fit_score"], entry["relevance_label"], entry["final_relevance_label"], entry["threshold_on_sum"], entry["query_text"], 
                
                entry["ground_truth_relevance_label"] 
                , passage_to_msmarco=passage_to_msmarco, qidtomsmarcoqids=qid_to_qidx
                # , binary_rel=binary_rel
                # , relevance_score_from_generated_qrel=True, generated_qrel_dict=generated_qrel_dict
                # , sum_flag=True
                
            )
            f.write(json_line + '\n')  # Write each JSON line in the compressed file
