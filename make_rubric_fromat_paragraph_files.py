import json
import os
import gzip
import re
def generate_json_line(query_id, paragraph_id, text, exactness_score, topicality_score, coverage_score, contextual_fit_score, relevance_label,query_text,ground_truth_relevance_label,passage_to_msmarco,qidtomsmarcoqids):
    # Define the structure of the JSON line
    json_line = [
        qidtomsmarcoqids[query_id],  # The query ID (e.g., q49)
        [
            {
                "paragraph_id": passage_to_msmarco[paragraph_id],  # The paragraph ID (e.g., p3659)
                "text": text,  # The paragraph text (e.g., passage)
                "paragraph": "",  # Empty or additional paragraph-related field (you can customize it)
                "paragraph_data": {
                    "judgments": [
                        {
                            "paragraphId": paragraph_id,
                            "query": qidtomsmarcoqids[query_id],
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
            ["aggregate", relevance_label]
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
            { "nugget_id": "aggregate", "self_rating": relevance_label }
          ],
          "prompt_type": "nugget",
          "relevance_label": relevance_label
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
                d[(query_id,document_id)]=relevance_score
    return d


def process_json_line(json_line,qrel_file_path): #qrel_path = the path to qrel of collection ,json_line:one entry of an input file in the llm4judge format
    qrel_dic = make_qrel_dic(qrel_file_path)
    # Extract the necessary fields from the JSON line
    qidx = json_line["qidx"]
    docidx = json_line["docidx"]
    query = json_line["query"]
    passage = json_line["passage"]
    
    # Extract the decomposed scores
    decomposed_scores = json_line["decomposed_scores_dict"]
    exactness_score = decomposed_scores["Exactness"]
    topicality_score = decomposed_scores["Topicality"]
    coverage_score = decomposed_scores["Coverage"]
    contextual_fit_score = decomposed_scores["Contextual Fit"]
    
    # Extract the final relevance score
    relevance_label = json_line["final_relevance_score"]
    
    # Build the data structure for the entry
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
        "relevance_label": relevance_label if relevance_label is not None else max(exactness_score,topicality_score,coverage_score, contextual_fit_score),
        "query_text":query,
        "ground_truth_relevance_label":groundtruth
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
                processed_data = process_json_line(obj, qrel_file_path)
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


passage_to_msmarco = make_mapping_dict("./private_data/gold_data/docid_to_docidx.txt")
qid_to_qidx = make_mapping_dict("./private_data/gold_data/qid_to_qidx.txt")


qrel_file_path = "./private_data/gold_data/llm4eval_test_qrel_2024_withRel.txt"
input_file = "./logs/test_decomposed_relavance_qrel.json"


data_to_write = generate_data_to_write(input_file,qrel_file_path)

output_dir = "./rubric_format_inputs/"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Open the file to write the JSON lines in compressed text mode
with gzip.open(f"./rubric_format_inputs/{os.path.basename(input_file).replace('.json', '.jsonl.gz')}", 'wt', encoding='utf-8') as f:
    for entry in data_to_write:
        json_line = generate_json_line(
            entry["query_id"], entry["paragraph_id"], entry["text"], 
            entry["exactness_score"], entry["topicality_score"], entry["coverage_score"],
            entry["contextual_fit_score"], entry["relevance_label"], entry["query_text"], 
            entry["ground_truth_relevance_label"], passage_to_msmarco, qid_to_qidx
        )
        f.write(json_line + '\n')  # Write each JSON line in the compressed file
