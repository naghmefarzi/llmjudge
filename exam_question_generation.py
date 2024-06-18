# from prompts import exam_prompt_generate_question
from typing import Tuple, List, Dict, Callable, NewType, Optional, Iterable
from hashlib import md5

def interact_with_model(system_message:str, prompt: str, pipeline, max_new_tokens: Optional[int]):
  messages = [
      {"role": "system", "content": system_message},
      {"role": "user", "content": prompt},
  ]
  
  
  if not hasattr(interact_with_model, "called"):
    interact_with_model.called = True
    print(messages)

  prompt = pipeline.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True,
          format = "JSON"
  )

  terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = pipeline(
      prompt,
      max_new_tokens=max_new_tokens,
      eos_token_id=terminators,
      pad_token_id=128009,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )

  return outputs[0]["generated_text"][len(prompt):]

def constants_q_generation(query_text:str)-> Tuple[str,str]:
    system_message = '''You are an advanced query decomposer.
    Generate concise, deep questions for query assessment related to a topic.
    Avoid basic questions. Return every answer in JSON format.'''
    prompt = '''Break the query ’{query_text}’ into concise questions that must be answered. 
    Generate 10 concise insightful questions that reveal whether information relevant for ’{query_text}’ was provided, showcasing a deep understanding of the subject matter. 
    Avoid basic or introductory-level inquiries. Keep the questions short.'''
    return system_message, prompt
    
def constant_q_rating(question:str,context:str)-> str:
    return '''Can the question be answered based on the available con- text? choose one:
    - 5: The answer is highly relevant, complete, and accurate.
    - 4: The answer is mostly relevant and complete but may have minor gaps or inaccuracies.
    - 3: The answer is partially relevant and complete, with noticeable gaps or inaccuracies.
    - 2: The answer has limited relevance and completeness, with significant gaps or inaccuracies.
    - 1: The answer is minimally relevant or complete, with substantial shortcomings.
    - 0: The answer is not relevant or complete at all.
    Question: {question} Context: {context}'''
    

def generate_question_set(test_qrel_path: str, test_qrel, qid_to_query, pipe):
    question_set_file_path = test_qrel_path.replace(".txt","_exam.jsonl")
    for qidx in test_qrel['qid'].unique():
        q_generation_system_message, q_generation_prompt = constants_q_generation(qid_to_query[qidx])
        output = interact_with_model(q_generation_system_message,q_generation_prompt,pipe,1000)
        print(output)
        break
        # question_set_file_path.dump({qidx:{md5(question):question for question in question_set}})
    