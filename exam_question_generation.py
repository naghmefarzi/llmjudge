from prompts import exam_prompt_generate_question
from hashlib import md5



def generate_question_set(test_qrel_path, test_qrel, qid_to_query):
    question_set_file_path = test_qrel_path.replace(".txt","_exam.jsonl")
    for qidx in test_qrel['qid'].unique():
        question_set = exam_prompt_generate_question(qid_to_query[qidx])
        question_set.split("\n")
        question_set_file_path.dump({qidx:{md5(question):question for question in question_set}})
    