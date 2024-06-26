# query.jsonl.gz -- which is a Dict[str,str] from query id to query text
from data_processing import *
from typing import Tuple, List, Dict, Callable, NewType, Optional, Iterable
import pandas as pd
import json

import gzip

q_path = "./data/llm4eval_query_2024.txt"




def get_all_query_id_to_query(query_path:str,list_unique_qids:List['str']) -> Dict[str,str]:
    qs=[]
    query_data = pd.read_csv(query_path, sep="\t", header=None, names=['qid', 'qtext'])
    qid_to_query = dict(zip(query_data.qid, query_data.qtext))
    for q in qid_to_query.items():
        print(qs)
        if q[0] in list_unique_qids:
            qs.append({q[0]:q[1]})
    return qs
def make_query_jsonl_file(path,unique_qids):
    with gzip.open(path, 'w') as fout:
        data = get_all_query_id_to_query(q_path,unique_qids)
        fout.write(json.dumps(data).encode('utf-8'))  
        
dev_path_qrel = "./data/llm4eval_dev_qrel_2024.txt"
test_path_qrel = "./data/llm4eval_test_qrel_2024.txt"
dev_qrel = pd.read_csv(dev_path_qrel, sep=" ", header=None, names=['qid', 'Q0', 'docid','rel_score'])
unique_dev_qids = list(set(dev_qrel['qid']))
test_qrel = pd.read_csv(test_path_qrel, sep=" ", header=None, names=['qid', 'Q0', 'docid','rel_score'])
unique_test_qids = list(set(test_qrel['qid']))        
path = "llm4judge_dev_queries.jsonl.gz"
make_query_jsonl_file(path,unique_dev_qids)
path = "llm4judge_test_queries.jsonl.gz"
make_query_jsonl_file(path,unique_test_qids)
    
    
    
    
from pydantic import BaseModel
from typing import List, Any, Optional, Dict, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import gzip
import json
from pathlib import Path



from .pydantic_helper import pydantic_dump

class SelfRating(BaseModel):
    question_id:Optional[str]
    nugget_id:Optional[str]=None
    self_rating:int
    def get_id(self)->str:
        if self.question_id is not None:
            return self.question_id
        elif self.nugget_id is not None:
            return self.nugget_id
        else:
            raise RuntimeError("Neither question_id nor nugget_id is given.")



class ExamGrades(BaseModel):
    correctAnswered: List[str]               # [question_id]
    wrongAnswered: List[str]                 # [question_id]
    answers: List[Tuple[str, str]]           # [ [question_id, answer_text]] 
    llm: str                                 # huggingface model name
    llm_options: Dict[str,Any]               # anything that seems relevant
    exam_ratio: float                        # correct / all questions
    prompt_info: Optional[Dict[str,Any]]     # more info about the style of prompting
    self_ratings: Optional[List[SelfRating]] # if availabel: self-ratings (question_id, rating)
    prompt_type: Optional[str]

    def self_ratings_as_iterable(self):
        if self.self_ratings is None:
            return []
        else:
            return self.self_ratings


class Grades(BaseModel):
    correctAnswered: bool               # true if relevant,  false otherwise
    answer: str                        #  llm_response_text
    llm: str                                 # huggingface model name  google/flan-t5-large
    llm_options: Dict[str,Any]               # anything that seems relevant
    prompt_info: Optional[Dict[str,Any]]     # more info about the style of prompting
    self_ratings: Optional[int]         #  if available: self-rating (e.g. 0-5)

        # must have fields:
        #  prompt_info["prompt_class"]="FagB"
            # info =  {
            #       "prompt_class":  Fag  # or self.__class__.__name__
            #     , "prompt_style":  old_prompt("prompt_style", "question-answering prompt")
            #     , "is_self_rated": false # false if not self-rated, otherwise true
            #     }

@dataclass
class GradeFilter():
    model_name: Optional[str]
    prompt_class: Optional[str]
    is_self_rated: Optional[bool]
    min_self_rating: Optional[int]
    question_set:Optional[str]

    @staticmethod
    def noFilter():
        return GradeFilter(model_name=None, prompt_class=None, is_self_rated=None, min_self_rating=None, question_set=None)

    def filter_grade(self, grade:Grades)-> bool:
        return self.filter(grade)

    def filter(self, grade:Union[ExamGrades,Grades])-> bool:
        # Note, the following code is based on inverse logic -- any grade that DOES NOT meet set filter requirements is skipped

        # grades are marked as using this model
        if self.model_name is not None:
            if grade.llm is None:  # old run, where we did not expose this was a flan-t5-large run
                if self.model_name == "google/flan-t5-large":
                    pass # this is acceptable
                else:
                    return False  # we are asking for a different model

            if not grade.llm == self.model_name:  # grade.llm is set, so lets see whether it matches
                return False    # sad trombone, did not match

        # grade.prompt_info is marked as using this prompt_class
        if self.prompt_class is not None:
            if grade.prompt_info is not None:
                grade_prompt_class = grade.prompt_info.get("prompt_class", None)
                if grade_prompt_class is not None:
                    if not grade_prompt_class == self.prompt_class:
                        return False
            elif not self.prompt_class == "QuestionPromptWithChoices":  # handle case before we tracked prompt info
                return False


        # grade.prompt_info is marked as is_self_rated
        if self.is_self_rated is not None:
            if grade.prompt_info is not None:
                grade_is_self_rated = grade.prompt_info.get("is_self_rated", None)
                if grade_is_self_rated is not None:
                    if not grade_is_self_rated == self.is_self_rated:
                        return False


        if isinstance(grade, ExamGrades):
            # for at least one question, the self_rating is at least self.min_self_rating
            if self.min_self_rating is not None:
                if grade.self_ratings is not None and len(grade.self_ratings)>0:  # grade has self_ratings
                    if not any( (rating.self_rating >= self.min_self_rating  for rating in grade.self_ratings) ):
                        return False

            if self.question_set is not None:
                if self.question_set == "tqa":
                    is_tqa_question = grade.answers[0][0].startswith("NDQ_")
                    if not is_tqa_question:
                        return False
                    
                if self.question_set == "genq":
                    is_genq_question = grade.answers[0][0].startswith("tqa2:")
                    if not is_genq_question:
                        return False

        return True

    def get_min_grade_filter(self, min_self_rating:int):
        return GradeFilter(model_name=self.model_name
                           , prompt_class=self.prompt_class
                           , is_self_rated=self.is_self_rated
                           , min_self_rating=min_self_rating
                           , question_set=self.question_set)

class Judgment(BaseModel):
    paragraphId : str
    query : str
    relevance : int
    titleQuery : str
    

class ParagraphRankingEntry(BaseModel):
    method : str
    paragraphId : str
    queryId : str # can be title query or query facet, e.g. "tqa2:L_0002/T_0020"
    rank : int
    score : float
    

class ParagraphData(BaseModel):
    judgments : List[Judgment]
    rankings : List[ParagraphRankingEntry] 


class FullParagraphData(BaseModel):
    paragraph_id : str
    text : str
    paragraph : Any
    paragraph_data : ParagraphData
    exam_grades : Optional[List[ExamGrades]]
    grades: Optional[List[Grades]]

    def retrieve_exam_grade_any(self, grade_filter:GradeFilter) -> List[ExamGrades]:
        if self.exam_grades is None:
            return []
        
        found = next((g for g in self.exam_grades if grade_filter.filter(g)), None)
        if found is not None:
            return [found]
        else: 
            return []
        

    def retrieve_exam_grade_all(self, grade_filter:GradeFilter) -> List[ExamGrades]:
        if self.exam_grades is None:
            return []
        
        # result = list(g for g in self.exam_grades if grade_filter.filter(g))
        # return result
        return [g for g in self.exam_grades if grade_filter.filter(g)]
        

    def retrieve_grade_any(self, grade_filter:GradeFilter) -> List[Grades]:
        if self.grades is None:
            return []
        
        found = next((g for g in self.grades if grade_filter.filter_grade(g)), None)
        if found is not None:
            return [found]
        else: 
            return []
        


    def exam_grades_iterable(self)-> List[ExamGrades]:
        return [] if self.exam_grades is None else self.exam_grades

       
    def get_any_judgment(self)->Optional[Judgment]:
        if self.paragraph_data.judgments is None or len(self.paragraph_data.judgments)<1: 
            return None
        else: 
            return self.paragraph_data.judgments[0]

    def get_any_ranking(self, method_name:str)->Optional[ParagraphRankingEntry]:
        if self.paragraph_data.rankings is None or len(self.paragraph_data.rankings)<1: 
            return None
        else:
            return next((item for item in self.paragraph_data.rankings if item.method==method_name), None)

@dataclass
class QueryWithFullParagraphList():
    queryId:str
    paragraphs: List[FullParagraphData]


def parseQueryWithFullParagraphList(line:str) -> QueryWithFullParagraphList:
    # Parse the JSON content of the line
    # print(line)
    data = json.loads(line)
    return QueryWithFullParagraphList(data[0], [FullParagraphData.parse_obj(paraInfo) for paraInfo in data[1]])


# Path to the benchmarkY3test-qrels-with-text.jsonl.gz file
def parseQueryWithFullParagraphs(file_path:Path) -> List[QueryWithFullParagraphList] :
    '''Load JSONL.GZ file with exam annotations in FullParagraph information'''
    # Open the gzipped file

    result:List[QueryWithFullParagraphList] = list()
    try: 
        with gzip.open(file_path, 'rt', encoding='utf-8') as file:
            # return [parseQueryWithFullParagraphList(line) for line in file]
            for line in file:
                result.append(parseQueryWithFullParagraphList(line))
    except  EOFError as e:
        print("Warning: Gzip EOFError on {file_path}. Use truncated data....\nFull Error:\n{e}")
    return result



def dumpQueryWithFullParagraphList(queryWithFullParagraph:QueryWithFullParagraphList)->str:
    '''Write `QueryWithFullParagraphList` to jsonl.gz'''
    return  json.dumps ([queryWithFullParagraph.queryId,[p.dict(exclude_none=True) for p in queryWithFullParagraph.paragraphs]])+"\n"

def writeQueryWithFullParagraphs(file_path:Path, queryWithFullParagraphList:List[QueryWithFullParagraphList]) :
    # Open the gzipped file
    with gzip.open(file_path, 'wt', encoding='utf-8') as file:
        # Iterate over each line in the file
        file.writelines([dumpQueryWithFullParagraphList(x) for x in queryWithFullParagraphList])



def main():
    """Entry point for the module."""
    x = parseQueryWithFullParagraphs("./benchmarkY3test-qrels-with-text.jsonl.gz")
    print(x[0])

if __name__ == "__main__":
    main()




def writeQueryWithFullParagraphs(file_path:Path, queryWithFullParagraphList:List[QueryWithFullParagraphList]) :
    # Open the gzipped file
    with gzip.open(file_path, 'wt', encoding='utf-8') as file:
        # Iterate over each line in the file
        file.writelines([dumpQueryWithFullParagraphList(x) for x in queryWithFullParagraphList])