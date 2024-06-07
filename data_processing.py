import os
import jsonlines
import pandas as pd
from typing import Tuple, List, Dict, Callable, NewType, Optional, Iterable


def load_data_files(docs_path: str, queries_path: str, test_qrel_path : str):
    docid_to_doc = get_all_docid_to_doc(docs_path)
    qid_to_query = get_all_query_id_to_query(queries_path)
    test_qrel = pd.read_csv(test_qrel_path, sep=" ", header=None, names=['qid', 'Q0', 'docid','rel_score'])
    return docid_to_doc, qid_to_query, test_qrel
      
def clean_files(*file_paths):
    """
    Removes the specified files if they exist.
    
    Args:
        *file_paths (str): Paths to the files to be removed.
    """
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)

def get_all_docid_to_doc(docs_path: str = './data/llm4eval_document_2024.jsonl') -> Dict[str,str]:
    docid_to_doc = dict()
    with jsonlines.open(docs_path, 'r') as document_file:
        for obj in document_file:
            docid_to_doc[obj['docid']] = obj['doc']
    return docid_to_doc

def get_all_query_id_to_query(query_path:str) -> Dict[str,str]:
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

