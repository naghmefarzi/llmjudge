from fuzzywuzzy import process
import pandas as pd
import os
class QueryMapper:
    def __init__(self, test_queries_path, llm4eval_queries_path):
        """
        Initialize query mapper with paths to query files
        
        Expected file format:
        query_id \t query_text
        """
        self.test_queries = self._load_queries(test_queries_path)
        self.llm4eval_queries = self._load_queries(llm4eval_queries_path)
    
    def _load_queries(self, query_file_path):
        """
        Load queries from file into a dictionary
        
        Returns:
        {query_id: query_text}
        """
        queries = {}
        with open(query_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) <2:
                    parts = line.strip().split(' ')
                    
                if len(parts) >= 2:
                    query_id, query_text = parts[0], parts[1]
                    queries[query_id] = query_text
        # print(queries)
        return queries
    
    def map_queries(self, similarity_threshold=80):
        """
        Map queries between test queries and LLM4Eval queries
        
        Args:
        - similarity_threshold: Minimum fuzzy match score (0-100)
        
        Returns:
        DataFrame with mapping information
        """
        query_mappings = []
        
        for llm4eval_id, llm4eval_text in self.llm4eval_queries.items():
            # Find best match in test queries
            best_match = process.extractOne(
                llm4eval_text, 
                self.test_queries.values()
            )
            
            if best_match and best_match[1] >= similarity_threshold:
                # Find the corresponding test query ID
                matched_test_query_id = [
                    test_id for test_id, test_text in self.test_queries.items() 
                    if test_text == best_match[0]
                ][0]
                
                query_mappings.append({
                    'llm4eval_query_id': llm4eval_id,
                    'llm4eval_query_text': llm4eval_text,
                    'test_query_id': matched_test_query_id,
                    'test_query_text': best_match[0],
                    'similarity_score': best_match[1]
                })
        # print(query_mappings)
        return pd.DataFrame(query_mappings)
    
    def preprocess_qrel(self, input_qrel_path, output_qrel_path, mapping_df):
        """
        Preprocess qrel file by replacing query IDs
        
        Args:
        - input_qrel_path: Original qrel file path
        - output_qrel_path: Preprocessed qrel file path
        - mapping_df: DataFrame with query mappings
        """
        # Create a mapping dictionary for quick lookup
        query_id_map = dict(zip(
            mapping_df['test_query_id'],
            mapping_df['llm4eval_query_id']
            
        ))
        
        with open(input_qrel_path, 'r') as input_file, \
             open(output_qrel_path, 'w') as output_file:
            
            for line in input_file:
                parts = line.strip().split()
                test_query_id = parts[0]
                
                # If query can be mapped, replace the ID
                if test_query_id in query_id_map:
                    new_query_id = query_id_map[str(test_query_id)]
                    new_line = f"{new_query_id} {' '.join(parts[1:])}\n"
                    output_file.write(new_line)
                else:
                    # Optionally log unmapped queries or handle differently
                    print(f"Could not map query ID: {test_query_id}")
    
    def preprocess_run_files(self, input_run_path, output_run_path, mapping_df):
        """
        Preprocess run files by replacing query IDs
        
        Args:
        - input_run_path: Original run file path
        - output_run_path: Preprocessed run file path
        - mapping_df: DataFrame with query mappings
        """
        # Create a mapping dictionary for quick lookup
        query_id_map = dict(zip(
            mapping_df['test_query_id'],
            mapping_df['llm4eval_query_id'] 
            
        ))
        print(query_id_map)
        with open(input_run_path, 'r') as input_file, \
             open(output_run_path, 'w') as output_file:
            
            for line in input_file:
                parts = line.strip().split()
                test_query_id = parts[0]
                
                # If query can be mapped, replace the ID
                if test_query_id in query_id_map:
                    new_query_id = query_id_map[str(test_query_id)]
                    new_line = f"{new_query_id} {' '.join(parts[1:])}\n"
                    output_file.write(new_line)
                else:
                    # Optionally log unmapped queries or handle differently
                    print(f"Could not map query ID: {test_query_id}")
                # break

# Usage example
def main():
    # Initialize the query mapper
    mapper = QueryMapper(
        test_queries_path='./runs_TRECDL_2023/TREC-DLs-files/TREC-DL-2023/test-queries.tsv',
        llm4eval_queries_path='./data/llm4eval_query_2024.txt'
    )
    
    # Generate query mappings
    mapping_df = mapper.map_queries(similarity_threshold=80)
    
    # Print mapping details
    # print(mapping_df)
    
    # Preprocess qrel file
    mapper.preprocess_qrel(
        input_qrel_path='./runs_TRECDL_2023/TREC-DLs-files/TREC-DL-2023/qrels-pass.txt',
        output_qrel_path='./runs_TRECDL_2023/TREC-DLs-files/TREC-DL-2023/processed-qrels-pass.txt',
        mapping_df=mapping_df
    )
    
    
    
    # input_runs_path='./runs_TRECDL_2023/passages-runs/'
    
    # run_files = [os.path.join(input_runs_path, file) for file in os.listdir(input_runs_path) if os.path.isfile(os.path.join(input_runs_path, file))]
    # # Preprocess run files
    # for run_path in run_files:
    #     mapper.preprocess_run_files(
    #         input_run_path= run_path,
    #         output_run_path=run_path.replace('passages-runs','processed-passages-runs'),
    #         mapping_df=mapping_df
    #     )

if __name__ == "__main__":
    main()