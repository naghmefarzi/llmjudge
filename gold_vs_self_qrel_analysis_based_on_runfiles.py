import argparse
import numpy as np
from scipy import stats
import pandas as pd
import os

class QrelComparator:
    def __init__(self, human_qrel_path, self_qrel_path):
        """
        Initialize the comparator with paths to qrel files
        
        Qrel file structure expected:
        - Format: query_id document_id relevance_score
        - Example:
          Q1 doc1 1
          Q1 doc2 0
          Q2 doc1 1
          ...
        """
        self.human_qrel = self._load_qrel(human_qrel_path)
        self.self_qrel = self._load_qrel(self_qrel_path)
    
    def _load_qrel(self, qrel_path):
        """
        Load qrel file into a dictionary
        
        Returns:
        {
            'query1': {'doc1': relevance_score, 'doc2': relevance_score},
            ...
        }
        """
        qrel_dict = {}
        with open(qrel_path, 'r') as f:
            for line in f:
                # print(line)
                parts = line.strip().split(" ")
                query_id, doc_id, relevance = parts[0], parts[2], int(parts[3])
                
                if query_id not in qrel_dict:
                    qrel_dict[query_id] = {}
                qrel_dict[query_id][doc_id] = relevance
        
        return qrel_dict
    
    def compute_system_effectiveness(self, run_file, qrel_dict):
        """
        Compute system effectiveness using Average Precision
        
        Args:
        - run_file: Path to system's ranking file
        - qrel_dict: Relevance judgments dictionary
        
        Returns:
        Dictionary of system effectiveness per query
        """
        system_effectiveness = {}
        
        with open(run_file, 'r') as f:
            current_query = None
            ranked_docs = []
            
            for line in f:
                parts = line.strip().split()
                query_id, doc_id = parts[0], parts[1]
                
                if current_query is None:
                    current_query = query_id
                
                if query_id != current_query:
                    # Compute AP for previous query
                    system_effectiveness[current_query] = self._average_precision(ranked_docs, qrel_dict.get(current_query, {}))
                    
                    # Reset for new query
                    current_query = query_id
                    ranked_docs = []
                
                ranked_docs.append(doc_id)
            
            # Compute for last query
            if current_query:
                system_effectiveness[current_query] = self._average_precision(ranked_docs, qrel_dict.get(current_query, {}))
        
        return system_effectiveness
    
    def _average_precision(self, ranked_docs, query_relevance):
        """
        Compute Average Precision for a single query
        """
        ap = 0.0
        relevant_found = 0
        
        for rank, doc in enumerate(ranked_docs, 1):
            if doc in query_relevance and query_relevance[doc] > 0:
                relevant_found += 1
                ap += relevant_found / rank
        
        return ap / (len(query_relevance) or 1)
    
    def compare_rankings(self, run_files):
        """
        Compare system rankings using human and self qrels
        
        Args:
        - run_files: List of system run file paths
        
        Returns:
        DataFrame with Kendall Tau correlations
        """
        human_effectiveness = {}
        self_effectiveness = {}
        
        # Compute effectiveness for each run file
        for run_file in run_files:
            system_name = os.path.basename(run_file)  # Extract system name from filename
            
            human_eff = self.compute_system_effectiveness(run_file, self.human_qrel)
            self_eff = self.compute_system_effectiveness(run_file, self.self_qrel)
            
            human_effectiveness[system_name] = human_eff
            self_effectiveness[system_name] = self_eff
        
        # Compute Kendall Tau for each system
        kendall_results = []
        
        for system_name in human_effectiveness.keys():
            human_scores = list(human_effectiveness[system_name].values())
            self_scores = list(self_effectiveness[system_name].values())
            
            # Compute Kendall Tau
            tau, p_value = stats.kendalltau(human_scores, self_scores)
            
            kendall_results.append({
                'System': system_name,
                'Kendall Tau': tau,
                'P-value': p_value
            })
        
        return pd.DataFrame(kendall_results)

def validate_file(parser, arg):
    """
    Validate that file exists and is readable
    """
    if not os.path.exists(arg):
        parser.error(f"The file {arg} does not exist!")
    if not os.access(arg, os.R_OK):
        parser.error(f"The file {arg} is not readable!")
    return arg

def main():
    # Create the parser
    parser = argparse.ArgumentParser(
        description='Compare system rankings using Kendall Tau correlation',
        epilog='Example: python script.py --human-qrel human.qrel --self-qrel self.qrel --run-files run1.txt run2.txt'
    )
    
    # Add arguments
    parser.add_argument('--human-qrel', 
                        type=lambda x: validate_file(parser, x),
                        required=True,
                        help='Path to human-annotated qrel file')
    
    parser.add_argument('--self-qrel', 
                        type=lambda x: validate_file(parser, x),
                        required=True,
                        help='Path to self-made qrel file')
    
    parser.add_argument('--run-files-directory', 
                        required=True,
                        help='Path to all system run files')
    
    parser.add_argument('--output', 
                        type=str, 
                        default='kendall_results.csv',
                        help='Output CSV file for Kendall Tau results (default: kendall_results.csv)')
    
    parser.add_argument('--verbose', 
                        action='store_true',
                        help='Print detailed results to console')

    # Parse arguments
    args = parser.parse_args()

    # Create comparator
    try:
        comparator = QrelComparator(
            human_qrel_path=args.human_qrel,
            self_qrel_path=args.self_qrel
        )

        # Compare rankings

        # List all files in the directory
        run_files = [os.path.join(args.run_files_directory, file) for file in os.listdir(args.run_files_directory) if os.path.isfile(os.path.join(args.run_files_directory, file))]

        results = comparator.compare_rankings(run_files)

        # Save results
        results.to_csv(args.output, index=False)

        # Verbose output
        if args.verbose:
            print("Kendall Tau Correlation Results:")
            print(results)
            print(f"\nResults saved to {args.output}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()