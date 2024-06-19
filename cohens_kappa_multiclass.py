import pandas as pd
from sklearn.metrics import cohen_kappa_score
import argparse
def ck_analysis(qrel1_path,qrel2_path,result_path):

    qrel1_df = pd.read_csv(qrel1_path, sep=' ', header=None, names=['qid', 'unused_col', 'docid', 'label_1'])
    qrel2_df = pd.read_csv(qrel2_path, sep=' ', header=None, names=['qid', 'unused_col', 'docid', 'label_2'])

    # Ensure the QREL files are sorted or merged correctly if necessary
    # Assuming both files have 'qid' and 'docid' columns to merge on
    merged = pd.merge(qrel1_df, qrel2_df, on=['qid', 'docid'], suffixes=('_1', '_2'))
    print(merged.head(5))
    # Extract labels
    labels1 = merged['label_1']  
    labels2 = merged['label_2']  

    # Calculate Cohen's kappa score for multiclass classification
    kappa = cohen_kappa_score(labels1, labels2)

    print(f"Cohen's kappa score: {kappa}")
    with open(result_path,"w") as f :
        f.write(f"first qrel file is: {qrel1_path}\n")
        f.write(f"second qrel file is: {qrel2_path}\n")
        f.write(f"Cohen's kappa score: {kappa}")
        
        

def main():
    parser = argparse.ArgumentParser(description="TREC Analysis")
    parser.add_argument("truth_qrel_file_path", type=str, help="Path to the ground truth TREC file")
    parser.add_argument("experiment_qrel_file_path", type=str, help="Path to the experiment TREC file")
    parser.add_argument("analysis_file", type=str, help="Path to results")
    
    args = parser.parse_args()
    ck_analysis(args.truth_qrel_file_path, args.experiment_qrel_file_path, args.analysis_file )
    
    
if __name__=="__main__":
    main()