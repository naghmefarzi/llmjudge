import argparse
from collections import defaultdict
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from trectools import TrecQrel, TrecRun, TrecEval
import pandas as pd

def read_trec_file(file_path):
    """
    Reads a TREC file and returns a dictionary mapping qid to a list of (pid, score) tuples.
    """
    data = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            qid, _, pid, score = parts
            data[qid].append((pid, int(score)))
    return data


def trectoolanalysis(truth_file,experiment_file, analysis_file):

    r1 = TrecRun(experiment_file)
    print(r1.topics()[:5]) # Shows the first 5 topics

    qrels = TrecQrel(truth_file)

    te = TrecEval(r1, qrels)
    rbp, residuals = te.get_rbp()           # RBP: 0.474, Residuals: 0.001
    p100 = te.get_precision(depth=100)     # P@100: 0.186
    
    # Check if documents retrieved by the system were judged:
    cover10 = r1.get_mean_coverage(qrels, topX=10)   # 9.99
    cover1000 = r1.get_mean_coverage(qrels, topX=1000) # 481.390 
    # Evaluates r1 using all implemented evaluation metrics
    result_r1_per_q = te.evaluate_all( per_query=True)
    result_r1 = te.evaluate_all( per_query=False)
    
    # print(result_r1.data)
    # On average for system 'input.aplrob03a' participating in robust03, 480 documents out of 1000 were judged.
    with open(analysis_file,"a") as output_file:
        output_file.write("Average number of documents judged among top 10: %.2f, among top 1000: %.2f \n" % (cover10, cover1000))
        output_file.write(f"p@100: {p100}\n")
        result_r1_per_q.data.to_csv(analysis_file.replace(".txt","_per_query.csv"))
        result_r1.data.to_csv(analysis_file.replace(".txt",".csv"))
        
        

def interannotation_agreement(truth_data, experiment_data):
    """
    Computes interannotation agreement for mutual qid and passageids.
    """
    agreements = []
    common_qids = set(truth_data.keys()) & set(experiment_data.keys())
    for qid in common_qids:
        truth_scores = {pid: score for pid, score in truth_data[qid]}
        experiment_scores = {pid: score for pid, score in experiment_data[qid]}
        common_pids = set(truth_scores.keys()) & set(experiment_scores.keys())
        for pid in common_pids:
            agreements.append((truth_scores[pid], experiment_scores[pid]))
    return agreements


def cf_matrix_figure(truth_scores, experiment_scores, analysis_file):
    # Compute confusion matrix
    cm = confusion_matrix(truth_scores, experiment_scores)
    # print(cm)
    # Create a heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

    # Add labels, title, and adjust axis
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    class_names = ['Score 0', 'Score 1', 'Score 2', 'Score 3']
    plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(analysis_file.replace(".txt", "_confusion_matrix.png"))
    plt.show()
    

def main(truth_file, experiment_file, analysis_file):
    # Read TREC files
    truth_data = read_trec_file(truth_file)
    experiment_data = read_trec_file(experiment_file)

    # Compute interannotation agreement
    agreements = interannotation_agreement(truth_data, experiment_data)
    truth_scores, experiment_scores = zip(*agreements)

    # Calculate Cohen's kappa
    kappa = cohen_kappa_score(truth_scores, experiment_scores)
    
    # Write results to a file
    with open(analysis_file, "w") as output_file:
        output_file.write("Interannotation Agreement:\n")
        output_file.write(f"Total agreements: {len(agreements)}\n")
        output_file.write(f"Cohen's Kappa: {kappa}\n\n")
        
    cf_matrix_figure(truth_scores, experiment_scores, analysis_file)
    trectoolanalysis(truth_file,experiment_file, analysis_file)
    



official_DL23_Leaderboard:Dict[str,str] = {'naverloo-rgpt4': 1,
                                           'naverloo-frgpt4': 2,
                                           'naverloo_fs_RR_duo': 3,
                                           'cip_run_1': 4,
                                           'cip_run_2': 5,
                                           'cip_run_3': 6,
                                           'cip_run_6': 7,
                                           'cip_run_4': 8,
                                           'cip_run_5': 9,
                                           'naverloo_fs_RR': 10,
                                           'cip_run_7': 11,
                                           'naverloo_bm25_splades_RR': 12,
                                           'uogtr_qr_be_gb': 13,
                                           'uogtr_b_grf_e_gb': 14,
                                           'uogtr_se_gb': 15,
                                           'uogtr_be_gb': 16,
                                           'uogtr_se': 17,
                                           'naverloo_bm25_RR': 18,
                                           'uogtr_qr_be': 19,
                                           'uogtr_b_grf_e': 20,
                                           'naverloo_fs': 21,
                                           'uogtr_be': 22,
                                           'slim-pp-0shot-uw': 23,
                                           'splade_pp_self_distil': 24,
                                           'splade_pp_ensemble_distil': 25,
                                           'bm25_splades': 26,
                                           'uogtr_s': 27,
                                           'agg-cocondenser': 28,
                                           'uot-yahoo_rankgpt4': 29,
                                           'WatS-LLM-Rerank': 30,
                                           'uot-yahoo_rankgpt35': 31,
                                           'WatS-Augmented-BM25': 32,
                                           'uot-yahoo_LLMs-blender': 33,
                                           'uogtr_dph': 34,
                                           'uogtr_dph_bo1': 35}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TREC Analysis")
    parser.add_argument("truth_file", type=str, help="Path to the ground truth TREC file")
    parser.add_argument("experiment_file", type=str, help="Path to the experiment TREC file")
    parser.add_argument("analysis_file", type=str, help="Path to the experiment TREC file")
    
    args = parser.parse_args()
    
    main(args.truth_file, args.experiment_file, args.analysis_file)
