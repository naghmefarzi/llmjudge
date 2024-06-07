import argparse
from collections import defaultdict
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TREC Analysis")
    parser.add_argument("truth_file", type=str, help="Path to the ground truth TREC file")
    parser.add_argument("experiment_file", type=str, help="Path to the experiment TREC file")
    parser.add_argument("analysis_file", type=str, help="Path to the experiment TREC file")
    
    args = parser.parse_args()
    
    main(args.truth_file, args.experiment_file, args.analysis_file)
