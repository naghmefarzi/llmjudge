import os
import glob
import pytrec_eval
import pandas as pd
from scipy.stats import kendalltau

# Load Qrels from a file
def load_qrels(qrel_path):
    qrels = {}
    with open(qrel_path, 'r') as f:
        for line in f:
            query_id, _, doc_id, relevance = line.strip().split()
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = int(float(relevance))
    # print(qrels)
    return qrels

# Load system runs from a file
def load_run(run_path):
    runs = {}
    with open(run_path, 'r') as f:
        for line in f:
            query_id, _, doc_id, rank, score, _ = line.strip().split()
            if query_id not in runs:
                runs[query_id] = {}
            runs[query_id][doc_id] = float(score)
    # print(runs)
    return runs

# Evaluate metrics for a single system run
def evaluate_system(qrels, runs):
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {'map', 'recip_rank', 'ndcg_cut_10', 'ndcg_cut_20', 'ndcg_cut_100'}
    )
    results = evaluator.evaluate(runs)

    # Compute Mean Metrics
    total_queries = len(results)
    metrics = {
        'AP': sum(results[q]['map'] for q in results) / total_queries,
        'RR': sum(results[q]['recip_rank'] for q in results) / total_queries,
        'NDCG@10': sum(results[q]['ndcg_cut_10'] for q in results) / total_queries,
        'NDCG@20': sum(results[q]['ndcg_cut_20'] for q in results) / total_queries,
        'NDCG@100': sum(results[q]['ndcg_cut_100'] for q in results) / total_queries,
    }
    return metrics

# Main function to evaluate and rank systems
def rank_systems(qrel_path, run_dir):
    qrels = load_qrels(qrel_path)
    run_files =[os.path.join(run_dir, run_name) for run_name in os.listdir(run_dir)]
    # print(run_files)
    leaderboard = []

    for run_file in run_files:
        run_name = os.path.basename(run_file)
        # print(run_file)
        runs = load_run(run_file)
        metrics = evaluate_system(qrels, runs)
        metrics['System'] = run_name
        leaderboard.append(metrics)

    # Sort leaderboard by AP (descending order)
    leaderboard = sorted(leaderboard, key=lambda x: x['NDCG@10'], reverse=True)

    # Convert leaderboard to pandas DataFrame for better formatting
    df = pd.DataFrame(leaderboard)
    return df

# Paths
qrel_path = "./private_data/my_qrels/llm4eval_test_qrel_2024_withRel.txt"  # Path to qrel file
run_dir = "./runs_TRECDL_2023/passages-runs/"          # Directory containing system run files

# Generate leaderboard
leaderboard = rank_systems(qrel_path, run_dir)
official_DL23_Leaderboard = {}
for i,system in enumerate(leaderboard["System"]):
    official_DL23_Leaderboard[system]= i+1

# Print leaderboard
# print(leaderboard.to_string(index=False))

# Save leaderboard to CSV
leaderboard.to_csv("./private_data/leaderboard/official_trecDL2023_leaderboard.csv", index=False)
# print(official_DL23_Leaderboard)




# Official leaderboard ranking (as provided in your code)
official_rankings = list(official_DL23_Leaderboard.values())
# print(official_rankings)
# Artificial leaderboard ranking based on the artificial evaluation
artificial_qrels_path = "./private_data/my_qrels"
artificial_qrel_files_paths = [os.path.join(artificial_qrels_path,qrel_name) for qrel_name in os.listdir(artificial_qrels_path) if "test" in qrel_name]
for artifitial_qrel_path in artificial_qrel_files_paths:
    artificial_leaderboard = rank_systems(artifitial_qrel_path, run_dir)
    # print(artifici al_leaderboard)
    artificial_Leaderboard_dic = {}
    for i,system in enumerate(artificial_leaderboard["System"]):
        artificial_Leaderboard_dic[system]= official_DL23_Leaderboard[system]
    artificial_rankings = list(artificial_Leaderboard_dic.values())
    # print(artificial_rankings)
    # Remove systems that are missing from the artificial leaderboard
    filtered_systems = [(official, artificial) for official, artificial in zip(official_rankings, artificial_rankings) if artificial is not None]

    # Separate the official and artificial rankings for Kendall Tau calculation
    official_filtered = [pair[0] for pair in filtered_systems]
    artificial_filtered = [pair[1] for pair in filtered_systems]

    # Calculate the Kendall Tau correlation
    tau, _ = kendalltau(official_filtered, artificial_filtered)

    # Output the result
    print(f"{os.path.basename(artifitial_qrel_path)}: Kendall Tau correlation: {tau:.4f}")
    
    
    
def evaluate_rankings(qrel_path, run_dir, official_leaderboard=None, multiple_qrels=False):
    # For single qrel evaluation or for making official leaderboard
    def get_single_ranking(qrel_path):
        leaderboard = rank_systems(qrel_path, run_dir)
        rankings_dict = {}
        for i, system in enumerate(leaderboard["System"]):
            
            rankings_dict[system.replace(".run","")] = official_leaderboard.get(system.replace(".run","")) if official_leaderboard else i+1
        return rankings_dict, leaderboard
    
    results = {}
    
    if not multiple_qrels:
        # Single qrel evaluation
        rankings_dict, leaderboard = get_single_ranking(qrel_path)
        results[os.path.basename(qrel_path)] = {
            'rankings': rankings_dict,
            'leaderboard': leaderboard,
            'tau': None  # Will be calculated if official_leaderboard provided
        }
    else:
        # Multiple qrels evaluation
        qrel_files = [os.path.join(qrel_path, f) for f in os.listdir(qrel_path) if "test" in f]
        for qrel_file in qrel_files:
            rankings_dict, leaderboard = get_single_ranking(qrel_file)
            results[os.path.basename(qrel_file)] = {
                'rankings': rankings_dict,
                'leaderboard': leaderboard,
                'tau': None
            }
    
    # Calculate Kendall's Tau if official leaderboard provided
    if official_leaderboard:
        official_rankings = list(official_leaderboard.values())
        for qrel_name, result in results.items():
            artificial_rankings = list(result['rankings'].values())
            filtered_systems = [(o, a) for o, a in zip(official_rankings, artificial_rankings) if a is not None]
            official_filtered = [pair[0] for pair in filtered_systems]
            artificial_filtered = [pair[1] for pair in filtered_systems]
            tau, _ = kendalltau(official_filtered, artificial_filtered)
            results[qrel_name]['tau'] = tau
            print(f"{qrel_path} tau is: {tau:.4f}")
            
    return results

from official_leaderboards import official_DL19_Leaderboard, official_DL20_Leaderboard
qrel_path = "./results/4_prompts_dl2019.txt"
run_dir = "./trec-dl-2019/runs"
res = evaluate_rankings(qrel_path, run_dir, official_leaderboard=official_DL19_Leaderboard, multiple_qrels=False)
# print(res)


print("***")
qrel_path = "./results/4_prompts_dl2020.txt"
run_dir = "./trec-dl-2020/runs"
res = evaluate_rankings(qrel_path, run_dir, official_leaderboard=official_DL20_Leaderboard, multiple_qrels=False)
# print(res)
