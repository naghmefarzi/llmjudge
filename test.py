# from scipy.stats import ttest_rel, wilcoxon
# from collections import Counter
# from scipy.stats import chi2_contingency, fisher_exact


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_dic_run_metrics(path):
    with open(path,"r") as f:
        data = f.readlines()
        d = dict()
        for line in data[2:]:
            print(line)
            _,met,_,val = line.split(",")
            d[met] = float(val)
        return d
        
        
# path = "./analysis/analysis_dev_baseline_prompt order: 0123_full.csv"
# metrics_run_A = create_dic_run_metrics(path)
# path = "./analysis/analysis_dev_baseline_prompt order: 3210_full.csv"
# metrics_run_B = create_dic_run_metrics(path)
# # for metric_name in metrics_run_B.keys():
# #     # Extract metric values for comparison (assuming you're interested in MAP for this example)
# #     metric_values_A = metrics_run_A[metric_name]
# #     metric_values_B = metrics_run_B[metric_name]

# #     # Perform paired t-test
# #     t_statistic, p_value = ttest_rel(metric_values_A, metric_values_B)

# #     # Output results
# #     print(f"P-value for paired t-test comparing {metric_name} between Run A and Run B: {p_value}")

#     # Perform paired t-test
# t_statistic, p_value = ttest_rel(list(metrics_run_A.values()), list(metrics_run_B.values()))

# # Output results
# print(f"P-value for paired t-test comparing between Run A and Run B: {p_value}")

# def count_scores(data):
#     counts = Counter()
#     for line in data:
#         parts = line.split()
#         score = int(parts[-1])
#         counts[score] += 1
#     return counts


# with open("./results/dev_baseline_prompt order: 0123_full.run","r") as f:
#     data = f.readlines()
#     print("order 0-1-2-3")
#     counts_3_2_1_0=count_scores(data)
#     print(counts_3_2_1_0)

# with open("./results/dev_baseline_prompt order: 3210_full.run","r") as f:
#     data = f.readlines()
#     print("order 3-2-1-0")
#     counts_0_1_2_3=count_scores(data)
#     print(counts_0_1_2_3)

# # Create a contingency table
# contingency_table = [
#     [counts_3_2_1_0[0], counts_3_2_1_0[3]],
#     [counts_0_1_2_3[0], counts_0_1_2_3[3]]
# ]

# # Perform the chi-squared test
# chi2, p, _, _ = chi2_contingency(contingency_table)

# # Alternatively, perform Fisher's exact test if counts are low
# _, p_fisher = fisher_exact(contingency_table)

# print("Chi-squared test p-value:", p)
# print("Fisher's exact test p-value:", p_fisher)
    
    
    
    
dev_path = "./data/llm4eval_dev_qrel_2024.txt"
dev_df = pd.read_csv(dev_path, sep=" ", header=None, names=['qid', 'Q0', 'docid','rel_score'])
score_counts = dev_df['rel_score'].value_counts().sort_index()

# Display the counts
print("Relevance Score Counts:")
print(score_counts)



# Separate data based on relevance scores
df_0 = dev_df[dev_df['rel_score'] == 0]
df_1 = dev_df[dev_df['rel_score'] == 1]
df_2 = dev_df[dev_df['rel_score'] == 2]
df_3 = dev_df[dev_df['rel_score'] == 3]

# Sample an equal number of instances from each relevance score group
sample_size = min(len(df_0), len(df_1), len(df_2), len(df_3))

df_0_sampled = df_0.sample(n=sample_size, random_state=42)
df_1_sampled = df_1.sample(n=sample_size, random_state=42)
df_2_sampled = df_2.sample(n=sample_size, random_state=42)
df_3_sampled = df_3.sample(n=sample_size, random_state=42)

# Concatenate the sampled dataframes
balanced_df = pd.concat([df_0_sampled, df_1_sampled, df_2_sampled, df_3_sampled])

# Split into train and test sets
train_df, test_df = train_test_split(balanced_df, test_size=0.2, random_state=42)

# Print the size of each split to verify balance
print("Train set sizes:")
print(train_df['rel_score'].value_counts())
print("\nTest set sizes:")
print(test_df['rel_score'].value_counts())