import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from collections import Counter
from scipy.stats import chi2_contingency, fisher_exact

def create_dic_run_metrics(path):
    with open(path,"r") as f:
        data = f.readlines()
        d = dict()
        for line in data[2:]:
            print(line)
            _,met,_,val = line.split(",")
            d[met] = float(val)
        return d
        
        
path = "./analysis/analysis_dev_baseline_prompt order: 0123_full.csv"
metrics_run_A = create_dic_run_metrics(path)
path = "./analysis/analysis_dev_baseline_prompt order: 3210_full.csv"
metrics_run_B = create_dic_run_metrics(path)
# for metric_name in metrics_run_B.keys():
#     # Extract metric values for comparison (assuming you're interested in MAP for this example)
#     metric_values_A = metrics_run_A[metric_name]
#     metric_values_B = metrics_run_B[metric_name]

#     # Perform paired t-test
#     t_statistic, p_value = ttest_rel(metric_values_A, metric_values_B)

#     # Output results
#     print(f"P-value for paired t-test comparing {metric_name} between Run A and Run B: {p_value}")

    # Perform paired t-test
t_statistic, p_value = ttest_rel(list(metrics_run_A.values()), list(metrics_run_B.values()))

# Output results
print(f"P-value for paired t-test comparing between Run A and Run B: {p_value}")

def count_scores(data):
    counts = Counter()
    for line in data:
        parts = line.split()
        score = int(parts[-1])
        counts[score] += 1
    return counts


with open("./results/dev_baseline_prompt order: 0123_full.run","r") as f:
    data = f.readlines()
    print("order 0-1-2-3")
    counts_3_2_1_0=count_scores(data)
    print(counts_3_2_1_0)

with open("./results/dev_baseline_prompt order: 3210_full.run","r") as f:
    data = f.readlines()
    print("order 3-2-1-0")
    counts_0_1_2_3=count_scores(data)
    print(counts_0_1_2_3)

# Create a contingency table
contingency_table = [
    [counts_3_2_1_0[0], counts_3_2_1_0[3]],
    [counts_0_1_2_3[0], counts_0_1_2_3[3]]
]

# Perform the chi-squared test
chi2, p, _, _ = chi2_contingency(contingency_table)

# Alternatively, perform Fisher's exact test if counts are low
_, p_fisher = fisher_exact(contingency_table)

print("Chi-squared test p-value:", p)
print("Fisher's exact test p-value:", p_fisher)
    