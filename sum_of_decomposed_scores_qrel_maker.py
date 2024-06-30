import pandas as pd
import json

dev_path = "./data/llm4eval_test_qrel_2024.txt"
df = pd.read_csv(dev_path, sep=" ", header=None, names=['qid', 'Q0', 'docid','rel_score'])

path_for_decomposed_scores = "./decomposed_scores/test_decomposed_relavance_qrel.json"
with open(path_for_decomposed_scores,"r") as f:
    dictionary = json.load(f)
    
result_decomposed_sum_path = "./results/test_sum_of_decomposed_prompts.txt" 
with open(result_decomposed_sum_path, "w") as result_file:
    
    for i in range(len(df)):
            try:
                sample = str((df.loc[i].qid,df.loc[i].docid))
                label = df.loc[i].rel_score
                values = list(dictionary[sample].values())
                exactness, topicality, depth, cfit = int(values[0]),int(values[1]),int(values[2]),int(values[3])
                overal_sum = exactness + topicality + depth + cfit
                if 10<=overal_sum <=12:
                    overal_score = 3
                elif 7<=overal_sum <=9:
                    overal_score = 2
                if 5<=overal_sum <=6:
                    overal_score = 1
                if 0<=overal_sum <=5: 
                    overal_score = 0    
                    
                result_file.write(f"{df.loc[i].qid} 0 {df.loc[i].docid} {overal_score}\n")
                
            except:
                # print(sample)
                row = [sample,0,0,0,0,label]
                
                result_file.write(f"{df.loc[i].qid} 0 {df.loc[i].docid} 0\n")
                