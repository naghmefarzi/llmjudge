import pandas as pd
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier





# Function to create DataFrame from decomposed scores
def make_dev_df(dev_path,path_for_decomposed_scores):
    df = pd.read_csv(dev_path, sep=" ", header=None, names=['qid', 'Q0', 'docid','rel_score'])
    with open(path_for_decomposed_scores, "r") as f:
        dictionary = json.load(f)
    rows = []
    for i in range(len(df)):
        try:
            sample = str((df.loc[i].qid, df.loc[i].docid))
            # print(sample)
            label = df.loc[i].rel_score
            values = list(dictionary[sample].values())
            row = [sample, int(values[0]), int(values[1]), int(values[2]), int(values[3]), label]
        except:
            row = [sample, 0, 0, 0, 0, label]
        rows.append(row)
    
    new_df = pd.DataFrame(rows, columns=['qid_docid', 'Exactness', 'Topicality', 'Depth', 'Contextual Fit', 'rel_label'])
    return new_df
def make_test_df(test_path,path_for_decomposed_scores):
    df = pd.read_csv(test_path, sep=" ", header=None, names=['qid', 'Q0', 'docid'])
    with open(path_for_decomposed_scores, "r") as f:
        dictionary = json.load(f)  
    rows = []
    for i in range(len(df)):
        try:
            sample = str((df.loc[i].qid, df.loc[i].docid))
            # print(sample)
            values = list(dictionary[sample].values())
            row = [sample, int(values[0]), int(values[1]), int(values[2]), int(values[3])]
        except:
            row = [sample, 0, 0, 0, 0]
        rows.append(row)
    
    new_df = pd.DataFrame(rows, columns=['qid_docid', 'Exactness', 'Topicality', 'Depth', 'Contextual Fit'])
    return new_df




# Create DataFrame from decomposed scores
# Load the data
dev_path = "./data/llm4eval_dev_qrel_2024.txt"
path_for_decomposed_scores = "./decomposed_scores/dev_decomposed_relavance_qrel.json"
train_df = make_dev_df(dev_path,path_for_decomposed_scores)

# Split features and labels
X = train_df.iloc[:, 1:-1]
y = train_df.iloc[:, -1]

# Define the models
models = {
    # 'Random Forest': RandomForestClassifier(),
    # 'Decision Tree': DecisionTreeClassifier(),
    # 'Logistic Regression': LogisticRegression(max_iter=2000),
    # 'SVM': SVC(),
    # 'Gradient Boosting': GradientBoostingClassifier(),
    # 'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    # 'Neural Network': MLPClassifier(max_iter=1500),
    # 'AdaBoost': AdaBoostClassifier(),
    # 'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}






test_path = "./data/llm4eval_test_qrel_2024.txt"
path_for_decomposed_scores = "./decomposed_scores/test_decomposed_relavance_qrel.json"
test_df = make_test_df(test_path,path_for_decomposed_scores)
X_TEST = test_df.iloc[:,1:]
# print(X_TEST)


# Perform stratified cross-validation and evaluate Cohen's kappa
skf = StratifiedKFold(n_splits=5)
results = {}

for model_name, model in models.items():
    kappa_scores = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        kappa = cohen_kappa_score(y_test, y_pred)
        kappa_scores.append(kappa)
        Y_PRED = model.predict(X_TEST)
    
    results[model_name] = kappa_scores

# Print average Cohen's kappa and standard deviation for each model
for model_name, kappa_scores in results.items():
    print(f'{model_name} Cohen\'s kappa: {np.mean(kappa_scores):.3f} ± {np.std(kappa_scores):.2f}')

# Train each model on the entire dataset and print Cohen's kappa on the entire dataset
# for model_name, model in models.items():
#     model.fit(X, y)
#     y_pred = model.predict(X)
    # kappa = cohen_kappa_score(y, y_pred)
    # print(f'{model_name} Cohen\'s kappa on entire dataset: {kappa:.2f}')
    




test_df = pd.concat([test_df,pd.DataFrame(Y_PRED,columns=["predicted_rel_score"])],axis=1)
# print(test_df)
# with open("./results/test_NaiveB_on_decomposed.txt","w") as f:
#     for row_idx in range(len(test_df)):
        
#         qid, docid = test_df.loc[row_idx].qid_docid.replace("(","").replace("'","").replace(")","").replace(" ","").split(",")
#         score = test_df.loc[row_idx].predicted_rel_score
#         # print(qid)
#         f.write(f"{qid} 0 {docid} {score}\n")