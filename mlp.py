import pandas as pd
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer

# Load the data
dev_path = "./data/llm4eval_dev_qrel_2024.txt"
dev_df = pd.read_csv(dev_path, sep=" ", header=None, names=['qid', 'Q0', 'docid','rel_score'])
path_for_decomposed_scores = "./decomposed_scores/dev_decomposed_relavance_qrel.json"
with open(path_for_decomposed_scores, "r") as f:
    dictionary = json.load(f)


# Function to create DataFrame from decomposed scores
def make_df_of_decomposed_scores(df):
    rows = []
    for i in range(len(df)):
        try:
            sample = str((df.loc[i].qid, df.loc[i].docid))
            label = df.loc[i].rel_score
            values = list(dictionary[sample].values())
            row = [sample, int(values[0]), int(values[1]), int(values[2]), int(values[3]), label]
        except KeyError:
            row = [sample, 0, 0, 0, 0, label]
        rows.append(row)
    
    new_df = pd.DataFrame(rows, columns=['qid_docid', 'Exactness', 'Topicality', 'Depth', 'Contextual Fit', 'rel_label'])
    return new_df

# Create DataFrame from decomposed scores
train_df = make_df_of_decomposed_scores(dev_df)

# Split features and labels
X = train_df.iloc[:, 1:-1]
y = train_df.iloc[:, -1]

# Define the models
models = {
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

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
    
    results[model_name] = kappa_scores

# Print average Cohen's kappa and standard deviation for each model
for model_name, kappa_scores in results.items():
    print(f'{model_name} Cohen\'s kappa: {np.mean(kappa_scores):.2f} ± {np.std(kappa_scores):.2f}')

# Train each model on the entire dataset and print Cohen's kappa on the entire dataset
for model_name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    kappa = cohen_kappa_score(y, y_pred)
    print(f'{model_name} Cohen\'s kappa on entire dataset: {kappa:.2f}')



# # Separate data based on relevance scores
# df_0 = dev_df[dev_df['rel_score'] == 0]
# df_1 = dev_df[dev_df['rel_score'] == 1]
# df_2 = dev_df[dev_df['rel_score'] == 2]
# df_3 = dev_df[dev_df['rel_score'] == 3]



# # Determine the desired sample size for training (adjust as needed)
# sample_size_train = 1000  # For example, select 500 instances per relevance score for training
# print(len(df_0))
# print(len(df_1))
# print(len(df_2))
# print(len(df_3))

# # Sample instances from each relevance score group for training
# df_0_sampled = df_0.sample(n=int(sample_size_train*(len(df_0)/len(dev_df))), random_state=2)
# df_1_sampled = df_1.sample(n=int(sample_size_train*(len(df_1)/len(dev_df))), random_state=2)
# df_2_sampled = df_2.sample(n=int(sample_size_train*(len(df_2)/len(dev_df))), random_state=2)
# df_3_sampled = df_3.sample(n=int(sample_size_train*(len(df_3)/len(dev_df))), random_state=2)

# # Concatenate the sampled dataframes for training
# train_df = pd.concat([df_0_sampled, df_1_sampled, df_2_sampled, df_3_sampled])

# # Create a test set with unbalanced data
# test_df = dev_df.drop(train_df.index)

# print(len(df_0_sampled))
# print(len(df_1_sampled))
# print(len(df_2_sampled))
# print(len(df_3_sampled))

# train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
# test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
# Print the size of each split to verify balance
# print("Train set sizes:")
# print(train_df['rel_score'].value_counts())
# print(train_df.head())
# print("\nTest set sizes:")
# print(test_df['rel_score'].value_counts())


path_for_decomposed_scores = "./decomposed_scores/dev_decomposed_relavance_qrel.json"
with open(path_for_decomposed_scores,"r") as f:
    dictionary = json.load(f)
# print(dictionary)
def make_df_of_decomposed_scores(df):
    rows = []
    for i in range(len(df)):
        try:
            sample = str((df.loc[i].qid,df.loc[i].docid))
            label = df.loc[i].rel_score
            # print(sample)
            values = list(dictionary[sample].values())
            row = [sample,int(values[0]),int(values[1]),int(values[2]),int(values[3]),label]
        except:
            # print(sample)
            row = [sample,0,0,0,0,label]
        rows.append(row)
           
    new_df = pd.DataFrame(rows,columns=['qid_docid','Exactness','Topicality','Depth','Contextual Fit','rel_label'])
    # print(new_df.head())
    return new_df

new_train_df = make_df_of_decomposed_scores(train_df)
new_test_df = make_df_of_decomposed_scores(test_df)
X_train, y_train = new_train_df.iloc[:,1:-1], new_train_df.iloc[:,-1]
X_test, y_test = new_test_df.iloc[:,1:-1], new_test_df.iloc[:,-1]


# # Initialize the model
# log_reg = LogisticRegression(multi_class='ovr', max_iter=1000)

# # Define the parameter grid
# param_grid = {
#     'C': [0.1, 1, 10, 100]
# }

# # Initialize Grid Search
# grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)

# # Get the best parameters and best model
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_

# # Make predictions with the best model
# y_pred = best_model.predict(X_test)

# # Evaluate the best model
# print("Logistic Regression Classification Report:")
# print(classification_report(y_test, y_pred))



# import xgboost as xgb
# from sklearn.metrics import classification_report

# # Initialize the model
# xgboost = xgb.XGBClassifier(random_state=42)

# # Define the parameter grid
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0]
# }

# # Initialize Grid Search
# grid_search = GridSearchCV(estimator=xgboost, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)

# # Get the best parameters and best model
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_

# # Make predictions with the best model
# y_pred = best_model.predict(X_test)

# # Evaluate the best model
# print("XGBoost Classification Report:")
# print(classification_report(y_test, y_pred))



##############LR#########

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
}
log_reg = LogisticRegression(multi_class='ovr', max_iter=800, class_weight='balanced')

# Initialize Grid Search
grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred = best_model.predict(X_test)

# Evaluate the best model
print("Best Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred))
