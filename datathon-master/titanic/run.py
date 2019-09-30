# +
# Imports

import time
import pandas as pd
import statistics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# +
# Read dataframes

print("\nReading dataframe...")
df = pd.read_csv("train.csv")
print(df.shape)
df

# +
# Feature engineering

pkey = "PassengerId"
response = "Survived"
categoricals = [
    "Pclass", # Ticket class
    "Name",
    "Sex",
    #"Cabin",
    #"Ticket",
    "Embarked",
    "Surname",
    "Title",
    "TicketHasLetters",
    "CabinClass"
]
numerics = [
    "Age",
    "SibSp", # of siblings / spouses aboard the Titanic
    "Parch", # of parents / children aboard the Titanic
    "Fare",
]

def extract_surname(name):
    return name.split(",")[0]

def extract_title(name):
    return name.split(",")[1].split(".")[0]

def extract_ticket_has_letters(ticket):
    return not ticket.isnumeric()

def extract_cabin_class(cabin):
    cabin_letters = ["A", "B", "C", "D", "E", "F"]
    cabin_class = "N/A"
    for c in cabin_letters:
        if c in cabin:
            cabin_class = c
    return cabin_class

def encode_categoricals(df, categoricals):
    for c in categoricals:
        df[c] = df[c].astype("category").cat.codes
    return df

def inpute_median(df, numerics):
    df[numerics] = df[numerics].fillna(df.median())
    return df

def engineer_features(df):
    df["Surname"] = df["Name"].apply(lambda name: extract_surname(name))
    df["Title"] = df["Name"].apply(lambda name: extract_title(name))
    df["TicketHasLetters"] = df["Ticket"].apply(lambda ticket: extract_ticket_has_letters(ticket))
    df["Cabin"] = df["Cabin"].fillna("0")
    df["CabinClass"] = df["Cabin"].apply(lambda cabin: extract_cabin_class(cabin))
    df = encode_categoricals(df, categoricals)
    df = inpute_median(df, numerics)
    return df

print("\nEngineering features...")
df = engineer_features(df)
print(df.shape)
df


# +
def split(df, seed, split_ratio=0.2):
    df_train, df_test = train_test_split(df, test_size=split_ratio, random_state=seed)
    return df_train, df_test

def extract_features(df):
    features = categoricals + numerics
    X = df[features].values
    return X

def extract_response(df):
    y = df[response].values
    return y

def train_random_forest(X, y, seed):
    model = RandomForestClassifier(
        n_estimators=1000, 
        max_depth=5,    
        random_state=seed
    )
    model.fit(X, y)
    return model

def train_gbm(X, y, seed):
    model = GradientBoostingClassifier(
        learning_rate=0.1, 
        n_estimators=1000, 
        max_depth=5,
        subsample=0.8,
        random_state=seed
    )
    model.fit(X, y)
    return model
    
def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    y_score = y_pred_proba[:,1]
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    return(acc, auc)
    
def feature_importance(model, columns):
    importances = model.feature_importances_
    df_importance = pd.DataFrame({
        "feature": columns,
        "importance": importances
    })
    df_importance = df_importance.sort_values(by="importance", ascending=False)
    return df_importance

def calculate_average_feature_importance(importances, n_bootstraps):
    importances_sorted = [imp.sort_values("feature") for imp in importances]
    values = [0 for _ in range(len(importances_sorted[0]))]
    for imp in importances_sorted:
        imp_vals = imp["importance"].values.tolist()
        for j, val in enumerate(imp_vals):
            values[j] = values[j] + val
    values = [v/n_bootstraps for v in values]
    ave_importance = importances_sorted[0]
    ave_importance.importance = values
    ave_importance = ave_importance.sort_values("importance", ascending=False).reset_index()
    ave_importance = ave_importance[["feature", "importance"]] 
    return ave_importance

def print_progress_bar(iteration, total, prefix="", suffix="", length=30, fill="=", head=">", track="."):
    filled_length = int(length * iteration // total)
    if filled_length == 0:
        bar = track * length
    elif filled_length == 1:
        bar = head + track * (length - 1)
    elif filled_length == length:
        bar = fill * filled_length
    else:
        bar = fill * (filled_length-1) + ">" + "." * (length-filled_length)
    print("\r" + prefix + "[" + bar + "] " + str(iteration) + "/" + str(total), suffix, end = "\r")
    if iteration == total: 
        print()

def bootstrap(df, train_function, n_bootstraps):
    print("\nBootstrapping", train_function.__name__, n_bootstraps, "times...")
    start = time.time()
    accs, aucs, importances = [], [], []
    for i, seed in enumerate(range(n_bootstraps)):
        df_train, df_test = split(df, seed)
        X_train = extract_features(df_train)
        y_train = extract_response(df_train)
        X_test = extract_features(df_test)
        y_test = extract_response(df_test)
        model = train_function(X_train, y_train, seed)
        acc, auc = evaluate_model(model, X_test, y_test)
        importance = feature_importance(model, categoricals + numerics)
        accs.append(acc)
        aucs.append(auc)
        importances.append(importance)
        print_progress_bar(i+1, n_bootstraps)
    acc_mean = statistics.mean(accs)
    acc_stdev = statistics.stdev(accs)
    print("\nacc: mean=" + str(acc_mean), "stdev=" + str(acc_stdev))
    auc_mean = statistics.mean(aucs)
    auc_stdev = statistics.stdev(aucs)
    print("auc: mean=" + str(auc_mean), "stdev=" + str(auc_stdev))
    ave_importance = calculate_average_feature_importance(importances, n_bootstraps)
    print(" ")
    print(ave_importance)
    best_index = [i for i, auc in enumerate(aucs) if auc == max(aucs)][0]
    print("\nRun time:", int(time.time() - start), "seconds")
    return best_index
        
bootstrap(df, train_random_forest, 500)
best_index = bootstrap(df, train_gbm, 500)
# + {}
def score_test(best_index):
    df_test = pd.read_csv("test.csv")
    df_test = engineer_features(df_test)
    X_test = extract_features(df_test)
    X = extract_features(df)
    y = extract_response(df)
    model = train_gbm(X, y, best_index)
    survived = model.predict(X_test)
    passengers = df_test["PassengerId"].values
    submission = pd.DataFrame({
        "PassengerId": passengers,
        "Survived": survived
    })
    return submission
    
submission = score_test(best_index)
submission.to_csv("submission.csv", index=False)
submission
# -

