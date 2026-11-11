"""
Random forest classification of DOM molecular formulas into three phase-distribution
classes (common, sediment, water) under different undersampling input sizes.

For each combination of input sizes (molecules common x molecules sediment, with
molecules water fixed at 1000), the training set is randomly undersampled 10 times.
In each iteration, hyperparameters are tuned by grid search with stratified, shuffled
10-fold cross-validation (scoring: macro-averaged F1), and the best model is evaluated
on the fixed, held-out test set.

Usage:
    python rf_undersampling_gridsearch.py --site XXH   # Three Gorges Reservoir (TGR)
    python rf_undersampling_gridsearch.py --site PYH   # Poyang Lake (PL)

Class-wise output columns: resistant = common, labile = sediment, product = water.
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--site", required=True, choices=["XXH", "PYH"],
                    help="XXH = Three Gorges Reservoir (TGR), PYH = Poyang Lake (PL)")
parser.add_argument("--data-dir", default="data")
parser.add_argument("--out-dir", default="results")
parser.add_argument("--n-jobs", type=int, default=-1)
args = parser.parse_args()

site = args.site
out_dir = os.path.join(args.out_dir, site)
os.makedirs(os.path.join(out_dir, "scoring_f1_macro"), exist_ok=True)

# Load data: 10 molecular features; the last column is the class label
soft_data = pd.read_csv(os.path.join(args.data_dir, f"{site}2.csv"))
data = soft_data[['H', 'O', 'DBE', 'MZ', 'OC', 'HC', 'NC', 'SC', 'NOSC', 'DBE.O']]
target = soft_data.iloc[:, -1]

# 80:20 train/test split; the test set is never resampled
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Random forest classifier
clf1 = RandomForestClassifier(random_state=2)

# Number of molecules in each class of the training set
print("common_count_max:", y_train.value_counts()['common'])
print("sediment_count_max:", y_train.value_counts()['sediment'])
print("water_count_max:", y_train.value_counts()['water'])

# Input sizes for undersampling
Common_counts = [500, 1000, 1500, 2000, 2500]   # molecules common
Sediment_counts = [600, 900, 1200, 1500, 1800]       # molecules sediment
Water_count = 1000                               # molecules water (fixed)

# Hyperparameter search space
param_grid1 = {
    "n_estimators": np.arange(100, 1000, step=100),
    "max_features": [1.0, "sqrt"],
    "max_depth": list(np.arange(10, 100, step=10)) + [None],
}

start = time.time()
for Common_count in Common_counts:
    for Sediment_count in Sediment_counts:
        # Random undersampler
        rus = RandomUnderSampler(sampling_strategy={'common': Common_count,
                                                    'sediment': Sediment_count,
                                                    'water': Water_count})

        # Grid search with stratified, shuffled 10-fold cross-validation
        grid_search = GridSearchCV(clf1, param_grid1,
                                   cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                                   verbose=1, n_jobs=args.n_jobs, scoring="f1_macro")

        # Scores of each undersampling iteration
        f1_scores, precision_scores, recall_scores, auc_scores, accuracy_scores = [], [], [], [], []
        f1_macro_scores, f1_micro_scores, f1_weighted_scores = [], [], []
        precision_macro_scores, precision_micro_scores, precision_weighted_scores = [], [], []
        recall_macro_scores, recall_micro_scores, recall_weighted_scores = [], [], []
        auc_macro_scores, auc_micro_scores, auc_weighted_scores = [], [], []
        accuracy_scores2 = []
        best_params_list = []

        # Repeated random undersampling (10 iterations, seeds 0-9)
        for i in range(10):
            rus.set_params(random_state=i)
            X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
            grid_search.fit(X_resampled, y_resampled)   # best model is refitted on X_resampled
            best_model = grid_search.best_estimator_
            best_params_list.append(grid_search.best_params_)
            predicts = best_model.predict(X_test)
            proba = best_model.predict_proba(X_test)    # class probabilities for ROC AUC

            # Class-wise metrics (classes in alphabetical order: common, sediment, water)
            f1 = f1_score(y_test, predicts, average=None)
            precision = precision_score(y_test, predicts, average=None)
            recall = recall_score(y_test, predicts, average=None)
            accuracy = accuracy_score(y_test, predicts)

            # Averaged metrics
            f1_macro = f1_score(y_test, predicts, average='macro')
            f1_micro = f1_score(y_test, predicts, average='micro')
            f1_weighted = f1_score(y_test, predicts, average='weighted')
            precision_macro = precision_score(y_test, predicts, average='macro')
            precision_micro = precision_score(y_test, predicts, average='micro')
            precision_weighted = precision_score(y_test, predicts, average='weighted')
            recall_macro = recall_score(y_test, predicts, average='macro')
            recall_micro = recall_score(y_test, predicts, average='micro')
            recall_weighted = recall_score(y_test, predicts, average='weighted')

            # One-vs-rest ROC AUC from predicted class probabilities
            y_test_bin = label_binarize(y_test, classes=best_model.classes_)
            auc_score = roc_auc_score(y_test_bin, proba, multi_class='ovr', average=None)
            auc_score_macro = roc_auc_score(y_test_bin, proba, multi_class='ovr', average='macro')
            auc_score_micro = roc_auc_score(y_test_bin, proba, multi_class='ovr', average='micro')
            auc_score_weighted = roc_auc_score(y_test_bin, proba, multi_class='ovr', average='weighted')

            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)
            auc_scores.append(auc_score)
            accuracy_scores.append([accuracy, accuracy, accuracy])
            accuracy_scores2.append(accuracy)
            f1_macro_scores.append(f1_macro)
            f1_micro_scores.append(f1_micro)
            f1_weighted_scores.append(f1_weighted)
            precision_macro_scores.append(precision_macro)
            precision_micro_scores.append(precision_micro)
            precision_weighted_scores.append(precision_weighted)
            recall_macro_scores.append(recall_macro)
            recall_micro_scores.append(recall_micro)
            recall_weighted_scores.append(recall_weighted)
            auc_macro_scores.append(auc_score_macro)
            auc_micro_scores.append(auc_score_micro)
            auc_weighted_scores.append(auc_score_weighted)

        print("Average F1 score:", np.mean(f1_scores))
        print("Average precision score:", np.mean(precision_scores))
        print("Average recall score:", np.mean(recall_scores))
        print("Average AUC score:", np.mean(auc_scores))

        # Class-wise scores (one row per iteration)
        all_scores = np.hstack([f1_scores, precision_scores, recall_scores, auc_scores, accuracy_scores])
        scores_df = pd.DataFrame(all_scores, columns=[
            'f1_score_resistant', 'f1_score_labile', 'f1_score_product',
            'precision_resistant', 'precision_labile', 'precision_product',
            'recall_resistant', 'recall_labile', 'recall_product',
            'auc_resistant', 'auc_labile', 'auc_product',
            'accuracy1', 'accuracy2', 'accuracy3'])

        # Averaged scores (one row per iteration)
        all_scores_bulk = np.column_stack([
            f1_macro_scores, f1_micro_scores, f1_weighted_scores,
            precision_macro_scores, precision_micro_scores, precision_weighted_scores,
            recall_macro_scores, recall_micro_scores, recall_weighted_scores,
            auc_macro_scores, auc_micro_scores, auc_weighted_scores, accuracy_scores2])
        scores_df_bulk = pd.DataFrame(all_scores_bulk, columns=[
            'f1_macro_scores', 'f1_micro_scores', 'f1_weighted_scores',
            'precision_macro_scores', 'precision_micro_scores', 'precision_weighted_scores',
            'recall_macro_scores', 'recall_micro_scores', 'recall_weighted_scores',
            'auc_macro_scores', 'auc_micro_scores', 'auc_weighted_scores', 'accuracy_scores2'])

        filename = f"{site}_Common_{Common_count}_Sediment_{Sediment_count}_Water_{Water_count}_f1_macro_R1.csv"
        scores_df.to_csv(os.path.join(out_dir, filename), index=False)
        scores_df_bulk.to_csv(os.path.join(out_dir, "scoring_f1_macro", "bulk_" + filename), index=False)
        print(pd.Series(y_resampled).value_counts())

        # Best hyperparameters of each iteration
        with open(os.path.join(out_dir, filename.replace('.csv', '.txt')), 'w') as f:
            f.write("Best parameters: " + str(grid_search.best_params_) + "\n")
            f.write("Training features: " + str(list(data.columns)) + "\n")
            for k, p in enumerate(best_params_list, 1):
                f.write("Iteration {} best parameters: {}\n".format(k, p))

print("Run time: %.2f min" % ((time.time() - start) / 60))
