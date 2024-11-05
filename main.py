import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from matplotlib.ticker import PercentFormatter
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Read raw dataset
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
os.chdir(script_dir)
df_normal_raw = pd.read_csv("Data_of_rotary_machine_defects/data_9_1.csv")
df_normal_raw["failure"] = 0
df_misaligned_raw = pd.read_csv("Data_of_rotary_machine_defects/data_9_2.csv")
df_misaligned_raw["failure"] = 1
df_raw = pd.concat([df_normal_raw, df_misaligned_raw])
df_raw.info()
#

# Count positive and negative samples
df_Y_count = df_raw["failure"].value_counts().reset_index()
df_Y_count["failure"] = ["No failure", "Misaligned bearings"]
sns.set_theme(palette='tab10',
              font='Calibri',
              font_scale=1.2,
              rc=None)
plt.figure(figsize=(4, 4))
plt.pie(df_Y_count["count"], labels=df_Y_count["failure"], autopct='%1.2f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title("Distribution of Failure Types")
plt.axis("equal")
plt.show()
#
# Data counts of each failure type are balanced. No need to adjust and balance the data count while training a ML model. #

# Check distributions and outliers of each feature
Xs = df_raw.columns.tolist()
Xs.remove("failure")
for x in Xs:
    sns.displot(data=df_raw, x=x, kde=True, bins=100, color="red", facecolor="lime", height=5, aspect=3)
#

# Study correlation between features
for i in range(len(Xs)):
    for j in range(i+1, len(Xs)):
        plt.figure(figsize=(18, 7))
        sns.scatterplot(data=df_raw, x=Xs[i], y=Xs[j], hue="failure", palette="tab10", alpha=0.5)
        plt.show()
#
# A positive relationship has been found between Y-fluctuations of left and right bearings, which separates positive and negative samples significantly. #

# Heatmap
plt.figure(figsize=(10, 5))
df_corr = df_raw.corr()
sns.heatmap(df_corr, annot=True, fmt=".2f")
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()
#
# Both temperature-related features (attr_9 & attr_10) have less correlations with Y. #
# There is a strong correlation between attr_1 and attr_5, and their distributions are similar. #

# Feature selection
to_drop = df_corr[abs(df_corr["failure"])<0.2].index
print(to_drop)
df_filtered = df_raw.drop(columns=to_drop) # Drop features with low correlations with Y
df_filtered = df_filtered.drop(columns=["Fluctuations_X_in_the_left_bearing"]) # Drop features with strong correlations and similar distributions with one another
#

# Standardize features
Y = df_filtered["failure"].to_list()
df_X_filtered = df_filtered.drop(columns=["failure"])
standard_scaler = StandardScaler()
df_X_standardized = standard_scaler.fit_transform(df_X_filtered)
df_X_standardized = pd.DataFrame(df_X_standardized, columns=df_X_filtered.columns)
#

# Find best component count for PCA
pca = PCA()
df_X_reduced = pca.fit_transform(df_X_standardized)
explained_ratio = pca.explained_variance_ratio_
cum_explained_ratio = np.cumsum(explained_ratio)
sns.barplot(
    x=np.arange(1, len(explained_ratio)+1),
    y=explained_ratio)
plt.plot(
    np.arange(0, len(cum_explained_ratio)),
    cum_explained_ratio,
    color="orange",
    marker="o"
)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=10)
plt.xticks(fontsize=10)
plt.xlabel("Component Count", fontsize=11)
plt.ylabel("Explained Variance Ratio (%)", fontsize=11)
plt.axhline(y=0.9, color="red", linestyle="-", linewidth=1)
plt.axvline(x=2.5, color="gray", linestyle="--", linewidth=1)
plt.show()
for i in range(len(cum_explained_ratio)):
    if cum_explained_ratio[i] >= 0.9:
        n_component = i + 1
        break
#

# PCA
pca = PCA(n_components=n_component)
df_X_reduced = pca.fit_transform(df_X_standardized)
df_X_reduced = pd.DataFrame(df_X_reduced)
df_X_reduced.columns = df_X_reduced.columns.astype(str)
df_reduced = df_X_reduced
df_reduced["failure"] = Y
#

# K=5-fold
indices = np.arange(0, df_reduced.shape[0])
np.random.seed(5269)
np.random.shuffle(indices)
split_indices = np.array_split(indices, 5)
#

def BuildSet(k, split_indices=split_indices):
    training_indices = list()
    test_indices = list()
    for i, group in enumerate(split_indices):
        if i == k:
            test_indices.extend(group)
        else:
            training_indices.extend(group)
    df_training_reduced = df_reduced.loc[training_indices]
    df_test_reduced = df_reduced.loc[test_indices]

    df_training_Y = df_training_reduced["failure"]
    df_training_X_reduced = df_training_reduced.drop(columns="failure")
    df_test_Y = df_test_reduced["failure"]
    df_test_X_reduced = df_test_reduced.drop(columns="failure")

    return df_training_X_reduced, df_training_Y, df_test_X_reduced, df_test_Y

# DT
accuracies = list()
for k in range(len(split_indices)):
    df_training_X_reduced, df_training_Y, df_test_X_reduced, df_test_Y = BuildSet(k=k)
    dt_clf = DecisionTreeClassifier(random_state=5269)
    dt_clf.fit(df_training_X_reduced, df_training_Y)
    Y_predicted = dt_clf.predict(df_test_X_reduced)
    accuracy = accuracy_score(df_test_Y, Y_predicted)
    accuracies.append(accuracy)
print(np.mean(accuracies))
#

# RF
accuracies = list()
precisions = list()
recalls = list()
f1_scores = list()
for k in range(len(split_indices)):
    df_training_X_reduced, df_training_Y, df_test_X_reduced, df_test_Y = BuildSet(k=k)
    rf_clf = RandomForestClassifier(random_state=5269, n_estimators=200, max_depth=30) # from GridSearchCV
    rf_clf.fit(df_training_X_reduced, df_training_Y)
    Y_predicted = rf_clf.predict(df_test_X_reduced)

    accuracies.append(accuracy_score(df_test_Y, Y_predicted))
    precisions.append(precision_score(df_test_Y, Y_predicted))
    recalls.append(recall_score(df_test_Y, Y_predicted))
    f1_scores.append(f1_score(df_test_Y, Y_predicted))
result = pd.DataFrame(
    data=[["Accuracy", np.mean(accuracies), np.std(accuracies)],
        ["Precision", np.mean(precisions), np.std(precisions)],
        ["Recall", np.mean(recalls), np.std(recalls)],
        ["F1-Score", np.mean(f1_scores), np.std(f1_scores)]],
    columns=["Metric", "Mean", "Std"])
print(result)

# SVM
accuracies = list()
for k in range(len(split_indices)):
    df_training_X_reduced, df_training_Y, df_test_X_reduced, df_test_Y = BuildSet(k=k)
    svm_clf = SVC(random_state=5269)
    svm_clf.fit(df_training_X_reduced, df_training_Y)
    Y_predicted = svm_clf.predict(df_test_X_reduced)
    accuracy = accuracy_score(df_test_Y, Y_predicted)
    accuracies.append(accuracy)
print(np.mean(accuracies))
#