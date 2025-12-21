
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns  # Optional: for heatmap styling

# Load the data
data = pd.read_csv("data.csv")

# Define window size
window_size = 15

# Initialize lists to store windowed features and labels
windowed_features = []
windowed_labels = []

# Segment the time-series data into windows
for i in range(0, len(data) - window_size + 1, window_size):
    window = data.iloc[i:i + window_size]
    if len(window) == window_size:  # Ensure the window is complete
        # Extract features from the window
        window_features = {
            'mean_acc_x': window['acc_x'].mean(),
            'mean_acc_y': window['acc_y'].mean(),
            'mean_acc_z': window['acc_z'].mean(),
            'std_acc_x': window['acc_x'].std(),
            'std_acc_y': window['acc_y'].std(),
            'std_acc_z': window['acc_z'].std(),
            'skew_acc_x': window['acc_x'].skew(),
            'skew_acc_y': window['acc_y'].skew(),
            'skew_acc_z': window['acc_z'].skew(),
            'kurt_acc_x': window['acc_x'].kurt(),
            'kurt_acc_y': window['acc_y'].kurt(),
            'kurt_acc_z': window['acc_z'].kurt()
        }
        windowed_features.append(window_features)

        # Assign label to the window (assuming it's the same for all samples within the window)
        window_label = window['behavior'].iloc[0]  # Adjust based on your specific label
        windowed_labels.append(window_label)

# Convert lists to DataFrame
X_windowed = pd.DataFrame(windowed_features)
y_windowed = np.array(windowed_labels)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_windowed)

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Deal with class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_windowed)

# Split the data into train and test using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Initialize a list to store accuracy results
accuracies = []

# Perform 10 iterations
for i in range(1):
    # Instantiate Random Forest classifier with random subspace
    rf_classifier = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)

    # Train the Random Forest classifier
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred_rf = rf_classifier.predict(X_test)

    # Evaluate the Random Forest model's performance
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    f1 = f1_score(y_test, y_pred_rf, average='weighted')
    precision = precision_score(y_test, y_pred_rf, average='weighted')
    recall = recall_score(y_test, y_pred_rf, average='weighted')

    # Gini score calculation
    def gini_score(y_true, y_pred):
        true_positive_rate = recall_score(y_true, y_pred, average='weighted')
        false_positive_rate = 1 - precision_score(y_true, y_pred, average='weighted')
        return 2 * (true_positive_rate - false_positive_rate)

    gini = gini_score(y_test, y_pred_rf)

    accuracies.append(accuracy_rf)
    print(f"Accuracy on test data (Random Forest with 0.5-second window size) for iteration {i + 1}: {accuracy_rf * 100:.2f}%")
    print(f"F1 Score: {f1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Gini Score: {gini:.2f}")

# Print average accuracy over all iterations
print(f"Average Accuracy over 10 iterations: {np.mean(accuracies) * 100:.2f}%")

# Plot confusion matrix for the last iteration
conf_matrix = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Get the unique labels and their corresponding encoded values
unique_labels, label_encoding = np.unique(y_windowed, return_inverse=True)

# Print the unique labels with their corresponding numbers
for i, label in enumerate(unique_labels):
    print(f"Encoded label {i} corresponds to behavior: {label}")

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, make_scorer
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# -------------------------
# 1) Load & window the data
# -------------------------
data = pd.read_csv("data.csv")

window_size = 15
windowed_features = []
windowed_labels = []

for i in range(0, len(data) - window_size + 1, window_size):
    window = data.iloc[i:i + window_size]
    if len(window) == window_size:
        feats = {
            'mean_acc_x': window['acc_x'].mean(),
            'mean_acc_y': window['acc_y'].mean(),
            'mean_acc_z': window['acc_z'].mean(),
            'std_acc_x': window['acc_x'].std(),
            'std_acc_y': window['acc_y'].std(),
            'std_acc_z': window['acc_z'].std(),
            'skew_acc_x': window['acc_x'].skew(),
            'skew_acc_y': window['acc_y'].skew(),
            'skew_acc_z': window['acc_z'].skew(),
            'kurt_acc_x': window['acc_x'].kurt(),
            'kurt_acc_y': window['acc_y'].kurt(),
            'kurt_acc_z': window['acc_z'].kurt()
        }
        windowed_features.append(feats)
        windowed_labels.append(window['behavior'].iloc[0])  # adjust if you want majority label in window

X = pd.DataFrame(windowed_features)
y = np.array(windowed_labels)

# ------------------------------------------
# 2) Define pipeline: Impute -> Scale -> SMOTE -> RF
#    SMOTE is inside the CV via imblearn.Pipeline
# ------------------------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42, k_neighbors=2)),
    ('rf', rf)
])

# ------------------------------------------------------
# 3) Custom "gini-like" scorer to mirror your prior code
#    (based on weighted recall/precision as you had it)
# ------------------------------------------------------
def gini_like(y_true, y_pred):
    tpr = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    fpr = 1 - precision_score(y_true, y_pred, average='weighted', zero_division=0)
    return 2 * (tpr - fpr)

gini_scorer = make_scorer(gini_like)

# -----------------------------------------
# 4) 5-fold stratified cross-validation
# -----------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'accuracy': 'accuracy',
    'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
    'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
    'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0),
    'gini_like': gini_scorer
}

cvres = cross_validate(
    pipe,
    X, y,
    cv=skf,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False
)

# -----------------------------------------
# 5) Out-of-fold predictions for confusion matrix
# -----------------------------------------
y_pred_oof = cross_val_predict(pipe, X, y, cv=skf, n_jobs=-1)
labels = np.unique(y)
cm = confusion_matrix(y, y_pred_oof, labels=labels)

# -----------------------------------------
# 6) Print results
# -----------------------------------------
def mean_std(name):
    vals = cvres[f'test_{name}']
    return f"{vals.mean():.4f} ± {vals.std():.4f}"

print("5-Fold CV Results (mean ± std)")
print(f"Accuracy:            {mean_std('accuracy')}")
print(f"F1 (weighted):       {mean_std('f1_weighted')}")
print(f"Precision (weighted):{mean_std('precision_weighted')}")
print(f"Recall (weighted):   {mean_std('recall_weighted')}")
print(f"Gini-like:           {mean_std('gini_like')}")

print("\nOut-of-fold classification report:")
print(classification_report(y, y_pred_oof, digits=3, zero_division=0))

print("Labels order in confusion matrix:", labels.tolist())
print("Confusion matrix (rows=true, cols=pred):\n", cm)

import pandas as pd
import numpy as np
from collections import Counter

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt

# -------------------------
# 0) Load raw data with individual IDs
# -------------------------
data = pd.read_csv("data.csv")  # must contain: date,time,cow_num,acc_x,acc_y,acc_z,behavior

# Basic sanity checks
required = {"cow_num", "acc_x", "acc_y", "acc_z", "behavior"}
missing = required - set(data.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# -------------------------
# 1) Window per-cow to avoid cross-individual leakage
#    (non-overlapping windows of length 15 samples)
# -------------------------
window_size = 15
X_windows, y_windows, cow_windows = [], [], []

for cow_id, df_cow in data.groupby("cow_num"):
    df_cow = df_cow.reset_index(drop=True)
    # Non-overlapping windows (step = window_size)
    for start in range(0, len(df_cow) - window_size + 1, window_size):
        window = df_cow.iloc[start:start + window_size]

        # Feature extraction (same style as your script)
        feats = {
            'mean_acc_x': window['acc_x'].mean(),
            'mean_acc_y': window['acc_y'].mean(),
            'mean_acc_z': window['acc_z'].mean(),
            'std_acc_x':  window['acc_x'].std(),
            'std_acc_y':  window['acc_y'].std(),
            'std_acc_z':  window['acc_z'].std(),
            'skew_acc_x': window['acc_x'].skew(),
            'skew_acc_y': window['acc_y'].skew(),
            'skew_acc_z': window['acc_z'].skew(),
            'kurt_acc_x': window['acc_x'].kurt(),
            'kurt_acc_y': window['acc_y'].kurt(),
            'kurt_acc_z': window['acc_z'].kurt(),
        }
        X_windows.append(feats)

        # Label for the window:
        # (A) first-sample label (matches your code)
        label = window['behavior'].iloc[0]
        # (B) If you prefer majority vote, uncomment the next 2 lines:
        # label = window['behavior'].mode()
        # label = label.iloc[0] if not label.empty else window['behavior'].iloc[0]

        y_windows.append(label)
        cow_windows.append(cow_id)

X = pd.DataFrame(X_windows)
y = np.asarray(y_windows)
cows = np.asarray(cow_windows)

# -------------------------
# 2) LOIO CV: hold out one cow at a time
# -------------------------
unique_cows = np.unique(cows)

# Containers for results
per_fold = []
overall_cm = None
labels_sorted = np.sort(np.unique(y))

for held_out in unique_cows:
    test_mask = (cows == held_out)
    train_mask = ~test_mask

    X_train_raw, X_test_raw = X.iloc[train_mask], X.iloc[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # --- Impute and scale (fit on TRAIN only) ---
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp  = imputer.transform(X_test_raw)

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled  = scaler.transform(X_test_imp)

    # --- Handle imbalance with SMOTE on TRAIN only ---
    # Determine min class count to set a safe k_neighbors
    class_counts = Counter(y_train)
    min_count = min(class_counts.values())
    # SMOTE requires k_neighbors <= min_count - 1, and >=1 to be valid
    k_neighbors = max(1, min(5, min_count - 1))
    # If any class is singleton, SMOTE cannot work; fallback to no SMOTE
    use_smote = (min_count >= 2)

    if use_smote:
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    else:
        X_train_bal, y_train_bal = X_train_scaled, y_train

    # --- Train model ---
    rf = RandomForestClassifier(
        n_estimators=100,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_bal, y_train_bal)

    # --- Evaluate ---
    y_pred = rf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    precw = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recw = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    per_fold.append({
        "held_out_cow": held_out,
        "n_test_windows": int(test_mask.sum()),
        "accuracy": acc,
        "f1_weighted": f1w,
        "precision_weighted": precw,
        "recall_weighted": recw,
        "used_smote": use_smote,
        "smote_k_neighbors": k_neighbors if use_smote else None
    })

    # Accumulate confusion matrices (aligned to sorted labels)
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    overall_cm = cm if overall_cm is None else overall_cm + cm

# -------------------------
# 3) Report results
# -------------------------
res_df = pd.DataFrame(per_fold).sort_values("held_out_cow").reset_index(drop=True)
print("\n=== LOIO per-cow results ===")
print(res_df)

print("\n=== LOIO mean ± std (across cows) ===")
for metric in ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]:
    m = res_df[metric].mean()
    s = res_df[metric].std(ddof=1) if len(res_df) > 1 else 0.0
    print(f"{metric}: {m:.4f} ± {s:.4f}")

# Optional: normalized confusion matrix
if overall_cm is not None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(overall_cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels_sorted)),
        yticks=np.arange(len(labels_sorted)),
        xticklabels=labels_sorted,
        yticklabels=labels_sorted,
        ylabel='True label',
        xlabel='Predicted label',
        title='Aggregated Confusion Matrix (LOIO)'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Annotate counts
    for i in range(overall_cm.shape[0]):
        for j in range(overall_cm.shape[1]):
            ax.text(j, i, int(overall_cm[i, j]), ha="center", va="center", color="w" if overall_cm[i,j] > overall_cm.max()/2 else "black")
    plt.tight_layout()
    plt.show()

# Optional: print class-wise precision/recall/F1 averaged over all folds by concatenating predictions
# (Requires storing all y_test and y_pred each fold if you want a single classification_report)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, make_scorer
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# -------------------------
# 1) Load & window the data
# -------------------------
file_names = ["cow1.csv", "cow2.csv", "cow3.csv", "cow4.csv", "cow5.csv", "cow6.csv"]

window_size = 15
windowed_features = []
windowed_labels = []

def standardize_columns(df):
    # Map possible headers to standard names
    colmap = {}
    for c in df.columns:
        c_str = c.strip()
        low = c_str.lower()
        if low in ("accx", "accx [g]"): colmap[c] = "AccX"
        elif low in ("accy", "accy [g]"): colmap[c] = "AccY"
        elif low in ("accz", "accz [g]"): colmap[c] = "AccZ"
        elif low in ("label", "label [-]", "behaviour", "behavior"): colmap[c] = "Label"
    return df.rename(columns=colmap)

for idx, file_name in enumerate(file_names, start=1):
    df = pd.read_csv(file_name)
    df = standardize_columns(df)

    required = {"AccX", "AccY", "AccZ", "Label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{file_name} missing columns: {missing}. Found: {list(df.columns)}")

    df = df.reset_index(drop=True)
    # Non-overlapping windows
    for start in range(0, len(df) - window_size + 1, window_size):
        w = df.iloc[start:start + window_size]

        feats = {
            'mean_acc_x': w['AccX'].mean(),
            'mean_acc_y': w['AccY'].mean(),
            'mean_acc_z': w['AccZ'].mean(),
            'std_acc_x':  w['AccX'].std(),
            'std_acc_y':  w['AccY'].std(),
            'std_acc_z':  w['AccZ'].std(),
            'skew_acc_x': w['AccX'].skew(),
            'skew_acc_y': w['AccY'].skew(),
            'skew_acc_z': w['AccZ'].skew(),
            'kurt_acc_x': w['AccX'].kurt(),
            'kurt_acc_y': w['AccY'].kurt(),
            'kurt_acc_z': w['AccZ'].kurt(),
        }
        windowed_features.append(feats)
        windowed_labels.append(w['Label'].iloc[0])  # adjust if you want majority label in window

X = pd.DataFrame(windowed_features)
y = np.array(windowed_labels)

# ------------------------------------------
# 2) Define pipeline: Impute -> Scale -> SMOTE -> RF
#    SMOTE is inside the CV via imblearn.Pipeline
# ------------------------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42, k_neighbors=2)),
    ('rf', rf)
])

# ------------------------------------------------------
# 3) Custom "gini-like" scorer to mirror your prior code
#    (based on weighted recall/precision as you had it)
# ------------------------------------------------------
def gini_like(y_true, y_pred):
    tpr = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    fpr = 1 - precision_score(y_true, y_pred, average='weighted', zero_division=0)
    return 2 * (tpr - fpr)

gini_scorer = make_scorer(gini_like)

# -----------------------------------------
# 4) 5-fold stratified cross-validation
# -----------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'accuracy': 'accuracy',
    'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
    'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
    'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0),
    'gini_like': gini_scorer
}

cvres = cross_validate(
    pipe,
    X, y,
    cv=skf,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False
)

# -----------------------------------------
# 5) Out-of-fold predictions for confusion matrix
# -----------------------------------------
y_pred_oof = cross_val_predict(pipe, X, y, cv=skf, n_jobs=-1)
labels = np.unique(y)
cm = confusion_matrix(y, y_pred_oof, labels=labels)

# -----------------------------------------
# 6) Print results
# -----------------------------------------
def mean_std(name):
    vals = cvres[f'test_{name}']
    return f"{vals.mean():.4f} ± {vals.std():.4f}"

print("5-Fold CV Results (mean ± std)")
print(f"Accuracy:            {mean_std('accuracy')}")
print(f"F1 (weighted):       {mean_std('f1_weighted')}")
print(f"Precision (weighted):{mean_std('precision_weighted')}")
print(f"Recall (weighted):   {mean_std('recall_weighted')}")
print(f"Gini-like:           {mean_std('gini_like')}")

print("\nOut-of-fold classification report:")
print(classification_report(y, y_pred_oof, digits=3, zero_division=0))

print("Labels order in confusion matrix:", labels.tolist())
print("Confusion matrix (rows=true, cols=pred):\n", cm)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns  # Optional: for heatmap styling

# Define file names
file_names = ["cow1.csv", "cow2.csv", "cow3.csv", "cow4.csv", "cow5.csv", "cow6.csv"]

# Initialize an empty list to store DataFrames
all_data = []

# Iterate over each file and read them
for file_name in file_names:
    data = pd.read_csv(file_name)
    all_data.append(data)

# Concatenate all the data into one DataFrame
combined_data = pd.concat(all_data, ignore_index=True)

# Define window size
window_size = 15

# Initialize lists to store windowed features and labels
windowed_features = []
windowed_labels = []

# Segment the time-series data into windows
for i in range(0, len(combined_data) - window_size + 1, window_size):
    window = combined_data.iloc[i:i + window_size]
    if len(window) == window_size:  # Ensure the window is complete
        # Extract features from the window
        window_features = {
            'mean_acc_x': window['AccX'].mean(),
            'mean_acc_y': window['AccY'].mean(),
            'mean_acc_z': window['AccZ'].mean(),
            'std_acc_x': window['AccX'].std(),
            'std_acc_y': window['AccY'].std(),
            'std_acc_z': window['AccZ'].std(),
            'skew_acc_x': window['AccX'].skew(),
            'skew_acc_y': window['AccY'].skew(),
            'skew_acc_z': window['AccZ'].skew(),
            'kurt_acc_x': window['AccX'].kurt(),
            'kurt_acc_y': window['AccY'].kurt(),
            'kurt_acc_z': window['AccZ'].kurt()
        }
        windowed_features.append(window_features)

        # Assign label to the window (assuming it's the same for all samples within the window)
        window_label = window['Label'].iloc[0]
        windowed_labels.append(window_label)

# Convert lists to DataFrame
X_windowed = pd.DataFrame(windowed_features)
y_windowed = np.array(windowed_labels)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_windowed)

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Deal with class imbalance using SMOTE with a reduced number of neighbors
smote = SMOTE(random_state=42, k_neighbors=2)  # Reduce k_neighbors to 2
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_windowed)

# Split the data into train and test using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Initialize a list to store accuracy results
accuracies = []

# Gini score calculation using AUC-ROC
def gini_score(y_true, y_pred_proba):
    # Calculate AUC-ROC score
    auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
    # Gini coefficient is derived from AUC
    return 2 * auc - 1

# Perform 1 iteration
for i in range(1):
    # Instantiate Random Forest classifier with random subspace
    rf_classifier = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)

    # Train the Random Forest classifier
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred_rf = rf_classifier.predict(X_test)
    y_pred_proba_rf = rf_classifier.predict_proba(X_test)  # Get predicted probabilities

    # Evaluate the Random Forest model's performance
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    f1 = f1_score(y_test, y_pred_rf, average='weighted')
    precision = precision_score(y_test, y_pred_rf, average='weighted')
    recall = recall_score(y_test, y_pred_rf, average='weighted')

    # Calculate Gini score using predicted probabilities
    gini = gini_score(y_test, y_pred_proba_rf)

    accuracies.append(accuracy_rf)
    print(f"Accuracy on test data (Random Forest with 0.5-second window size) for iteration {i + 1}: {accuracy_rf * 100:.2f}%")
    print(f"F1 Score: {f1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Gini Score: {gini:.2f}")

import pandas as pd
import numpy as np
from collections import Counter
import math

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

# -------------------------
# 0) File list (one cow per file)
# -------------------------
file_names = ["cow1.csv", "cow2.csv", "cow3.csv", "cow4.csv", "cow5.csv", "cow6.csv"]

# -------------------------
# 1) Window per-cow (NO cross-file windows)
#    Non-overlapping windows of 15 samples
# -------------------------
window_size = 15
X_feats, y_labels, cow_ids = [], [], []

def standardize_columns(df):
    # Map possible headers to standard names
    colmap = {}
    for c in df.columns:
        c_str = c.strip()
        low = c_str.lower()
        if low in ("accx", "accx [g]"): colmap[c] = "AccX"
        elif low in ("accy", "accy [g]"): colmap[c] = "AccY"
        elif low in ("accz", "accz [g]"): colmap[c] = "AccZ"
        elif low in ("label", "label [-]", "behaviour", "behavior"): colmap[c] = "Label"
    return df.rename(columns=colmap)

for idx, file_name in enumerate(file_names, start=1):
    df = pd.read_csv(file_name)
    df = standardize_columns(df)

    required = {"AccX", "AccY", "AccZ", "Label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{file_name} missing columns: {missing}. Found: {list(df.columns)}")

    df = df.reset_index(drop=True)
    # Non-overlapping windows
    for start in range(0, len(df) - window_size + 1, window_size):
        w = df.iloc[start:start + window_size]

        feats = {
            'mean_acc_x': w['AccX'].mean(),
            'mean_acc_y': w['AccY'].mean(),
            'mean_acc_z': w['AccZ'].mean(),
            'std_acc_x':  w['AccX'].std(),
            'std_acc_y':  w['AccY'].std(),
            'std_acc_z':  w['AccZ'].std(),
            'skew_acc_x': w['AccX'].skew(),
            'skew_acc_y': w['AccY'].skew(),
            'skew_acc_z': w['AccZ'].skew(),
            'kurt_acc_x': w['AccX'].kurt(),
            'kurt_acc_y': w['AccY'].kurt(),
            'kurt_acc_z': w['AccZ'].kurt(),
        }
        X_feats.append(feats)

        # Label for the window: first-sample label (matches your original code)
        # (switch to majority vote with w['Label'].mode().iloc[0] if you prefer)
        y_labels.append(w['Label'].iloc[0])

        # Cow ID = file index (1..6) or use the filename itself
        cow_ids.append(idx)

X = pd.DataFrame(X_feats)
y = np.asarray(y_labels)
cows = np.asarray(cow_ids)

labels_sorted = np.sort(np.unique(y))
print("Windowed class counts:", Counter(y))

# -------------------------
# 2) LOCO: hold out one cow at a time
#    - Fit imputer/scaler on TRAIN only
#    - SMOTE on TRAIN only (prefer k=2; fallback if needed)
# -------------------------
results = []
overall_cm = np.zeros((len(labels_sorted), len(labels_sorted)), dtype=int)

for held_out in np.unique(cows):
    test_mask = (cows == held_out)
    train_mask = ~test_mask

    X_train_raw, X_test_raw = X.iloc[train_mask], X.iloc[test_mask]
    y_train, y_test         = y[train_mask], y[test_mask]

    # Impute & scale (fit on train)
    imputer = SimpleImputer(strategy='mean')
    scaler  = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp  = imputer.transform(X_test_raw)

    X_train = scaler.fit_transform(X_train_imp)
    X_test  = scaler.transform(X_test_imp)

    # SMOTE on TRAIN: choose safe k (prefer 2)
    class_counts = Counter(y_train)
    min_count = min(class_counts.values())
    if min_count >= 3:
        k_neighbors = 2
    elif min_count == 2:
        k_neighbors = 1
    else:
        k_neighbors = None  # cannot SMOTE

    if k_neighbors is None:
        X_train_bal, y_train_bal = X_train, y_train
        used_smote = False
    else:
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        used_smote = True

    # Model
    rf = RandomForestClassifier(
        n_estimators=100,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_bal, y_train_bal)

    # Evaluate on held-out cow
    y_pred = rf.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1w  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    results.append({
        "held_out_cow": held_out,
        "n_test_windows": int(test_mask.sum()),
        "accuracy": acc,
        "f1_weighted": f1w,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "used_smote": used_smote,
        "smote_k": k_neighbors if used_smote else None,
        "min_class_in_train": min_count
    })

    overall_cm += confusion_matrix(y_test, y_pred, labels=labels_sorted)

# -------------------------
# 3) Report
# -------------------------
res_df = pd.DataFrame(results).sort_values("held_out_cow").reset_index(drop=True)
print("\n=== Leave-One-Cow-Out (LOCO) per-cow results ===")
print(res_df.to_string(index=False))

print("\n=== LOCO mean ± std (across cows) ===")
for m in ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]:
    vals = res_df[m].values
    print(f"{m:>18}: {vals.mean():.4f} ± {vals.std(ddof=1):.4f}")

print("\n=== Aggregated classification report (concatenated across folds) ===")
# (If you want a single report, you can store all y_test/y_pred across folds and call classification_report on the concatenation)
# For simplicity here, show per-class support from the aggregated confusion matrix:
print("Labels order:", labels_sorted.tolist())
print("Aggregated Confusion Matrix (rows=true, cols=pred):\n", overall_cm)

# Optional: visualize aggregated confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(overall_cm, interpolation='nearest')
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(len(labels_sorted)),
    yticks=np.arange(len(labels_sorted)),
    xticklabels=labels_sorted,
    yticklabels=labels_sorted,
    ylabel='True label',
    xlabel='Predicted label',
    title='Aggregated Confusion Matrix (LOCO)'
)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(overall_cm.shape[0]):
    for j in range(overall_cm.shape[1]):
        ax.text(j, i, int(overall_cm[i, j]),
                ha="center", va="center",
                color="w" if overall_cm[i, j] > overall_cm.max()/2 else "black")
plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data files
data_files = ["resampled_2_1S.csv", "resampled_2_3S.csv", "resampled_2_5S.csv", "resampled_16_1S.csv", "resampled_16_3S.csv", "resampled_16_5S.csv"]
data_frames = [pd.read_csv(file) for file in data_files]
data = pd.concat(data_frames, ignore_index=True)

# Set fixed window size
window_size = 15  # Fixed number of samples

# Extract windowed features and labels
windowed_features = []
windowed_labels = []

for i in range(0, len(data) - window_size + 1, window_size):
    window = data.iloc[i:i + window_size]
    if len(window) == window_size:
        window_features = window.drop(columns=['date', 'label']).mean()
        window_label = window['label'].iloc[0]
        windowed_features.append(window_features)
        windowed_labels.append(window_label)

# Convert to DataFrame
X = pd.DataFrame(windowed_features)
y = np.array(windowed_labels)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Handle imbalance
smote = SMOTE(random_state=42, k_neighbors=2)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Initialize classifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict
y_pred_rf = rf_classifier.predict(X_test)
y_pred_proba_rf = rf_classifier.predict_proba(X_test)

# Evaluate
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf, average='weighted')
precision = precision_score(y_test, y_pred_rf, average='weighted')
recall = recall_score(y_test, y_pred_rf, average='weighted')
gini = 2 * roc_auc_score(y_test, y_pred_proba_rf, multi_class='ovr', average='weighted') - 1

print(f"\nResults using fixed window size of 15 samples:")
print(f"  Accuracy: {accuracy_rf * 100:.2f}%")
print(f"  F1 Score: {f1:.2f}")
print(f"  Precision: {precision:.2f}")
print(f"  Recall: {recall:.2f}")
print(f"  Gini Score: {gini:.2f}")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data files
data_files = ["resampled_2_1S.csv", "resampled_2_3S.csv", "resampled_2_5S.csv"]
data_frames = [pd.read_csv(file) for file in data_files]
data = pd.concat(data_frames, ignore_index=True)

# Set fixed window size
window_size = 15  # Fixed number of samples

# Extract windowed features and labels
windowed_features = []
windowed_labels = []

for i in range(0, len(data) - window_size + 1, window_size):
    window = data.iloc[i:i + window_size]
    if len(window) == window_size:
        window_features = window.drop(columns=['date', 'label']).mean()
        window_label = window['label'].iloc[0]
        windowed_features.append(window_features)
        windowed_labels.append(window_label)

# Convert to DataFrame
X = pd.DataFrame(windowed_features)
y = np.array(windowed_labels)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Handle imbalance
smote = SMOTE(random_state=42, k_neighbors=2)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Initialize classifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict
y_pred_rf = rf_classifier.predict(X_test)
y_pred_proba_rf = rf_classifier.predict_proba(X_test)

# Evaluate
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf, average='weighted')
precision = precision_score(y_test, y_pred_rf, average='weighted')
recall = recall_score(y_test, y_pred_rf, average='weighted')
gini = 2 * roc_auc_score(y_test, y_pred_proba_rf, multi_class='ovr', average='weighted') - 1

print(f"\nResults using fixed window size of 15 samples:")
print(f"  Accuracy: {accuracy_rf * 100:.2f}%")
print(f"  F1 Score: {f1:.2f}")
print(f"  Precision: {precision:.2f}")
print(f"  Recall: {recall:.2f}")
print(f"  Gini Score: {gini:.2f}")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data files
data_files = ["resampled_16_1S.csv", "resampled_16_3S.csv", "resampled_16_5S.csv"]
data_frames = [pd.read_csv(file) for file in data_files]
data = pd.concat(data_frames, ignore_index=True)

# Set fixed window size
window_size = 15  # Fixed number of samples

# Extract windowed features and labels
windowed_features = []
windowed_labels = []

for i in range(0, len(data) - window_size + 1, window_size):
    window = data.iloc[i:i + window_size]
    if len(window) == window_size:
        window_features = window.drop(columns=['date', 'label']).mean()
        window_label = window['label'].iloc[0]
        windowed_features.append(window_features)
        windowed_labels.append(window_label)

# Convert to DataFrame
X = pd.DataFrame(windowed_features)
# Convert to categorical codes for classification
y = pd.Series(windowed_labels).astype("category").cat.codes


# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Count label occurrences
label_counts = pd.Series(y).value_counts()

# Keep only classes with at least 6 samples
valid_labels = label_counts[label_counts >= 6].index

# Filter data
X_filtered = X_scaled[np.isin(y, valid_labels)]
y_filtered = y[np.isin(y, valid_labels)]

# Then SMOTE
X_resampled, y_resampled = SMOTE(random_state=42, k_neighbors=2).fit_resample(X_filtered, y_filtered)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Initialize classifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict
y_pred_rf = rf_classifier.predict(X_test)
y_pred_proba_rf = rf_classifier.predict_proba(X_test)

# Evaluate
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf, average='weighted')
precision = precision_score(y_test, y_pred_rf, average='weighted')
recall = recall_score(y_test, y_pred_rf, average='weighted')
gini = 2 * roc_auc_score(y_test, y_pred_proba_rf, multi_class='ovr', average='weighted') - 1

print(f"\nResults using fixed window size of 15 samples:")
print(f"  Accuracy: {accuracy_rf * 100:.2f}%")
print(f"  F1 Score: {f1:.2f}")
print(f"  Precision: {precision:.2f}")
print(f"  Recall: {recall:.2f}")
print(f"  Gini Score: {gini:.2f}")

import pandas as pd
import numpy as np
from collections import Counter
import math

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, make_scorer
)
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# -------------------------
# Helpers
# -------------------------
def build_windows_from_files(file_list, window_size=15, date_col="date", label_col="label"):
    """
    Loads CSVs, concatenates, and builds non-overlapping windows.
    Features = mean of all numeric columns except date/label over the window.
    Label = first label in the window (swap to majority if preferred).
    """
    df = pd.concat([pd.read_csv(f) for f in file_list], ignore_index=True)

    # Make sure label column is present
    if label_col not in df.columns:
        # Try some common fallbacks if needed
        alt = [c for c in df.columns if c.lower() in ("label", "label [-]", "behaviour", "behavior")]
        if not alt:
            raise ValueError(f"Couldn't find label column '{label_col}' in: {list(df.columns)}")
        label_col = alt[0]

    # Select numeric feature columns (drop date/label if present)
    drop_cols = [c for c in [date_col, label_col] if c in df.columns]
    feat_df = df.drop(columns=drop_cols, errors="ignore")

    # Keep only numeric columns for features
    feat_df = feat_df.select_dtypes(include=[np.number])
    if feat_df.shape[1] == 0:
        raise ValueError("No numeric feature columns found after dropping date/label.")

    windowed_features, windowed_labels = [], []

    # Non-overlapping windows
    for i in range(0, len(df) - window_size + 1, window_size):
        window_idx = slice(i, i + window_size)
        # features: mean over window for each numeric column
        window_feats = feat_df.iloc[window_idx].mean(axis=0)
        # label: first label in the window (or use majority vote)
        window_label = df[label_col].iloc[i]
        # majority option:
        # window_label = df[label_col].iloc[i:i+window_size].mode(dropna=False).iloc[0]

        windowed_features.append(window_feats.to_dict())
        windowed_labels.append(window_label)

    X = pd.DataFrame(windowed_features)
    y = np.array(windowed_labels)
    return X, y

def gini_like(y_true, y_pred):
    """Match your custom gini-like scorer (weighted recall vs precision)."""
    tpr = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    fpr = 1 - precision_score(y_true, y_pred, average='weighted', zero_division=0)
    return 2 * (tpr - fpr)

def choose_k2_safe_splits(y, max_splits=5, needed_per_class_train=3):
    """
    Find the largest n_splits in [2..max_splits] such that the smallest class
    will have at least 'needed_per_class_train' samples in the TRAIN portion.
    Uses min_count * (n_splits - 1) / n_splits (approx expected train count).
    """
    counts = Counter(y)
    min_count = min(counts.values())
    for s in range(max_splits, 1, -1):  # try 5,4,3,2
        train_min = math.floor(min_count * (s - 1) / s)
        if train_min >= needed_per_class_train:
            return s
    return None, min_count

# -------------------------
# Main runner
# -------------------------
def run_5fold_cv(file_list, window_size=15, k_neighbors=2, random_state=42, max_splits=5):
    # 1) Build windows
    X, y = build_windows_from_files(file_list, window_size=window_size)

    # 2) Build pipeline (Impute -> Scale -> SMOTE(k=2) -> RF)
    rf = RandomForestClassifier(
        n_estimators=100,
        max_features='sqrt',
        random_state=random_state,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=random_state, k_neighbors=k_neighbors)),
        ('rf', rf)
    ])

    # 3) Scoring dict
    scoring = {
        'accuracy': 'accuracy',
        'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
        'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0),
        'gini_like': make_scorer(gini_like)
    }

    # 4) Choose k=2-safe number of CV splits (based on post-windowing counts)
    n_splits = None
    counts = Counter(y)
    min_count = min(counts.values())
    # Need >= (k+1) per class in TRAIN; with k=2, need 3 minimum in train
    needed_train = k_neighbors + 1
    for s in range(max_splits, 1, -1):
        train_min = math.floor(min_count * (s - 1) / s)
        if train_min >= needed_train:
            n_splits = s
            break

    if n_splits is None:
        raise ValueError(
            f"Rarest label has only {min_count} windows after windowing; "
            f"no CV split in [2..{max_splits}] yields ≥{needed_train} per-class in TRAIN for SMOTE(k={k_neighbors}). "
            f"Options: overlap windows (stride < {window_size}), merge/drop ultra-rare labels for this run, or use k_neighbors=1."
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # 5) Cross-validate + OOF predictions
    cvres = cross_validate(
        pipe,
        X, y,
        cv=skf,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )
    y_pred_oof = cross_val_predict(pipe, X, y, cv=skf, n_jobs=-1)
    labels = np.unique(y)
    cm = confusion_matrix(y, y_pred_oof, labels=labels)

    # 6) Print results
    def mean_std(name):
        vals = cvres[f'test_{name}']
        return f"{vals.mean():.4f} ± {vals.std():.4f}"

    print(f"Files used: {file_list}")
    print(f"Window size: {window_size}   |   CV splits used (k=2-safe): {n_splits}")
    print(f"Class counts after windowing: {dict(Counter(y))}\n")

    print("5-Fold CV Results (mean ± std)")
    print(f"Accuracy:            {mean_std('accuracy')}")
    print(f"F1 (weighted):       {mean_std('f1_weighted')}")
    print(f"Precision (weighted):{mean_std('precision_weighted')}")
    print(f"Recall (weighted):   {mean_std('recall_weighted')}")
    print(f"Gini-like:           {mean_std('gini_like')}")

    print("\nOut-of-fold classification report:")
    print(classification_report(y, y_pred_oof, digits=3, zero_division=0))

    print("Labels order in confusion matrix:", labels.tolist())
    print("Confusion matrix (rows=true, cols=pred):\n", cm)

# -------------------------
# Example runs
# -------------------------

# 1) All six files together
run_5fold_cv(
    ["resampled_2_1S.csv", "resampled_2_3S.csv", "resampled_2_5S.csv",
     "resampled_16_1S.csv", "resampled_16_3S.csv", "resampled_16_5S.csv"],
    window_size=15,
    k_neighbors=2
)

# 2) Only the "2_*S" set
run_5fold_cv(
    ["resampled_2_1S.csv", "resampled_2_3S.csv", "resampled_2_5S.csv"],
    window_size=15,
    k_neighbors=2
)

def build_windows_from_files(file_list, window_size=15, date_col="date", label_col="label", min_samples=3):
    """
    Merges labels with fewer than `min_samples` with the most frequent class.
    """
    df = pd.concat([pd.read_csv(f) for f in file_list], ignore_index=True)

    # Ensure the label column exists
    if label_col not in df.columns:
        alt = [c for c in df.columns if c.lower() in ("label", "label [-]", "behaviour", "behavior")]
        if not alt:
            raise ValueError(f"Couldn't find label column '{label_col}' in: {list(df.columns)}")
        label_col = alt[0]

    # Count label occurrences
    label_counts = df[label_col].value_counts()
    rare_labels = label_counts[label_counts < min_samples].index.tolist()

    # Merge rare labels with the most frequent one (or another strategy)
    most_frequent_label = label_counts.idxmax()  # Replace with the most frequent class
    df[label_col] = df[label_col].replace(rare_labels, most_frequent_label)

    # Select numeric feature columns (drop date/label if present)
    drop_cols = [c for c in [date_col, label_col] if c in df.columns]
    feat_df = df.drop(columns=drop_cols, errors="ignore")

    # Keep only numeric columns for features
    feat_df = feat_df.select_dtypes(include=[np.number])
    if feat_df.shape[1] == 0:
        raise ValueError("No numeric feature columns found after dropping date/label.")

    windowed_features, windowed_labels = [], []

    # Non-overlapping windows
    for i in range(0, len(df) - window_size + 1, window_size):
        window_idx = slice(i, i + window_size)
        window_feats = feat_df.iloc[window_idx].mean(axis=0)
        window_label = df[label_col].iloc[i]

        windowed_features.append(window_feats.to_dict())
        windowed_labels.append(window_label)

    X = pd.DataFrame(windowed_features)
    y = np.array(windowed_labels)
    return X, y

# 3) Only the "16_*S" set
run_5fold_cv(
    ["resampled_16_1S.csv", "resampled_16_3S.csv", "resampled_16_5S.csv"],
    window_size=15,
    k_neighbors=1  # Use k_neighbors=1 instead of 2 to handle rare labels
)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv("dataset_6.csv")

# Define window size
window_size = 15

# Initialize lists to store windowed features and labels
windowed_features = []
windowed_labels = []

# Segment the time-series data into windows
for i in range(0, len(data) - window_size + 1, window_size):
    window = data.iloc[i:i + window_size]
    if len(window) == window_size:  # Ensure the window is complete
        # Extract features from the window
        window_features = {
            'mean_acc_x': window['acc_x'].mean(),
            'mean_acc_y': window['acc_y'].mean(),
            'mean_acc_z': window['acc_z'].mean(),
            'std_acc_x': window['acc_x'].std(),
            'std_acc_y': window['acc_y'].std(),
            'std_acc_z': window['acc_z'].std(),
            'skew_acc_x': window['acc_x'].skew(),
            'skew_acc_y': window['acc_y'].skew(),
            'skew_acc_z': window['acc_z'].skew(),
            'kurt_acc_x': window['acc_x'].kurt(),
            'kurt_acc_y': window['acc_y'].kurt(),
            'kurt_acc_z': window['acc_z'].kurt()
        }
        windowed_features.append(window_features)

        # Assign label to the window (assuming it's the same for all samples within the window)
        window_label = window['label'].iloc[0]
        windowed_labels.append(window_label)

# Convert lists to DataFrame
X_windowed = pd.DataFrame(windowed_features)
y_windowed = np.array(windowed_labels)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_windowed)

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Deal with class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_windowed)

# Split the data into train and test using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Initialize lists to store metrics
f1_scores = []
precisions = []
recalls = []
accuracies = []

# Perform 10 iterations
for i in range(10):
    # Instantiate Random Forest classifier with random subspace
    rf_classifier = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)

    # Train the Random Forest classifier
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred_rf = rf_classifier.predict(X_test)

    # Evaluate the Random Forest model's performance
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    f1 = f1_score(y_test, y_pred_rf, average='weighted')
    precision = precision_score(y_test, y_pred_rf, average='weighted')
    recall = recall_score(y_test, y_pred_rf, average='weighted')

    accuracies.append(accuracy_rf)
    f1_scores.append(f1)
    precisions.append(precision)
    recalls.append(recall)

    print(f"Iteration {i + 1}:")
    print(f"  Accuracy: {accuracy_rf * 100:.2f}%")
    print(f"  F1 Score: {f1:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")

# Print average metrics
print("\nAverage Metrics over 10 iterations:")
print(f"  Average Accuracy: {np.mean(accuracies) * 100:.2f}%")
print(f"  Average F1 Score: {np.mean(f1_scores):.2f}")
print(f"  Average Precision: {np.mean(precisions):.2f}")
print(f"  Average Recall: {np.mean(recalls):.2f}")

# Plot confusion matrix for the last iteration
conf_matrix = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

import pandas as pd
import numpy as np
from collections import Counter
import math

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, make_scorer
)
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# -------------------------
# 1) Load & window the data (non-overlapping windows of 15)
# -------------------------
data = pd.read_csv("dataset_6.csv")  # expects: acc_x, acc_y, acc_z, label

required = {"acc_x", "acc_y", "acc_z", "label"}
missing = required - set(data.columns)
if missing:
    raise ValueError(f"dataset_6.csv missing columns: {missing}. Found: {list(data.columns)}")

window_size = 15
X_feats, y_labels = [], []

for i in range(0, len(data) - window_size + 1, window_size):
    w = data.iloc[i:i + window_size]

    feats = {
        'mean_acc_x': w['acc_x'].mean(),
        'mean_acc_y': w['acc_y'].mean(),
        'mean_acc_z': w['acc_z'].mean(),
        'std_acc_x':  w['acc_x'].std(),
        'std_acc_y':  w['acc_y'].std(),
        'std_acc_z':  w['acc_z'].std(),
        'skew_acc_x': w['acc_x'].skew(),
        'skew_acc_y': w['acc_y'].skew(),
        'skew_acc_z': w['acc_z'].skew(),
        'kurt_acc_x': w['acc_x'].kurt(),
        'kurt_acc_y': w['acc_y'].kurt(),
        'kurt_acc_z': w['acc_z'].kurt(),
    }
    X_feats.append(feats)

    # Label: first sample in window (swap to majority vote if preferred)
    # label = w['label'].mode(dropna=False).iloc[0]
    label = w['label'].iloc[0]
    y_labels.append(label)

X = pd.DataFrame(X_feats)
y = np.asarray(y_labels)

print("Windowed class counts:", Counter(y))

# -------------------------
# 2) Pipeline: Impute -> Scale -> SMOTE(k=2) -> RF
# -------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42, k_neighbors=2)),  # keep k=2
    ('rf', rf)
])

# -------------------------
# 3) Scoring (same family you’ve been using)
# -------------------------
def gini_like(y_true, y_pred):
    tpr = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    fpr = 1 - precision_score(y_true, y_pred, average='weighted', zero_division=0)
    return 2 * (tpr - fpr)

from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score
scoring = {
    'accuracy': 'accuracy',
    'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
    'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
    'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0),
    'gini_like': make_scorer(gini_like)
}

# -------------------------
# 4) Choose a k=2-safe number of splits (prefer 5, fallback to 4/3/2 if needed)
# -------------------------
counts = Counter(y)
min_count = min(counts.values())
needed_train = 3  # k=2 needs >=3 per class in the TRAIN portion
n_splits = None
for s in (5, 4, 3, 2):
    train_min = math.floor(min_count * (s - 1) / s)  # approx min per class in train
    if train_min >= needed_train:
        n_splits = s
        break

if n_splits is None:
    raise ValueError(
        f"Rarest label has only {min_count} windows after windowing; "
        f"no CV split in [2..5] yields ≥{needed_train} per-class in TRAIN for SMOTE(k=2).\n"
        f"Options: use overlapping windows (stride < {window_size}), merge/drop ultra-rare labels for this run, or set k_neighbors=1."
    )

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# -------------------------
# 5) Cross-validate and OOF predictions
# -------------------------
cvres = cross_validate(
    pipe,
    X, y,
    cv=skf,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False
)

y_pred_oof = cross_val_predict(pipe, X, y, cv=skf, n_jobs=-1)
labels = np.unique(y)
cm = confusion_matrix(y, y_pred_oof, labels=labels)

# -------------------------
# 6) Print results
# -------------------------
def mean_std(name):
    vals = cvres[f'test_{name}']
    return f"{vals.mean():.4f} ± {vals.std():.4f}"

print(f"\nCV splits used (k=2-safe): {n_splits}")
print("5-Fold-style CV Results (mean ± std)")
print(f"Accuracy:            {mean_std('accuracy')}")
print(f"F1 (weighted):       {mean_std('f1_weighted')}")
print(f"Precision (weighted):{mean_std('precision_weighted')}")
print(f"Recall (weighted):   {mean_std('recall_weighted')}")
print(f"Gini-like:           {mean_std('gini_like')}")

print("\nOut-of-fold classification report:")
print(classification_report(y, y_pred_oof, digits=3, zero_division=0))

print("Labels order in confusion matrix:", labels.tolist())
print("Confusion matrix (rows=true, cols=pred):\n", cm)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Load the dataset (replace 'CURC.csv' with your actual file path)
data = pd.read_csv('CURC.csv')

# Convert the 'Time' column to datetime format (assuming Time is in HH:MM:SS format)
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.time

# Define a larger window size in seconds
window_size_seconds = 6  # Adjust this value as needed
window_size_samples = window_size_seconds  # 6 seconds = 6 data points

# Initialize lists to store windowed features and labels
windowed_features = []
windowed_labels = []

# Segment the time-series data into windows
for i in range(0, len(data) - window_size_samples + 1, window_size_samples):
    window = data.iloc[i:i + window_size_samples]
    if len(window) == window_size_samples:  # Ensure the window is complete
        # Extract features from the window
        window_features = {
            'mean_x': window['X-axis (g)'].mean(),
            'mean_y': window['Y-axis (g)'].mean(),
            'mean_z': window['Z-axis (g)'].mean(),
            'std_x': window['X-axis (g)'].std(),
            'std_y': window['Y-axis (g)'].std(),
            'std_z': window['Z-axis (g)'].std(),
            'skew_x': window['X-axis (g)'].skew(),
            'skew_y': window['Y-axis (g)'].skew(),
            'skew_z': window['Z-axis (g)'].skew(),
            'kurt_x': window['X-axis (g)'].kurt(),
            'kurt_y': window['Y-axis (g)'].kurt(),
            'kurt_z': window['Z-axis (g)'].kurt()
        }
        windowed_features.append(window_features)

        # Assign labels to the window based on 'IteragreementLocom' and 'IteragreementFeeding'
        locomotion_label = window['IteragreementLocom'].mode().iloc[0]
        feeding_label = window['IteragreementFeeding'].mode().iloc[0]
        combined_label = f"{locomotion_label}{feeding_label}"
        windowed_labels.append(combined_label)

# Convert lists to DataFrame
X_windowed = pd.DataFrame(windowed_features)
y_windowed = np.array(windowed_labels)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_windowed)

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Deal with class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_windowed)

# Split the data into train and test using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Initialize lists to store metrics
accuracies = []
f1_scores = []
precisions = []
recalls = []
gini_scores = []

# Gini score calculation using AUC-ROC
def gini_score(y_true, y_pred_proba):
    if y_pred_proba.ndim == 1:  # Binary classification
        auc = roc_auc_score(y_true, y_pred_proba)
    else:  # Multi-class classification
        auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
    return 2 * auc - 1

# Perform 10 iterations
for i in range(10):
    # Instantiate Random Forest classifier with random subspace
    rf_classifier = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)

    # Train the Random Forest classifier
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred_rf = rf_classifier.predict(X_test)
    y_pred_proba_rf = rf_classifier.predict_proba(X_test)  # Get predicted probabilities

    # Evaluate the Random Forest model's performance
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    f1 = f1_score(y_test, y_pred_rf, average='weighted')
    precision = precision_score(y_test, y_pred_rf, average='weighted')
    recall = recall_score(y_test, y_pred_rf, average='weighted')

    # Calculate Gini score using predicted probabilities
    gini = gini_score(y_test, y_pred_proba_rf)

    # Append metrics to lists
    accuracies.append(accuracy_rf)
    f1_scores.append(f1)
    precisions.append(precision)
    recalls.append(recall)
    gini_scores.append(gini)

    print(f"Iteration {i + 1}:")
    print(f"  Accuracy: {accuracy_rf * 100:.2f}%")
    print(f"  F1 Score: {f1:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  Gini Score: {gini:.2f}")

# Print average metrics
print("\nAverage Metrics over 10 iterations:")
print(f"  Average Accuracy: {np.mean(accuracies) * 100:.2f}%")
print(f"  Average F1 Score: {np.mean(f1_scores):.2f}")
print(f"  Average Precision: {np.mean(precisions):.2f}")
print(f"  Average Recall: {np.mean(recalls):.2f}")
print(f"  Average Gini Score: {np.mean(gini_scores):.2f}")

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=np.unique(y_resampled), yticklabels=np.unique(y_resampled))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

import pandas as pd
import numpy as np
from collections import Counter
import math
import warnings

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, make_scorer, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# -------------------------
# 0) Load & basic parsing
# -------------------------
warnings.filterwarnings("ignore")
data = pd.read_csv("CURC.csv")  # expects Time, X-axis (g), Y-axis (g), Z-axis (g), IteragreementLocom, IteragreementFeeding

# Optional: parse Time if you need it downstream (not used for modeling)
if "Time" in data.columns:
    try:
        data["Time"] = pd.to_datetime(data["Time"], format="%H:%M:%S").dt.time
    except Exception:
        pass  # skip if already parsed or different format

# -------------------------
# 1) Windowing (non-overlapping)
#    Using "6 seconds = 6 samples" as in your script
# -------------------------
window_size_seconds = 6
window_size_samples = window_size_seconds  # if sampling is 1 Hz; adjust if different

feat_cols_map = {
    "X-axis (g)": "X",
    "Y-axis (g)": "Y",
    "Z-axis (g)": "Z"
}

for col_old, col_new in feat_cols_map.items():
    if col_old not in data.columns:
        raise ValueError(f"Missing column: '{col_old}'. Found: {list(data.columns)}")
data = data.rename(columns=feat_cols_map)

# Label columns
loc_col = "IteragreementLocom"
feed_col = "IteragreementFeeding"
for c in [loc_col, feed_col]:
    if c not in data.columns:
        raise ValueError(f"Missing label column: '{c}'")

X_feats, y_labels = [], []

for i in range(0, len(data) - window_size_samples + 1, window_size_samples):
    w = data.iloc[i:i + window_size_samples]
    if len(w) != window_size_samples:
        continue

    feats = {
        "mean_x":  w["X"].mean(),
        "mean_y":  w["Y"].mean(),
        "mean_z":  w["Z"].mean(),
        "std_x":   w["X"].std(),
        "std_y":   w["Y"].std(),
        "std_z":   w["Z"].std(),
        "skew_x":  w["X"].skew(),
        "skew_y":  w["Y"].skew(),
        "skew_z":  w["Z"].skew(),
        "kurt_x":  w["X"].kurt(),
        "kurt_y":  w["Y"].kurt(),
        "kurt_z":  w["Z"].kurt()
    }
    X_feats.append(feats)

    # Combine locomotion + feeding via majority vote (mode)
    loc_mode  = w[loc_col].mode(dropna=False).iloc[0]
    feed_mode = w[feed_col].mode(dropna=False).iloc[0]
    combined_label = f"{loc_mode}{feed_mode}"
    y_labels.append(combined_label)

X = pd.DataFrame(X_feats)
y = np.asarray(y_labels)

print("Windowed class counts:", Counter(y))

# -------------------------
# 2) Pipeline: Impute -> Scale -> SMOTE(k=2) -> RF
# -------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42, k_neighbors=2)),  # keep k=2
    ('rf', rf)
])

# -------------------------
# 3) Scorers
# -------------------------
def gini_like(y_true, y_pred):
    tpr = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    fpr = 1 - precision_score(y_true, y_pred, average='weighted', zero_division=0)
    return 2 * (tpr - fpr)

scoring = {
    'accuracy': 'accuracy',
    'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
    'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
    'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0),
    'gini_like': make_scorer(gini_like)
}

# -------------------------
# 4) Choose k=2-safe #folds (prefer 5, fallback 4/3/2 if needed)
# -------------------------
min_count = min(Counter(y).values())
needed_train = 3  # k=2 requires >=3 per class in the TRAIN portion
n_splits = None
for s in (5, 4, 3, 2):
    train_min = math.floor(min_count * (s - 1) / s)
    if train_min >= needed_train:
        n_splits = s
        break

if n_splits is None:
    raise ValueError(
        f"Rarest label has only {min_count} windows after windowing; "
        f"no CV split in [2..5] yields ≥{needed_train} per-class in TRAIN for SMOTE(k=2).\n"
        f"Options: increase windows (overlap), merge/drop ultra-rare labels for this run, or set k_neighbors=1."
    )

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# -------------------------
# 5) Cross-validate + OOF predictions (labels and probabilities)
# -------------------------
cvres = cross_validate(
    pipe,
    X, y,
    cv=skf,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False
)

y_pred_oof = cross_val_predict(pipe, X, y, cv=skf, n_jobs=-1)
y_proba_oof = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba", n_jobs=-1)

labels = np.unique(y)
cm = confusion_matrix(y, y_pred_oof, labels=labels)

# AUC-Gini from OOF probabilities (multiclass OVR, weighted)
# Encode y to numeric for roc_auc_score
y_encoded = pd.Series(y).astype("category")
y_codes = y_encoded.cat.codes.to_numpy()
# Align proba columns to codes ordering of labels
# cross_val_predict returns columns in clf.classes_ order inside folds; but we used OOF—assume consistent
# Safer: remap using the final fitted pipeline on full data to get class ordering for proba columns
# For simplicity here, use labels order as columns order:
try:
    auc = roc_auc_score(y_codes, y_proba_oof, multi_class='ovr', average='weighted')
    gini_auc = 2 * auc - 1
except Exception:
    gini_auc = np.nan

# -------------------------
# 6) Print results
# -------------------------
def mean_std(name):
    vals = cvres[f'test_{name}']
    return f"{vals.mean():.4f} ± {vals.std():.4f}"

print(f"\nCV splits used (k=2-safe): {n_splits}")
print("Cross-Validation Results (mean ± std)")
print(f"Accuracy:            {mean_std('accuracy')}")
print(f"F1 (weighted):       {mean_std('f1_weighted')}")
print(f"Precision (weighted):{mean_std('precision_weighted')}")
print(f"Recall (weighted):   {mean_std('recall_weighted')}")
print(f"Gini-like:           {mean_std('gini_like')}")
print(f"AUC-based Gini (OOF): {gini_auc:.4f}")

print("\nOut-of-fold classification report:")
print(classification_report(y, y_pred_oof, digits=3, zero_division=0))

print("Labels order in confusion matrix:", labels.tolist())
print("Confusion matrix (rows=true, cols=pred):\n", cm)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------
# Confusion Matrix for 5-Fold Cross-Validation (original counts)
cm_5fold = np.array([[119, 51, 8],
                     [57, 247, 18],
                     [9, 18, 15]])

# Confusion Matrix for 80/20 Split (from the image you uploaded)
cm_8020 = np.array([[57, 5, 2],
                    [14, 46, 5],
                    [0, 1, 64]])

# Normalize to row-wise percentages
cm_5fold_percent = cm_5fold / cm_5fold.sum(axis=1, keepdims=True) * 100
cm_8020_percent = cm_8020 / cm_8020.sum(axis=1, keepdims=True) * 100

labels = ['eating', 'standing', 'walking']

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # slightly smaller to reduce whitespace

# --- 5-Fold CV ---
sns.heatmap(
    cm_5fold_percent,
    annot=True, fmt=".2f",
    cmap="Blues", cbar=False,
    annot_kws={"size": 16, "weight": "bold"},
    linewidths=2, linecolor='black',
    square=True, ax=axes[0]
)
axes[0].set_xticklabels(labels, fontsize=16, rotation=0, fontweight = 'bold')
axes[0].set_yticklabels(labels, fontsize=16, rotation=90, fontweight = 'bold')
# remove redundant axis titles to declutter
axes[0].set_xlabel("")
axes[0].set_ylabel("")

# --- 80/20 split ---
sns.heatmap(
    cm_8020_percent,
    annot=True, fmt=".2f",
    cmap="Blues", cbar=False,
    annot_kws={"size": 16, "weight": "bold"},
    linewidths=2, linecolor='black',
    square=True, ax=axes[1]
)
axes[1].set_xticklabels(labels, fontsize=16, rotation=0, fontweight = 'bold')
axes[1].set_yticklabels(labels, fontsize=16, rotation=90, fontweight = 'bold')
axes[1].set_xlabel("")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()

# cross_dataset_all_pairs_harmonized_v2.py
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)

# =========================
# CONFIG: DATASET PATHS (your files are in the working directory)
# =========================
D1_PATH   = "data.csv"  # Feng (0..4 labels)
D2_FILES  = ("cow1.csv","cow2.csv","cow3.csv","cow4.csv","cow5.csv","cow6.csv")  # Ito
D3A_FILES = ("resampled_2_1S.csv","resampled_2_3S.csv","resampled_2_5S.csv")     # Mladenova 2 Hz
D3B_FILES = ("resampled_16_1S.csv","resampled_16_3S.csv","resampled_16_5S.csv")  # Mladenova 16 Hz
D3C_FILES = D3A_FILES + D3B_FILES
D4_PATH   = "dataset_6.csv"     # Andersen (label column 'k')
D5_PATH   = "CURC.csv"          # CURC

# Window sizes (seconds / samples per your data rate)
WIN_OTHERS = 15   # D1/D2/D3/D4
WIN_CURC   = 6    # CURC (~1 Hz → 6 s windows)

# Minimum shared features to allow a train→test pair (if schemas differ)
MIN_SHARED_FEATS = 6

# =========================
# Helpers
# =========================
COMMON_CLASSES = {"Walking-other", "Standing-other", "Standing-eating"}

def normalize_axes(df, candidates=None):
    """Find X/Y/Z accelerometer columns despite naming differences."""
    import re
    if candidates is None:
        candidates = {
            "x": [r"^x[-_\s]*axis", r"^acc[_-]?x", r"^ax$", r"^x\(g\)$", r"^x.*\(g\)$", r"^x$"],
            "y": [r"^y[-_\s]*axis", r"^acc[_-]?y", r"^ay$", r"^y\(g\)$", r"^y.*\(g\)$", r"^y$"],
            "z": [r"^z[-_\s]*axis", r"^acc[_-]?z", r"^az$", r"^z\(g\)$", r"^z.*\(g\)$", r"^z$"],
        }
    cols = {c.lower(): c for c in df.columns}
    picked = {}
    for axis, pats in candidates.items():
        for p in pats:
            for cl, orig in cols.items():
                if re.search(p, cl):
                    picked[axis] = orig
                    break
            if axis in picked:
                break
    if not {"x","y","z"}.issubset(set(picked.keys())):
        raise ValueError(f"Could not find X/Y/Z accelerometer columns. Found: {picked}. "
                         f"Available columns: {list(df.columns)}")
    return picked["x"], picked["y"], picked["z"]

def find_label_column(df, preferred=("label","Label","behavior","Behavior","k","K")):
    """Return first matching label column (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for p in preferred:
        if p.lower() in lower:
            return lower[p.lower()]
    for c in df.columns:
        if c.lower() in {"lbl","class","activity","behaviour","behaviors","categories"}:
            return c
    raise KeyError("Could not locate a label column; checked common names.")

def map_numeric(series, mapping_num_to_name):
    """Map numeric-coded labels (int/float/str digits) to names; None if unknown."""
    def _m(v):
        if pd.isna(v): return None
        s = str(v).strip()
        try:
            k = int(float(s))
            return mapping_num_to_name.get(k)
        except ValueError:
            return None
    return series.apply(_m)

def report_label_alignment(name, raw_series, mapped_series):
    raw_unique = pd.Series(raw_series).dropna().astype(str).str.strip().unique()
    mapped = pd.Series(mapped_series)
    unmapped_mask = mapped.isna()
    unmapped_count = int(unmapped_mask.sum())
    total = int(mapped.shape[0])
    dropped_pct = 100.0 * unmapped_count / max(total, 1)

    print(f"\n=== {name}: label alignment report ===")
    print(f"Raw unique labels (sample): {list(raw_unique)[:15]}{' ...' if len(raw_unique)>15 else ''}")
    if unmapped_count > 0:
        um_raw = pd.Series(raw_series)[unmapped_mask].astype(str).str.strip()
        ex = um_raw.value_counts().head(10)
        print(f"Unmapped dropped: {unmapped_count}/{total} ({dropped_pct:.1f}%). Examples:\n{ex.to_string()}")
    else:
        print("All labels mapped successfully (no drops).")
    kept = mapped.dropna()
    print("Class counts after mapping:")
    print(kept.value_counts().reindex(sorted(COMMON_CLASSES)).fillna(0).astype(int).to_string())
    if dropped_pct > 25:
        print(f"[WARN] >25% labels dropped in {name}. Please review mapping.")
    if kept.nunique() < 2:
        raise ValueError(f"{name}: Fewer than two classes after mapping; cannot proceed.")

def build_stat_features_from_axes(df, x_col, y_col, z_col, window_size=15):
    """Always produce the same 12 stats from X/Y/Z in non-overlapping windows."""
    feats, labs_idx = [], []
    n = len(df)
    for i in range(0, n - window_size + 1, window_size):
        w = df.iloc[i:i+window_size]
        if len(w) < window_size:
            continue
        f = {
            'mean_x': w[x_col].mean(), 'std_x': w[x_col].std(),
            'skew_x': w[x_col].skew(), 'kurt_x': w[x_col].kurt(),
            'mean_y': w[y_col].mean(), 'std_y': w[y_col].std(),
            'skew_y': w[y_col].skew(), 'kurt_y': w[y_col].kurt(),
            'mean_z': w[z_col].mean(), 'std_z': w[z_col].std(),
            'skew_z': w[z_col].skew(), 'kurt_z': w[z_col].kurt(),
        }
        feats.append(f)
        labs_idx.append((i, i+window_size))
    X = pd.DataFrame(feats)
    return X, labs_idx

def assign_window_labels(label_series, windows):
    """Label each window by mode (most frequent) within the window slice."""
    out = []
    for i0, i1 in windows:
        wlab = label_series.iloc[i0:i1]
        lab = wlab.mode().iloc[0] if not wlab.mode().empty else wlab.iloc[0]
        out.append(lab)
    return np.array(out)

# =========================
# Dataset-specific mappings
# =========================
# D1 (Feng) 0..4
D1_NUM_TO_NAME = {0:"Feeding", 1:"Rumination", 2:"Standing", 3:"Lying", 4:"Walking"}
D1_NAME_TO_COMMON = {
    "Walking": "Walking-other",
    "Standing": "Standing-other",
    "Lying": "Standing-other",
    "Feeding": "Standing-eating",
    "Rumination": "Standing-other",
}

# D2 (Ito) — supports 13-code abbreviations and text
ITO_ABBR_TO_NAME = {
    "RES":"resting standing","RUS":"ruminating standing","MOV":"moving","GRZ":"grazing",
    "SLT":"salt licking","FES":"feeding stanchion","DRN":"drinking","LCK":"licking",
    "REL":"resting lying","URI":"urinating","ATT":"attacking","ESC":"escaping","BMN":"being mounted",
    "ETC":"resting standing"  # safety for 'ETC' seen in some files
}
D2_NAME_TO_COMMON = {
    "moving":"Walking-other","walking":"Walking-other","running":"Walking-other",
    "grazing":"Standing-eating","feeding":"Standing-eating","feeding stanchion":"Standing-eating",
    "resting standing":"Standing-other","ruminating standing":"Standing-other",
    "resting lying":"Standing-other","ruminating":"Standing-other",
    "standing":"Standing-other","lying":"Standing-other",
    "drinking":"Standing-other","licking":"Standing-other","salt licking":"Standing-other",
    "urinating":"Standing-other","attacking":"Standing-other","escaping":"Standing-other",
    "being mounted":"Standing-other",
}

# D3 (Mladenova) — numeric codes appear in your files; codebook below.
MLAD_NUM_TO_TEXT = {
    1: "standing and eating",
    2: "standing and ruminating",
    3: "laying and ruminating",
    4: "standing and ruminating",
    5: "standing and eating",
    6: "standing and ruminating",
}
MLAD_TEXT_TO_COMMON = {
    "standing and eating": "Standing-eating",
    "standing and ruminating": "Standing-other",
    "laying and ruminating": "Standing-other",
}

# D4 (Andersen) — label in column 'k'
D4_NAME_TO_COMMON = {
    "Walking":"Walking-other",
    "Grazing":"Standing-eating",
    "Standing":"Standing-other",
    "Standing-Ruminating":"Standing-other",
    "Standing-Resting":"Standing-other",
    "Ruminating":"Standing-other",
    "Resting":"Standing-other",
    "Lying-Ruminating":"Standing-other",
    "Lying-Resting":"Standing-other",
}

# =========================
# Dataset loaders (try raw axes → 12 stats; else keep engineered features)
# =========================
def load_d1(path=D1_PATH, window_size=WIN_OTHERS):
    df = pd.read_csv(path)
    labcol = find_label_column(df, preferred=("behavior","Behavior"))
    raw = df[labcol]
    raw_named = map_numeric(raw, D1_NUM_TO_NAME)
    mapped = raw_named.map(D1_NAME_TO_COMMON)
    report_label_alignment("D1_Feng", raw, mapped)
    try:
        xcol, ycol, zcol = normalize_axes(df)
        X, windows = build_stat_features_from_axes(df, xcol, ycol, zcol, window_size)
        y = assign_window_labels(mapped, windows)
        m = pd.Series(y).notna().values
        return X.iloc[m], y[m]
    except Exception:
        X = df.drop(columns=[labcol, "date", "time"], errors="ignore").copy()
        y = mapped.values
        m = pd.Series(y).notna().values
        return X[m], y[m]

def load_d2(files=D2_FILES, window_size=WIN_OTHERS):
    data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    labcol = find_label_column(data, preferred=("Label","label"))
    raw_series = data[labcol].astype(str).str.strip()
    def to_common(s):
        s_up = s.upper()
        if s_up in ITO_ABBR_TO_NAME:
            long_name = ITO_ABBR_TO_NAME[s_up]
            return D2_NAME_TO_COMMON.get(long_name, None)
        s_low = s.lower()
        if s_low == "resting": s_low = "resting standing"
        if s_low == "ruminating": s_low = "ruminating standing"
        if s_low in D2_NAME_TO_COMMON:
            return D2_NAME_TO_COMMON[s_low]
        if "graz" in s_low or "feed" in s_low: return "Standing-eating"
        if "mov" in s_low or "walk" in s_low or "run" in s_low: return "Walking-other"
        return "Standing-other"
    mapped = raw_series.apply(to_common)
    report_label_alignment("D2_Ito", raw_series, mapped)
    try:
        xcol, ycol, zcol = normalize_axes(data)
        X, windows = build_stat_features_from_axes(data, xcol, ycol, zcol, window_size)
        y = assign_window_labels(mapped, windows)
        m = pd.Series(y).notna().values
        return X.iloc[m], y[m]
    except Exception:
        X = data.drop(columns=[labcol, "date", "time"], errors="ignore").copy()
        y = mapped.values
        m = pd.Series(y).notna().values
        return X[m], y[m]

def _load_mlad(files, tag, window_size):
    data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    labcol = find_label_column(data, preferred=("label","Label"))
    raw = data[labcol]
    def to_text(v):
        s = str(v).strip()
        try:
            k = int(float(s))
            return MLAD_NUM_TO_TEXT.get(k)
        except ValueError:
            return s.lower()
    text = raw.apply(to_text)
    mapped = text.map(MLAD_TEXT_TO_COMMON)
    report_label_alignment(tag, raw, mapped)
    # Try to build 12 stats if raw axes exist; else keep engineered features
    try:
        xcol, ycol, zcol = normalize_axes(data)
        X, windows = build_stat_features_from_axes(data, xcol, ycol, zcol, window_size)
        y = assign_window_labels(mapped, windows)
        m = pd.Series(y).notna().values
        return X.iloc[m], y[m]
    except Exception:
        X = data.drop(columns=[labcol, "date", "time", "x_va"], errors="ignore").copy() # Drop "x_va" column
        y = mapped.values
        m = pd.Series(y).notna().values
        return X[m], y[m]

def load_d3a(files=D3A_FILES, window_size=WIN_OTHERS):
    return _load_mlad(files, "D3a_Mladenova_2Hz", window_size)

def load_d3b(files=D3B_FILES, window_size=WIN_OTHERS):
    return _load_mlad(files, "D3b_Mladenova_16Hz", window_size)

def load_d3c(files=D3C_FILES, window_size=WIN_OTHERS):
    return _load_mlad(files, "D3c_Mladenova_all", window_size)

def load_d4(path=D4_PATH, window_size=WIN_OTHERS):
    df = pd.read_csv(path)
    labcol = find_label_column(df, preferred=("k","K","label","Label"))
    raw = df[labcol].astype(str).str.strip()
    mapped = raw.apply(lambda s: D4_NAME_TO_COMMON.get(s, D4_NAME_TO_COMMON.get(s.title(), None)))
    report_label_alignment("D4_Andersen", raw, mapped)
    try:
        xcol, ycol, zcol = normalize_axes(df)
        X, windows = build_stat_features_from_axes(df, xcol, ycol, zcol, window_size)
        y = assign_window_labels(mapped, windows)
        m = pd.Series(y).notna().values
        return X.iloc[m], y[m]
    except Exception:
        X = df.drop(columns=[labcol, "date", "time"], errors="ignore").copy()
        y = mapped.values
        m = pd.Series(y).notna().values
        return X[m], y[m]

def load_d5(path=D5_PATH, window_size=WIN_CURC):
    df = pd.read_csv(path)
    # Heuristics for axes naming in CURC
    xcol, ycol, zcol = normalize_axes(df, candidates={
        "x":[r"x[-_\s]*axis.*\(g\)", r"^x-axis", r"^x.*\(g\)$", r"^x$"],
        "y":[r"y[-_\s]*axis.*\(g\)", r"^y-axis", r"^y.*\(g\)$", r"^y$"],
        "z":[r"z[-_\s]*axis.*\(g\)", r"^z-axis", r"^z.*\(g\)$", r"^z$"],
    })
    # Map labels (precombined in CURC or build from binary flags)
    if "Behavior" in df.columns:
        mapped = df["Behavior"]
        raw_src = df["Behavior"]
    else:
        def to_behavior(row):
            loc = str(row.get("Locomotion", row.get("IteragreementLocom",""))).lower()
            feed = str(row.get("Feeding", row.get("IteragreementFeeding",""))).lower()
            def is_yes(v):
                s = str(v).strip().lower()
                return s in {"1","true","yes"} or ("walk" in s) or ("eat" in s) or ("feed" in s)
            is_walk = ("walk" in loc) or is_yes(row.get("IteragreementLocom",0))
            is_feed = ("eat" in feed or "feed" in feed) or is_yes(row.get("IteragreementFeeding",0))
            if is_walk and not is_feed: return "Walking-other"
            if (not is_walk) and is_feed: return "Standing-eating"
            return "Standing-other"
        mapped = df.apply(to_behavior, axis=1)
        raw_src = mapped
    report_label_alignment("D5_CURC", raw_src, mapped)
    X, windows = build_stat_features_from_axes(df, xcol, ycol, zcol, window_size)
    y = assign_window_labels(mapped, windows)
    m = pd.Series(y).notna().values
    return X.iloc[m], y[m]

# =========================
# Train on A, test on B (schema align + speed knobs)
# =========================
def train_A_test_B(XA, yA, XB, yB, use_smote=True, random_state=42):
    # Ensure DataFrame
    if not isinstance(XA, pd.DataFrame):
        XA = pd.DataFrame(XA)
    if not isinstance(XB, pd.DataFrame):
        XB = pd.DataFrame(XB)

    # Align feature schema by intersection (same order)
    shared = [c for c in XA.columns if c in set(XB.columns)]
    if len(shared) < MIN_SHARED_FEATS:
        raise ValueError(f"Schema mismatch: only {len(shared)} shared features; "
                         f"need ≥{MIN_SHARED_FEATS}. Train cols={len(XA.columns)}, Test cols={len(XB.columns)}.")

    XA = XA[shared].copy()
    XB = XB[shared].copy()

    # Optional: cap huge training sets per class for speed
    CAP_PER_CLASS = 8000
    if CAP_PER_CLASS is not None:
        idx_keep, counts = [], {}
        for i, yi in enumerate(yA):
            c = counts.get(yi, 0)
            if c < CAP_PER_CLASS:
                idx_keep.append(i); counts[yi] = c + 1
        XA, yA = XA.iloc[idx_keep], np.array(yA)[idx_keep]

    imp = SimpleImputer(strategy="mean")
    scl = StandardScaler()
    XA_ = scl.fit_transform(imp.fit_transform(XA))
    XB_ = scl.transform(imp.transform(XB))

    # Guard SMOTE for very large sets
    if use_smote:
        fac = pd.Series(yA).factorize()[0]
        min_count = np.bincount(fac).min() if len(np.unique(fac)) > 1 else 0
        total_train = XA_.shape[0]
        do_smote = (min_count > 1) and (total_train < 40000)
        if do_smote:
            k = max(1, min(5, min_count - 1))
            XA_, yA = SMOTE(random_state=random_state, k_neighbors=k).fit_resample(XA_, yA)

    clf = RandomForestClassifier(
        n_estimators=150, max_depth=20, max_features="sqrt",
        class_weight="balanced_subsample", random_state=random_state, n_jobs=-1
    )
    clf.fit(XA_, yA)
    yhat = clf.predict(XB_)

    return {
        "acc": accuracy_score(yB, yhat),
        "f1_macro": f1_score(yB, yhat, average="macro"),
        "f1_weighted": f1_score(yB, yhat, average="weighted"),
        "prec_weighted": precision_score(yB, yhat, average="weighted", zero_division=0),
        "rec_weighted": recall_score(yB, yhat, average="weighted", zero_division=0),
        "cm": confusion_matrix(yB, yhat, labels=np.unique(yB)),
        "labels": np.unique(yB),
        "n_feats": XA.shape[1],
    }

# =========================
# Run all train→test pairs
# =========================
if __name__ == "__main__":
    datasets = {
        "D1_Feng":     (load_d1,  {}),
        "D2_Ito":      (load_d2,  {}),
        "D3a_Mlad2":   (load_d3a, {}),
        "D3b_Mlad16":  (load_d3b, {}),
        "D3c_MladAll": (load_d3c, {}),
        "D4_Andersen": (load_d4,  {}),
        "D5_CURC":     (load_d5,  {}),
    }

    cached, schema_info = {}, {}
    for name, (fn, kwargs) in datasets.items():
        try:
            X, y = fn(**kwargs)
            if len(y) < 20 or len(set(y)) < 2:
                print(f"[WARN] {name}: too few usable samples/classes; skipping.")
                continue
            cached[name] = (X, y)
            schema_info[name] = list(X.columns)
            print(f"[OK] Loaded {name}: X={X.shape}, classes={sorted(set(y)))}")
        except Exception as e:
            print(f"[ERROR] Loading {name} failed: {e}")

    # Print a compact schema summary
    if schema_info:
        print("\n=== Feature schema summary ===")
        for k, cols in schema_info.items():
            print(f"{k:14s} | {len(cols):2d} features | sample: {cols[:6]}{' ...' if len(cols)>6 else ''}")

    rows = []
    keys = list(cached.keys())
    for A in keys:
        XA, yA = cached[A]
        for B in keys:
            if B == A:  # skip same→same
                continue
            XB, yB = cached[B]
            try:
                res = train_A_test_B(XA, yA, XB, yB, use_smote=True)
                rows.append({
                    "Train": A,
                    "Test": B,
                    "Shared_feats": res["n_feats"],
                    "Accuracy": round(res["acc"], 4),
                    "F1_macro": round(res["f1_macro"], 4),
                    "F1_weighted": round(res["f1_weighted"], 4),
                    "Precision_weighted": round(res["prec_weighted"], 4),
                    "Recall_weighted": round(res["rec_weighted"], 4),
                })
                print(f"Train → Test: {A} → {B} | Feats: {res['n_feats']:2d} | "
                      f"Acc: {res['acc']*100:.2f}% | F1_macro: {res['f1_macro']:.3f} | F1_w: {res['f1_weighted']:.3f}")
            except Exception as e:
                print(f"[SKIP] {A}→{B}: {e}")

    if rows:
        dfres = pd.DataFrame(rows).sort_values(["Train","Test"]).reset_index(drop=True)
        outpath = Path("cross_dataset_results.csv")
        dfres.to_csv(outpath, index=False)
        print("\n=== Summary (train→test) ===")
        print(dfres.to_string(index=False))
        print(f"\nSaved: {outpath.resolve()}")
    else:
        print("No results produced. Check dataset paths, labels, or schema compatibility.")

import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Base data
# -------------------------
datasets = ['1', '2', '3a', '3b', '3c', '4', '5']
acc_8020 = [92.72, 93.41, 94.25, 96.95, 94.25, 90.17, 86.08]  # %
f1_8020  = [0.93, 0.93, 0.94, 0.97, 0.94, 0.90, 0.86]         # fractions
published = [85.67, 98.7, 88, 99, "NA", 87.15, "NA"]

# Convert published to numeric with NaNs
published_numeric = [p if p != "NA" else np.nan for p in published]

# -------------------------
# Dataset 1 extra results
# -------------------------
# Dataset 1
acc_5fold_mean_d1 = 0.8355 * 100
acc_5fold_std_d1  = 0.0032 * 100
f1_5fold_mean_d1  = 0.8343 * 100
f1_5fold_std_d1   = 0.0031 * 100

acc_loio_mean_d1  = 0.7924 * 100
acc_loio_std_d1   = 0.0274 * 100
f1_loio_mean_d1   = 0.7942 * 100
f1_loio_std_d1    = 0.0275 * 100

# -------------------------
# Dataset 2 results (LOCO)
# -------------------------
# LOCO for Dataset 2
acc_loio_mean_d2  = 0.6149 * 100
acc_loio_std_d2   = 0.1132 * 100
f1_loio_mean_d2   = 0.6134 * 100
f1_loio_std_d2    = 0.1270 * 100

# -------------------------
# Dataset 3a, 3b, and 3c extra results
# -------------------------
# Dataset 3a
acc_5fold_mean_d3a = 0.8315 * 100
acc_5fold_std_d3a  = 0.0150 * 100
f1_5fold_mean_d3a  = 0.8251 * 100
f1_5fold_std_d3a   = 0.0135 * 100

# Dataset 3b
acc_5fold_mean_d3b = 0.8824 * 100
acc_5fold_std_d3b  = 0.0076 * 100
f1_5fold_mean_d3b  = 0.8838 * 100
f1_5fold_std_d3b   = 0.0077 * 100

# Dataset 3c
acc_5fold_mean_d3c = 0.8489 * 100
acc_5fold_std_d3c  = 0.0119 * 100
f1_5fold_mean_d3c  = 0.8470 * 100
f1_5fold_std_d3c   = 0.0113 * 100


# -------------------------
# Dataset 4 extra results
# -------------------------
# Dataset 4 (Andersen) - Results
acc_5fold_mean_d4 = 0.6376 * 100
acc_5fold_std_d4 = 0.0342 * 100
f1_5fold_mean_d4 = 0.6377 * 100
f1_5fold_std_d4 = 0.0324 * 100

# -------------------------
# Dataset 5 extra results
# -------------------------
# Dataset 5 (Standing) - Results
acc_5fold_mean_d5 = 0.7030 * 100
acc_5fold_std_d5 = 0.0428 * 100
f1_5fold_mean_d5 = 0.7032 * 100
f1_5fold_std_d5 = 0.0408 * 100


# -------------------------
# Prepare plotting arrays
# -------------------------
n = len(datasets)
idx = np.arange(n)
bar_width = 0.11

# -------------------------
# Plotting
# -------------------------
fig, ax1 = plt.subplots(figsize=(14, 8))
ax2 = ax1.twinx()

# Color palette: shades per family
colors = {
    "8020_acc": "royalblue",
    "8020_f1": "lightskyblue",
    "5fold_acc": "firebrick",
    "5fold_f1": "lightcoral",
    "loio_acc": "purple",
    "loio_f1": "orchid",
    "published": "mediumseagreen"
}

# Lists to store handles for the legend in desired order
handles = []

# Plot bars dynamically for each dataset
for i, dataset in enumerate(datasets):
    current_x = idx[i] - 3 * bar_width # Start position for the group of bars for this dataset
    bars_plotted = 0

    # 80/20 Accuracy and F1
    h_8020_acc = ax1.bar(current_x + bars_plotted * bar_width, acc_8020[i], bar_width, color=colors["8020_acc"], edgecolor='black')
    h_8020_f1 = ax2.bar(current_x + (bars_plotted + 1) * bar_width, f1_8020[i] * 100, bar_width, color=colors["8020_f1"], edgecolor='black')
    if i == 0: # Add handles only once for the legend
        handles.extend([h_8020_acc, h_8020_f1])
    bars_plotted += 2

    # 5-Fold Accuracy and F1
    if dataset == '1':
        h_5fold_acc = ax1.bar(current_x + bars_plotted * bar_width, acc_5fold_mean_d1, bar_width, color=colors["5fold_acc"], edgecolor='black')
        ax1.errorbar(current_x + bars_plotted * bar_width, acc_5fold_mean_d1, yerr=acc_5fold_std_d1, fmt='none', ecolor='black', capsize=5)
        h_5fold_f1 = ax2.bar(current_x + (bars_plotted + 1) * bar_width, f1_5fold_mean_d1, bar_width, color=colors["5fold_f1"], edgecolor='black')
        ax2.errorbar(current_x + (bars_plotted + 1) * bar_width, f1_5fold_mean_d1, yerr=f1_5fold_std_d1, fmt='none', ecolor='black', capsize=5)
        if i == 0:
            handles.extend([h_5fold_acc, h_5fold_f1])
        bars_plotted += 2
    elif dataset == '3a':
        h_5fold_acc = ax1.bar(current_x + bars_plotted * bar_width, acc_5fold_mean_d3a, bar_width, color=colors["5fold_acc"], edgecolor='black')
        ax1.errorbar(current_x + bars_plotted * bar_width, acc_5fold_mean_d3a, yerr=acc_5fold_std_d3a, fmt='none', ecolor='black', capsize=5)
        h_5fold_f1 = ax2.bar(current_x + (bars_plotted + 1) * bar_width, f1_5fold_mean_d3a, bar_width, color=colors["5fold_f1"], edgecolor='black')
        ax2.errorbar(current_x + (bars_plotted + 1) * bar_width, f1_5fold_mean_d3a, yerr=f1_5fold_std_d3a, fmt='none', ecolor='black', capsize=5)
        if i == 2:
            handles.extend([h_5fold_acc, h_5fold_f1])
        bars_plotted += 2
    elif dataset == '3b':
        h_5fold_acc = ax1.bar(current_x + bars_plotted * bar_width, acc_5fold_mean_d3b, bar_width, color=colors["5fold_acc"], edgecolor='black')
        ax1.errorbar(current_x + bars_plotted * bar_width, acc_5fold_mean_d3b, yerr=acc_5fold_std_d3b, fmt='none', ecolor='black', capsize=5)
        h_5fold_f1 = ax2.bar(current_x + (bars_plotted + 1) * bar_width, f1_5fold_mean_d3b, bar_width, color=colors["5fold_f1"], edgecolor='black')
        ax2.errorbar(current_x + (bars_plotted + 1) * bar_width, f1_5fold_mean_d3b, yerr=f1_5fold_std_d3b, fmt='none', ecolor='black', capsize=5)
        if i == 3:
            handles.extend([h_5fold_acc, h_5fold_f1])
        bars_plotted += 2
    elif dataset == '3c':
        h_5fold_acc = ax1.bar(current_x + bars_plotted * bar_width, acc_5fold_mean_d3c, bar_width, color=colors["5fold_acc"], edgecolor='black')
        ax1.errorbar(current_x + bars_plotted * bar_width, acc_5fold_mean_d3c, yerr=acc_5fold_std_d3c, fmt='none', ecolor='black', capsize=5)
        h_5fold_f1 = ax2.bar(current_x + (bars_plotted + 1) * bar_width, f1_5fold_mean_d3c, bar_width, color=colors["5fold_f1"], edgecolor='black')
        ax2.errorbar(current_x + (bars_plotted + 1) * bar_width, f1_5fold_mean_d3c, yerr=f1_5fold_std_d3c, fmt='none', ecolor='black', capsize=5)
        if i == 4:
            handles.extend([h_5fold_acc, h_5fold_f1])
        bars_plotted += 2
    elif dataset == '4':
        h_5fold_acc = ax1.bar(current_x + bars_plotted * bar_width, acc_5fold_mean_d4, bar_width, color=colors["5fold_acc"], edgecolor='black')
        ax1.errorbar(current_x + bars_plotted * bar_width, acc_5fold_mean_d4, yerr=acc_5fold_std_d4, fmt='none', ecolor='black', capsize=5)
        h_5fold_f1 = ax2.bar(current_x + (bars_plotted + 1) * bar_width, f1_5fold_mean_d4, bar_width, color=colors["5fold_f1"], edgecolor='black')
        ax2.errorbar(current_x + (bars_plotted + 1) * bar_width, f1_5fold_mean_d4, yerr=f1_5fold_std_d4, fmt='none', ecolor='black', capsize=5)
        if i == 5:
            handles.extend([h_5fold_acc, h_5fold_f1])
        bars_plotted += 2
    elif dataset == '5':
        h_5fold_acc = ax1.bar(current_x + bars_plotted * bar_width, acc_5fold_mean_d5, bar_width, color=colors["5fold_acc"], edgecolor='black')
        ax1.errorbar(current_x + bars_plotted * bar_width, acc_5fold_mean_d5, yerr=acc_5fold_std_d5, fmt='none', ecolor='black', capsize=5)
        h_5fold_f1 = ax2.bar(current_x + (bars_plotted + 1) * bar_width, f1_5fold_mean_d5, bar_width, color=colors["5fold_f1"], edgecolor='black')
        ax2.errorbar(current_x + (bars_plotted + 1) * bar_width, f1_5fold_mean_d5, yerr=f1_5fold_std_d5, fmt='none', ecolor='black', capsize=5)
        if i == 6:
            handles.extend([h_5fold_acc, h_5fold_f1])
        bars_plotted += 2


    # LOIO Accuracy and F1
    if dataset == '1':
        h_loio_acc = ax1.bar(current_x + bars_plotted * bar_width, acc_loio_mean_d1, bar_width, color=colors["loio_acc"], edgecolor='black')
        ax1.errorbar(current_x + bars_plotted * bar_width, acc_loio_mean_d1, yerr=acc_loio_std_d1, fmt='none', ecolor='black', capsize=5)
        h_loio_f1 = ax2.bar(current_x + (bars_plotted + 1) * bar_width, f1_loio_mean_d1, bar_width, color=colors["loio_f1"], edgecolor='black')
        ax2.errorbar(current_x + (bars_plotted + 1) * bar_width, f1_loio_mean_d1, yerr=f1_loio_std_d1, fmt='none', ecolor='black', capsize=5)
        if i == 0:
            handles.extend([h_loio_acc, h_loio_f1])
        bars_plotted += 2
    elif dataset == '2':
        h_loio_acc = ax1.bar(current_x + bars_plotted * bar_width, acc_loio_mean_d2, bar_width, color=colors["loio_acc"], edgecolor='black')
        ax1.errorbar(current_x + bars_plotted * bar_width, acc_loio_mean_d2, yerr=acc_loio_std_d2, fmt='none', ecolor='black', capsize=5)
        h_loio_f1 = ax2.bar(current_x + (bars_plotted + 1) * bar_width, f1_loio_mean_d2, bar_width, color=colors["loio_f1"], edgecolor='black')
        ax2.errorbar(current_x + (bars_plotted + 1) * bar_width, f1_loio_mean_d2, yerr=f1_loio_std_d2, fmt='none', ecolor='black', capsize=5)
        if i == 1:
            handles.extend([h_loio_acc, h_loio_f1])
        bars_plotted += 2

    # Published Accuracy
    if not np.isnan(published_numeric[i]):
        h_published = ax1.bar(current_x + bars_plotted * bar_width, published_numeric[i], bar_width, color=colors["published"], edgecolor='black')
        if i == 0: # Add handle only once for the legend
             handles.append(h_published)
        bars_plotted += 1


# -------------------------
# Formatting
# -------------------------
group_centers = idx  # Center the groups of bars on the tick marks
ax1.set_xticks(group_centers)
ax1.set_xticklabels(datasets, fontsize=16, fontweight='bold')
ax1.set_xlabel('Dataset', fontsize=18, fontweight='bold')
ax1.set_ylabel('Score (%)', fontsize=18, fontweight='bold')
ax1.tick_params(axis='y', labelsize=16)

# Remove the tick labels from the right-hand Y-axis
ax2.set_yticklabels([])  # This removes the tick labels
ax2.tick_params(axis='y', which='both', length=0)  # This removes the ticks themselves

ax2.tick_params(axis='y', labelsize=14)

# Set y-axis limits for better visualization
ax1.set_ylim(0, 110)
ax2.set_ylim(0, 1.1 * 100)  # F1 is a fraction, scale to percentage

# Legends
labels = ['80/20 Accuracy', '80/20 F1', '5-Fold Accuracy', '5-Fold F1', 'LOIO Accuracy', 'LOIO F1', 'Published Accuracy']
fig.legend(handles, labels, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=4)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data
# -----------------------------
datasets = {
    'Dataset 1': {
        'models': ['RF', 'XGB', 'LGBM', 'KNN', 'LR', 'DT', 'SVM', 'AdaBoost', 'GNB', 'ET'],
        'accuracies': [0.65, 0.64, 0.64, 0.62, 0.61, 0.56, 0.63, 0.60, 0.52, 0.63],
        'time': [1.66, 1.97, 1.55, 0.33, 0.14, 0.09, 6.17, 0.49, 0.02, 1.45],
    },
    'Dataset 2': {
        'models': ['RF', 'XGB', 'LGBM', 'KNN', 'LR', 'DT', 'SVM', 'AdaBoost', 'GNB', 'ET'],
        'accuracies': [0.69, 0.68, 0.69, 0.67, 0.63, 0.61, 0.63, 0.63, 0.44, 0.69],
        'time': [1.31, 4.64, 2.10, 0.31, 0.34, 0.09, 3.15, 0.05, 0.03, 1.50],
    },
    'Dataset 3': {
        'models': ['RF', 'XGB', 'LGBM', 'KNN', 'LR', 'DT', 'SVM', 'AdaBoost', 'GNB', 'ET'],
        'accuracies': [0.97, 0.97, 0.96, 0.94, 0.81, 0.89, 0.86, 0.54, 0.49, 0.96],
        'time': [2.98, 34.4, 9.66, 8.72, 32.57, 0.70, 13.10, 0.20, 0.09, 1.14],
    },
    'Dataset 4': {
        'models': ['RF', 'XGB', 'LGBM', 'KNN', 'LR', 'DT', 'SVM', 'AdaBoost', 'GNB', 'ET'],
        'accuracies': [0.72, 0.76, 0.75, 0.73, 0.75, 0.67, 0.75, 0.68, 0.62, 0.79],
        'time': [0.09, 0.22, 0.09, 0.05, 0.07, 0.09, 0.02, 0.32, 0.02, 1.03],
    },
    'Dataset 5': {
        'models': ['RF', 'XGB', 'LGBM', 'KNN', 'LR', 'DT', 'SVM', 'AdaBoost', 'GNB', 'ET'],
        'accuracies': [0.63, 0.64, 0.65, 0.63, 0.58, 0.57, 0.63, 0.58, 0.59, 0.61],
        'time': [0.35, 1.26, 0.20, 0.06, 0.05, 0.02, 0.21, 0.08, 0.01, 0.29],
    },
}

# Color-blind friendly palette
cb_colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]
dataset_names = list(datasets.keys())
colors = [cb_colors[i % len(cb_colors)] for i in range(len(dataset_names))]

# Plot formatting
models = datasets['Dataset 1']['models']
accuracy_data = np.array([datasets[ds]['accuracies'] for ds in dataset_names]) * 100
time_data = np.array([datasets[ds]['time'] for ds in dataset_names])
x = np.arange(len(models))
width = 0.14

label_fs = 16
tick_fs = 13
bold = {'fontweight': 'bold'}

# -------------------------
# FIGURE 2A — ACCURACY (%)
# -------------------------
plt.figure(figsize=(16, 6))
for i, ds in enumerate(dataset_names):
    plt.bar(
        x + i*width, accuracy_data[i], width,
        label=ds, color=colors[i], edgecolor='black', linewidth=0.8
    )

plt.ylabel('Simple Accuracy (%)', fontsize=label_fs, **bold)
plt.xticks(x + width*(len(dataset_names)-1)/2, models, rotation=45, ha='right', fontsize=tick_fs, **bold)
plt.yticks(fontsize=tick_fs, **bold)
plt.ylim(0, 100)

plt.legend(
    title='Figure 2A. Classification Accuracy by Model and Dataset',
    ncol=len(dataset_names), fontsize=12, title_fontsize=12,
    loc='lower center', bbox_to_anchor=(0.5, 1.02), frameon=False
)

plt.tight_layout()
plt.savefig('figure2A_accuracy_cleaned.png', dpi=300, bbox_inches='tight')
plt.close()

# ------------------------------
# FIGURE 2B — TRAINING TIME (s)
# ------------------------------
plt.figure(figsize=(16, 6))
for i, ds in enumerate(dataset_names):
    plt.bar(
        x + i*width, time_data[i], width,
        label=ds, color=colors[i], edgecolor='black', linewidth=0.8
    )

plt.ylabel('Training Time (s)', fontsize=label_fs, **bold)
plt.xticks(x + width*(len(dataset_names)-1)/2, models, rotation=45, ha='right', fontsize=tick_fs, **bold)
plt.yticks(fontsize=tick_fs, **bold)

plt.legend(
    title='Figure 2B. Training Time by Model and Dataset',
    ncol=len(dataset_names), fontsize=12, title_fontsize=12,
    loc='lower center', bbox_to_anchor=(0.5, 1.02), frameon=False
)

plt.tight_layout()
plt.savefig('figure2B_time_cleaned.png', dpi=300, bbox_inches='tight')
plt.close()

