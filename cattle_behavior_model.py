# cattle_behavior_model.py
# Core pipeline for cattle behavior classification using Random Forest

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

# Load accelerometer dataset (replace with actual file path)
data = pd.read_csv("data.csv")

# Segment data into 0.5-second windows
window_size = 15  # 1Hz sampling × 0.5 seconds
windowed_features = []
windowed_labels = []

for i in range(0, len(data) - window_size + 1, window_size):
    window = data.iloc[i:i + window_size]
    if len(window) == window_size:
        features = {
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
        windowed_features.append(features)
        windowed_labels.append(window['behavior'].iloc[0])

# Convert to arrays
X = pd.DataFrame(windowed_features)
y = np.array(windowed_labels)

# Impute and scale
X = SimpleImputer(strategy='mean').fit_transform(X)
X = StandardScaler().fit_transform(X)

# Handle class imbalance
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()