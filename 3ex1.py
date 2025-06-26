import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nTarget names:", data.target_names)

plt.figure(figsize=(6,4))
sns.countplot(x='target', data=df)
plt.title("Class Distribution (0 = Malignant, 1 = Benign)")
plt.xticks(ticks=[0,1], labels=['Malignant', 'Benign'])
plt.show()

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nEvaluation Metrics:")
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Confusion Matrix:\n", conf_matrix)
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title("Top 10 Important Features")
plt.show()

print("\nFeature Importance Explanation:")
print("Random Forest uses decision trees, which split the data based on features that best reduce impurity (Gini/Entropy).")
print("The importance scores show how often a feature was used in splitting and how much it improved the prediction.")
print("This helps us understand which features contribute most to classifying malignant vs benign tumors.")

new_sample = np.array([[14.0, 20.0, 90.0, 600.0, 0.1, 0.12, 0.13, 0.09, 0.18, 0.06,
                        0.5, 1.0, 3.0, 40.0, 0.005, 0.025, 0.04, 0.015, 0.02, 0.004,
                        16.0, 30.0, 110.0, 800.0, 0.14, 0.25, 0.3, 0.15, 0.25, 0.08]])
result = model.predict(new_sample)
print("Prediction:", "Benign" if result[0] == 1 else "Malignant")
