import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load Dataset
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
df = pd.read_csv(dataset_url, names=columns, index_col=0)

# Step 2: Data Exploration
print(df.head())
print(df.info())
print(df.describe())

# Step 3: Visualizing Class Distributions
sns.pairplot(df, hue='Type', diag_kind='kde')
plt.show()

# Step 4: Preprocessing
y = df['Type']
X = df.drop(columns=['Type'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Model Training & Evaluation
def train_and_evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {acc:.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Plot
    plt.figure(figsize=(6,5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

# Naïve Bayes
nb_model = GaussianNB()
train_and_evaluate_model(nb_model, "Naïve Bayes")

# LDA
lda_model = LinearDiscriminantAnalysis()
train_and_evaluate_model(lda_model, "LDA")

# QDA
qda_model = QuadraticDiscriminantAnalysis()
train_and_evaluate_model(qda_model, "QDA")

# Step 6: Compare Results & Draw Conclusions
print("Final Comparison:")
model_accuracies = {
    "Naïve Bayes": accuracy_score(y_test, nb_model.predict(X_test)),
    "LDA": accuracy_score(y_test, lda_model.predict(X_test)),
    "QDA": accuracy_score(y_test, qda_model.predict(X_test))
}
for model, acc in model_accuracies.items():
    print(f"{model}: {acc:.4f}")

best_model = max(model_accuracies, key=model_accuracies.get)
print(f"Best performing model: {best_model}")