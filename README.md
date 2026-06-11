🩺Breast Cancer Detection: Predictive Triage Model
A Logistic Regression pipeline predicting breast cancer malignancy with 95% Recall, designed as a second-opinion clinical triage tool.

📌 Project Overview
In medical diagnostics, raw accuracy can be fatally misleading due to the Accuracy Paradox. This binary classification pipeline focuses heavily on maximizing Recall (Sensitivity). While achieving 97.37% overall accuracy, the true engineering goal was driving False Negatives (missed diagnoses) to near-zero, ensuring this model serves as a reliable safety net for medical professionals.

🛠 Tech Stack
Core: Python, Pandas, NumPy

Machine Learning: Scikit-Learn (LogisticRegression, StandardScaler)

Visualization: Seaborn, Matplotlib

📂 Architecture & Workflow
Data Pipeline (pre.py): Processed 569 patient records across 30 numerical cell features. Applied an 80/20 train-test split before standardizing data to strictly prevent data leakage.

Model Training (pre.ipynb): Trained a baseline LogisticRegression model. This algorithm was specifically chosen for its high mathematical interpretability, a requirement in healthcare.

Evaluation: Evaluated via targeted medical metrics, specifically focusing on Type II Errors.

📊 Key Results
Confusion Matrix Metrics
Accuracy: 97.37%

Malignant Recall: 95%

False Negatives: Only 2 out of 114 test cases.

The model successfully caught 41/43 malignant tumors, proving highly viable as an automated diagnostic prompt.


