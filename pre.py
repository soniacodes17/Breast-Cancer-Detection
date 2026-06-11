import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

#loading the sheet frim the master excel file
file_path = r'C:\Users\s o n i a\f o l d e r s\p a d h a i\s h o n a\coding projects\breast cancer detection\65c0bae494f27_breast_cancer_project\Breast Cancer\All Files & Datasets\Dataset_master.xlsx'

print("Loading data...(this might take a few seconds)")

df = pd.read_excel(file_path, sheet_name='Breast Cancer Detection Classif')
print("Data Loaded Successfully!")

#checking for massive imbalances or completely empty columns
print("\nDATASET INFO: ")
print(df.info())
print("\nDiagnosis counts (1 = Malignant, 0 = Benign): ")
print(df['diagnosis'].value_counts())

#isolate the dependent variable (y) and independent vars (X)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values #only the last column

#imputation for NaN values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

X = imputer.fit_transform(X)

#train-test split
#splitting before feature scaling to prevent any data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

#feature scaling - ensures all features are treated equally
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) #apply the same scale w/o learning any new patterns

#saving processed arrays for phase2 - model training
save_path = 'processed_cancer_data.npz'
np.savez(save_path, X_train = X_train, X_test= X_test, y_train = y_train, y_test=y_test)

print(f"\nPhase1 Complete! Cleaned data saved successfully to {save_path}")

