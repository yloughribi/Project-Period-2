import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.impute import KNNImputer
import ipywidgets as widgets
from IPython.display import display
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier
from sklearn.svm import SVC
import catboost as cb
from catboost import CatBoostClassifier
from catboost import CatBoostRegressor 
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

st.title("☝️🤓End Project Presentation 📚")
st.write(
    "Presented by Malaika Paddison and Yahya Loughribi."
)

df = pd.read_csv('https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv')

numeric_df = df.select_dtypes(include='number')
correlation_matrix = numeric_df.corr()

# Identify correlations greater than 0.3 (absolute value)
threshold = 0.3
correlations_with_ANYCHD = correlation_matrix['ANYCHD'][correlation_matrix['ANYCHD'].abs() > threshold]

# Set display options to show all columns
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Set width to avoid line wrapping

# Get columns with missing values and their count
missing_info = df.isnull().sum()
missing_info = missing_info[missing_info > 0]

# removing people with over 10% of data missing. 39 columns = over 4 values --> remove
threshold = df.shape[1] - 4

# Drop rows with more than 3 missing values
df = df.dropna(thresh=threshold)
columns_for_imputation = ['TOTCHOL', 'AGE', 'SYSBP', 'DIABP', 'BMI']
knn_imputer = KNNImputer(n_neighbors=5)
df_to_impute = df[columns_for_imputation]
df_imputed = knn_imputer.fit_transform(df_to_impute)
df[columns_for_imputation] = df_imputed

df = df[df['TOTCHOL'] <= 300]

# impute Heartrate with mean (normal distribution)
df.loc[(df['HEARTRTE'].isnull()) | (df['HEARTRTE'] == (220 - df['AGE'])), 'HEARTRTE'] = df['HEARTRTE'].mean()

# Create a mask for rows where glucose is missing or out of range
mask = df['GLUCOSE'].isnull() | (df['GLUCOSE'] < 70) | (df['GLUCOSE'] > 200)

# Select only the rows that need imputation
columns_to_impute = df.loc[mask, ['GLUCOSE']]

# Apply KNNImputer on the selected column
imputer = KNNImputer(n_neighbors=3)
imputed_column = imputer.fit_transform(columns_to_impute)

# Replace the imputed values in the original DataFrame
df.loc[mask, 'GLUCOSE'] = imputed_column

# Create a mask for rows where CIGPDAY is missing or below the threshold (e.g., < 1)
mask = df['CIGPDAY'].isnull() | (df['CIGPDAY'] < 1)

# Select only the rows that need imputation
columns_to_impute = df.loc[mask, ['CIGPDAY']]

# Apply KNNImputer on the selected column
imputer = KNNImputer(n_neighbors=3)
imputed_column = imputer.fit_transform(columns_to_impute)

# Replace the imputed values in the original DataFrame
df.loc[mask, 'CIGPDAY'] = imputed_column

# Impute values based on the condition --> if SYSBP 160, then BPMEDS == 1.0, else BPMEDS == 0.0
df['BPMEDS'] = np.where(
    df['BPMEDS'].isnull() & (df['SYSBP'] > 150.0),  # Condition for SYSBP == 160.0
    1.0,  # Value to assign if condition is True
    np.where(
        df['BPMEDS'].isnull(),  # Condition for null values
        0.0,  # Value to assign if null but SYSBP is not 160.0
        df['BPMEDS']  # Keep original value if not null
    )
)

# impute HDLC with median (normal distribution)
df.loc[df['HDLC'].isnull() | (df['HDLC'] > 60.0), 'HDLC'] = df['HDLC'].median()

# impute HDLC with median (normal distribution)
df.loc[df['LDLC'].isnull() | (df['LDLC'] > 190.0), 'LDLC'] = df['LDLC'].median()

missing_info = df.isnull().sum()
missing_info = missing_info[missing_info > 0]

numeric_df = df.select_dtypes(include='number').drop(columns=['RANDID', 'TIME'])
corr = numeric_df.corr()
mask = np.logical_and(corr > -0.15, corr < 0.15)

fig, ax = plt.subplots(figsize = (20, 20))
sns.heatmap(corr, 
            mask=mask, 
            cmap='Blues', 
            annot=True, 
            fmt=".2f", 
            linewidths=.5, 
            cbar_kws={"shrink": .8},
            ax = ax)


plt.title('Correlation Heatmap with Filtered Values', fontsize=20)

st.pyplot(fig)

model_features = ['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE', 'CIGPDAY', 'BMI', 'DIABETES', 'BPMEDS', 'GLUCOSE']
target_variable = 'ANYCHD'

df = df.dropna(subset=model_features + [target_variable])

X = df[model_features]
y = df[target_variable]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model_sel = st.selectbox('Select the classifier of the model', ('Logistic Regression', 'Random Forest', 'Gradient Boost', 'SVM', 'Catboost'))

if model_sel == 'Logistic Regression':
    model = LogisticRegression(random_state=42)
elif model_sel == 'Random Forest':
    model = RandomForestClassifier(
    n_estimators=500,
    max_depth=2,
    random_state=42,
    min_samples_split=2,
    min_samples_leaf=1)
elif model_sel == 'Gradient Boost':
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
elif model_sel == 'SVM':
    model = SVC(probability=True, kernel='rbf', random_state=42)
elif model_sel == 'Catboost':
    model = CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, verbose=0)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)


if model_sel == 'Catboost':
    "Model Performance:"
    f"Accuracy: {accuracy:.2f}"
    "\nClassification Report:"
    st.dataframe(classification_report(y_test, y_pred, output_dict=True))
else:
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    "Model Performance:"
    f"Accuracy: {accuracy:.2f}"
    f"ROC-AUC: {roc_auc:.2f}"
    "\nClassification Report:"
    st.dataframe(classification_report(y_test, y_pred, output_dict=True))
