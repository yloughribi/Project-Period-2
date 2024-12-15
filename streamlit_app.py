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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

st.title("☝️🤓End Project Presentation 📚")
st.title('Who is better at feature selection? Man🧔 or Machine🤖?')
st.write(
    "*Presented by Malaika Paddison and Yahya Loughribi.*"
)

st.write("# Research Questions:")
st.write("Which variables have an impact on predicting the target 'ANYCHD'?")
st.write("Which variables have the *most* impact on the prediction of target 'ANYCHD'?")

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
mask = np.logical_and(corr > -0.1, corr < 0.1)

fig, ax = plt.subplots(figsize = (35, 25))
sns.heatmap(corr, 
            cmap='Blues', 
            annot=True, 
            fmt=".2f", 
            linewidths=.5, 
            cbar_kws={"shrink": .8},
            ax = ax)


plt.title('Correlation Heatmap with Filtered Values', fontsize=20)

st.pyplot(fig)

model_features = ['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE', 'CIGPDAY', 'BMI', 'DIABETES', 'BPMEDS', 'GLUCOSE', 'SEX']
target_variable = 'ANYCHD'

df = df.dropna(subset=model_features + [target_variable])

X = df[model_features]
y = df[target_variable]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model_sel = st.selectbox('Select the classifier of the model', ('Logistic Regression', 'Random Forest', 'Gradient Boost', 'SVM'))

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

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
y_pred_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_prob)
"Model Performance:"
f"Accuracy: {accuracy:.2f}"
f"ROC-AUC: {roc_auc:.2f}"
"\nClassification Report:"
st.dataframe(classification_report(y_test, y_pred, output_dict=True))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)
auc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds})
auc_data.to_csv('auc_data.csv', index=False)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
'\n'
f"Accuracy: {accuracy:.2f}"
f"Precision: {precision:.2f}"
f"Recall: {recall:.2f}"
f"F1 Score: {f1:.2f}"
'\n'
if model_sel == 'Random Forest':
    feature_importances = pd.DataFrame(
    {'Feature': model_features, 'Importance': model.feature_importances_}
    ).sort_values(by='Importance', ascending=False)
    "\nFeature Importance:"
    st.dataframe(feature_importances)
fig, ax = plt.subplots(figsize = (8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)
st.pyplot(fig)


st.title('Machine 🦾🤖')
#Filter Feature Selection
'# Filter Feature selection'
'ANOVA'
numeric_df = numeric_df.dropna()
X = numeric_df.drop(['ANYCHD', 'TIMEHYP', 'TIMECHD', 'TIMEMIFC', 'TIMEMI', 'TIMEDTH', 'TIMEAP', 'TIMESTRK', 'TIMECVD', 'ANGINA', 'MI_FCHD', 'CVD', 'HOSPMI', 'PREVCHD', 'PREVAP', 'DEATH', 'PREVMI', 'educ', 'PREVSTRK', 'PERIOD', 'PREVHYP'], axis = 1)
y = numeric_df['ANYCHD']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

best_features = SelectKBest(score_func = f_classif,k = 'all')

fit = best_features.fit(X_train,y_train)

featureScores = pd.DataFrame(data = fit.scores_,index = list(X.columns),columns = ['ANOVA Score'])

fig, ax = plt.subplots(figsize = (4, 8))

sns.heatmap(featureScores.sort_values(ascending = False,by = 'ANOVA Score'),annot = True, ax = ax)
plt.title('Filter selection of features (ANOVA)')

st.pyplot(fig)

#Wrapper feature selection
'# Wrapper feature selection'
estimator = LogisticRegression(class_weight='balanced', max_iter=1000)
n_features_to_select = 5
selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(X_train, y_train)

featureSupport = pd.DataFrame(data = selector.ranking_,index = list(X.columns),columns = ['Feature ranking'])

fig, ax = plt.subplots(figsize = (5, 5))
sns.heatmap(featureSupport.sort_values(ascending = True,by = 'Feature ranking'),annot = True)
plt.title('Wrapper selection of features')

st.pyplot(fig) 

#Embedded feature selection
'# Embedded Feature selection'

logreg_l1 = LogisticRegression(class_weight='balanced', max_iter=1000,
                            penalty='l1', solver='liblinear')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg_l1.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = logreg_l1.predict(X_test_scaled)
st.dataframe(classification_report(y_test, y_pred, output_dict=True))

ConfusionMatrixDisplay.from_estimator(logreg_l1, X_test_scaled, y_test)
st.pyplot(plt.gcf())

numeric_df = numeric_df[['SYSBP', 'HYPERTEN', 'SEX', 'AGE', 'DIABETES', 'ANYCHD']]
X_machine = numeric_df.drop('ANYCHD', axis = 1)
y_machine = numeric_df['ANYCHD']
scaler = StandardScaler()
X_machine_scaled = scaler.fit_transform(X_machine)

X_train_machine, X_test_machine, y_train_machine, y_test_machine = train_test_split(X_machine_scaled, y_machine, test_size=0.2, random_state=42)

model_sel = st.selectbox('Select the classifier of the model', ('Logistic Regression (Machine)', 'Random Forest (Machine)', 'Gradient Boost (Machine)', 'SVM (Machine)'))

if model_sel == 'Logistic Regression (Machine)':
    model = LogisticRegression(random_state=42)
elif model_sel == 'Random Forest (Machine)':
    model = RandomForestClassifier(
    n_estimators=500,
    max_depth=2,
    random_state=42,
    min_samples_split=2,
    min_samples_leaf=1)
elif model_sel == 'Gradient Boost (Machine)':
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
elif model_sel == 'SVM (Machine)':
    model = SVC(probability=True, kernel='rbf', random_state=42)

model.fit(X_train_machine, y_train_machine)

# Make predictions
y_pred_machine = model.predict(X_test_machine)
# Evaluate the model
accuracy = accuracy_score(y_test_machine, y_pred_machine)
y_pred_prob_machine = model.predict_proba(X_test_machine)[:, 1]
roc_auc = roc_auc_score(y_test_machine, y_pred_prob_machine)
"Model Performance:"
f"Accuracy: {accuracy:.2f}"
f"ROC-AUC: {roc_auc:.2f}"
"\nClassification Report:"
st.dataframe(classification_report(y_test_machine, y_pred_machine, output_dict=True))

st.title('Model Comparison Man vs Machine 🏋🏻‍♀️')

'Please select your contestant(s)'
man_check = st.checkbox('Man', True)
machine_check = st.checkbox('Machine')

if man_check == True:
    model_features = ['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE', 'CIGPDAY', 'BMI', 'DIABETES', 'BPMEDS', 'GLUCOSE', 'SEX']
    target_variable = 'ANYCHD'
    
    df = df.dropna(subset=model_features + [target_variable])
    X = df[model_features]
    y = df[target_variable]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
elif machine_check == True:
    numeric_df = numeric_df[['SYSBP', 'HYPERTEN', 'SEX', 'AGE', 'DIABETES', 'ANYCHD']]
    X_machine = numeric_df.drop('ANYCHD', axis = 1)
    y_machine = numeric_df['ANYCHD']
    scaler = StandardScaler()
    X_machine_scaled = scaler.fit_transform(X_machine)
    X_train_machine, X_test_machine, y_train_machine, y_test_machine = train_test_split(X_machine_scaled, y_machine, test_size=0.2, random_state=42)
elif man_check == False and machine_check == False:
    'No contestant is selected.'
    '\nPlease select at least one or select both options to compare'
'\n'
'\n'
'\nPlease select the models you wish to compare'
LR_check = st.checkbox('Logistic Regression')
RF_check = st.checkbox('Random Forest')
GB_check = st.checkbox('Gradient Boost')
SVM_check = st.checkbox('SVM')