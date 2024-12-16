import streamlit as st
import pandas as pd
import numpy as np
import time
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
import statsmodels.api as sm
import scipy.stats as stats
import shapely
import shap
from sklearn import tree
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

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

st.subheader('Q-Q plots')

# Sample data for four columns
data1 = df['TOTCHOL']
data2 = df['BMI']
data3 = df['HEARTRTE']
data4 = df['GLUCOSE']

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Q-Q plot for each dataset
stats.probplot(data1, dist="norm", plot=axes[0, 0])
axes[0, 0].set_title("Q-Q Plot for TOTCHOL")

stats.probplot(data2, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title("Q-Q Plot for BMI")

stats.probplot(data3, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title("Q-Q Plot for HEARTRTE")

stats.probplot(data4, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Q-Q Plot for GLUCOSE")

# Adjust layout and show
plt.tight_layout()
st.pyplot(fig)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
# Q-Q plot for each dataset
stats.probplot(df['SYSBP'], dist="norm", plot=axes[0, 0])
axes[0, 0].set_title("Q-Q Plot for SYSBP")

stats.probplot(df['DIABP'], dist="norm", plot=axes[0, 1])
axes[0, 1].set_title("Q-Q Plot for DIABP")

stats.probplot(df['HDLC'], dist="norm", plot=axes[1, 0])
axes[1, 0].set_title("Q-Q Plot for HDLC")

stats.probplot(df['LDLC'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Q-Q Plot for LDLC")
plt.tight_layout()
st.pyplot(fig) 

st.subheader('Heat Map')
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
st.title('The "Man-made" models')
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
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
y_pred_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_prob)
"Model Performance:"
f"Accuracy: {accuracy:.2f}"
f"ROC-AUC: {roc_auc:.2f}"
"\nClassification Report:"
st.dataframe(classification_report(y_test, y_pred, output_dict=True))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
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

st.subheader('Odds ratio and 95% Confidence Interval')
X_train_Sm_m = sm.add_constant(X_train_machine)
feature_names = ['const'] + list(X_machine.columns)  # Include the constant term

# Fit Logistic Regression (Statsmodels)
model_sm = sm.Logit(y_train_machine, X_train_Sm_m).fit()

# Get parameters and confidence intervals
params = model_sm.params
conf = model_sm.conf_int()
conf['Odds Ratio'] = params
conf.columns = ['2.5%', '97.5%', 'Odds Ratio']

# Convert log odds to odds ratios
odds = pd.DataFrame(np.exp(conf))
odds['pvalues'] = model_sm.pvalues
odds['significance'] = ['significant' if pval <= 0.05 else 'not significant' for pval in model_sm.pvalues]

# Assign meaningful feature names
odds.index = feature_names

# Display odds ratios
st.dataframe(odds)

# Fit Logistic Regression (sklearn)
model = LogisticRegression(random_state=42)
model.fit(X_train_machine, y_train_machine)

# Predict and evaluate
y_pred_machine = model.predict(X_test_machine)
accuracy_lr_m = accuracy_score(y_test_machine, y_pred_machine)
y_pred_prob_machine = model.predict_proba(X_test_machine)[:, 1]

# Display odds ratios for sklearn model
coeff_parameter = pd.DataFrame(
    np.transpose(np.exp(model.coef_)),
    index=X_machine.columns,
    columns=['Odds ratio']
)
st.dataframe(coeff_parameter)

fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True, figsize=(6, 4), dpi=150)
for idx, row in odds.iloc[::-1].iterrows():
    ci = [[row['Odds Ratio'] - row[::-1]['2.5%']], [row['97.5%'] - row['Odds Ratio']]]
    if row['significance'] == 'significant':
        plt.errorbar(x=[row['Odds Ratio']], y=[row.name], xerr=ci,
            ecolor='tab:red', capsize=3, linestyle='None', linewidth=1, marker="o", 
                     markersize=5, mfc="tab:red", mec="tab:red")
    else:
        plt.errorbar(x=[row['Odds Ratio']], y=[row.name], xerr=ci,
            ecolor='tab:gray', capsize=3, linestyle='None', linewidth=1, marker="o", 
                     markersize=5, mfc="tab:gray", mec="tab:gray")
plt.axvline(x=1, linewidth=0.8, linestyle='--', color='black')
plt.tick_params(axis='both', which='major', labelsize=8)
plt.xlabel('Odds Ratio and 95% Confidence Interval', fontsize=8)
plt.tight_layout()
st.pyplot(fig)

st.subheader('Decision Tree')
model = tree.DecisionTreeClassifier(random_state=1000, max_depth=4, min_samples_leaf=1)
model.fit(X_train_machine, y_train_machine)

# Predictions and Accuracy
prediction = model.predict(X_test_machine)
accuracy = accuracy_score(y_test_machine, prediction)
roc_auc = roc_auc_score(y_test_machine, y_pred_prob_machine)
st.write(f"Accuracy for the decision tree is:  {accuracy:.2f}")
st.write(f"ROC score for the decision tree is: {roc_auc:.2f}")
"\nClassification Report:"
st.dataframe(classification_report(y_test_machine, y_pred_machine, output_dict=True))
fpr, tpr, thresholds = roc_curve(y_test_machine, y_pred_prob_machine)
auc_data_tree = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds})
auc_data_tree.to_csv('auc_data_tree.csv', index=False)
text_representation = tree.export_text(model, feature_names=X_machine.columns.tolist())
# Plotting the Decision Tree
fig = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(
    model,
    feature_names=X_machine.columns.tolist(),  # Feature names from DataFrame
    class_names=[str(cls) for cls in y_machine.unique()],  # Class names as strings
    filled=True
)
st.pyplot(fig)

fig_roctree, ax = plt.subplots(figsize = (8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)
st.pyplot(fig_roctree)

feature_names = X_machine.columns.tolist()
importances = model.feature_importances_  # Feature importances from the tree

# Convert importances to a pandas Series
forest_importances = pd.Series(importances, index=feature_names)

# Plot feature importances
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
forest_importances.plot.barh(ax=ax)  # Horizontal bar plot
ax.set_title("Feature importances using MDI")
ax.set_xlabel("Mean decrease in impurity")
fig.tight_layout()

st.pyplot(fig)

result = permutation_importance(
    model, X_test_machine, y_test_machine, n_repeats=10, random_state=42, n_jobs=2
)
forest_importances = pd.Series(result.importances_mean, index=feature_names)
fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True, figsize=(6, 4), dpi=150)
forest_importances.plot.barh(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
st.pyplot(fig)


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_machine)
# Select shap values for the desired class (e.g., class 0 or 1) or for all classes
# Replace 1 with the desired class index if you need a specific class
# shap_values_for_class = shap_values[1]  # Selecting SHAP values for the second class (index 1) - This was the problem
shap_values_for_class = shap_values[:, :, 1] # Selecting SHAP values for the second class (index 1) for all samples
#shap_values_for_class = shap_values # For all classes (might require different plotting function)

if not isinstance(X_test_machine, pd.DataFrame):
    X_test_machine = pd.DataFrame(X_test_machine, columns=feature_names)

fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True, figsize=(6, 4), dpi=150)
shap.summary_plot(shap_values_for_class, X_test_machine, plot_type="dot")
st.pyplot(fig)

numeric_df = numeric_df[['SYSBP', 'HYPERTEN', 'SEX', 'AGE', 'DIABETES', 'ANYCHD']]
X_machine = numeric_df.drop('ANYCHD', axis = 1)
y_machine = numeric_df['ANYCHD']
scaler = StandardScaler()
X_machine_scaled = scaler.fit_transform(X_machine)

X_train_machine, X_test_machine, y_train_machine, y_test_machine = train_test_split(X_machine_scaled, y_machine, test_size=0.2, random_state=42)

st.title('The "Machine-made" models')
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
fpr, tpr, thresholds = roc_curve(y_test_machine, y_pred_prob_machine)
auc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds})
auc_data.to_csv('auc_data.csv', index=False)
accuracy = accuracy_score(y_test_machine, y_pred_machine)
precision = precision_score(y_test_machine, y_pred_machine)
recall = recall_score(y_test_machine, y_pred_machine)
f1 = f1_score(y_test_machine, y_pred_machine)
'\n'
f"Accuracy: {accuracy:.2f}"
f"Precision: {precision:.2f}"
f"Recall: {recall:.2f}"
f"F1 Score: {f1:.2f}"
'\n'
if model_sel == 'Random Forest (Machine)':
    model_features = ['SYSBP', 'HYPERTEN', 'SEX', 'AGE', 'DIABETES']
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

# Man all models
model_features = ['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE', 'CIGPDAY', 'BMI', 'DIABETES', 'BPMEDS', 'GLUCOSE', 'SEX']
target_variable = 'ANYCHD'
df = df.dropna(subset=model_features + [target_variable])
X = df[model_features]
y = df[target_variable]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
y_pred_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_prob)
roc_auc_lr = roc_auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds})
auc_data.to_csv('auc_data_lg.csv', index=False)

model = RandomForestClassifier(
n_estimators=500,
max_depth=2,
random_state=42,
min_samples_split=2,
min_samples_leaf=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
y_pred_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_prob)
roc_auc_rf = roc_auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds})
auc_data.to_csv('auc_data_rf.csv', index=False)

model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
y_pred_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_prob)
roc_auc_gb = roc_auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds})
auc_data.to_csv('auc_data_gb.csv', index=False)

model = SVC(probability=True, kernel='rbf', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
y_pred_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_prob)
roc_auc_svm = roc_auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds})
auc_data.to_csv('auc_data_svm.csv', index=False)

auc_data_lr = pd.read_csv('auc_data_lg.csv')  # Logistic Regression
auc_data_rf = pd.read_csv('auc_data_rf.csv')  # Random Forest
auc_data_xgb = pd.read_csv('auc_data_gb.csv')  # XGBoost
auc_data_svm = pd.read_csv('auc_data_svm.csv')  # SVM

# Extract FPR and TPR for each model
fpr_lr, tpr_lr = auc_data_lr['FPR'], auc_data_lr['TPR']
fpr_rf, tpr_rf = auc_data_rf['FPR'], auc_data_rf['TPR']
fpr_xgb, tpr_xgb = auc_data_xgb['FPR'], auc_data_xgb['TPR']
fpr_svm, tpr_svm = auc_data_svm['FPR'], auc_data_svm['TPR']

fig1, ax = plt.subplots(figsize = (10, 8))
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression', color='blue', lw=2)
plt.plot(fpr_rf, tpr_rf, label='Random Forest', color='green', lw=2)
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost', color='orange', lw=2)
plt.plot(fpr_svm, tpr_svm, label='SVM', color='red', lw=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Random guess')

# Plot formatting
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Comparison of ROC Curves for All Models', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)

# Machine all plots
numeric_df = numeric_df[['SYSBP', 'HYPERTEN', 'SEX', 'AGE', 'DIABETES', 'ANYCHD']]
X_machine = numeric_df.drop('ANYCHD', axis = 1)
y_machine = numeric_df['ANYCHD']
scaler = StandardScaler()
X_machine_scaled = scaler.fit_transform(X_machine)
X_train_machine, X_test_machine, y_train_machine, y_test_machine = train_test_split(X_machine_scaled, y_machine, test_size=0.2, random_state=42)

model = LogisticRegression(random_state=42)
model.fit(X_train_machine, y_train_machine)
y_pred_machine = model.predict(X_test_machine)
accuracy_lr_m = accuracy_score(y_test_machine, y_pred_machine)
y_pred_prob_machine = model.predict_proba(X_test_machine)[:, 1]
roc_auc = roc_auc_score(y_test_machine, y_pred_prob_machine)
roc_lr_m = roc_auc
fpr, tpr, thresholds = roc_curve(y_test_machine, y_pred_prob_machine)
auc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds})
auc_data.to_csv('auc_data_lg_machine.csv', index=False)

model = RandomForestClassifier(
n_estimators=500,
max_depth=2,
random_state=42,
min_samples_split=2,
min_samples_leaf=1)
model.fit(X_train_machine, y_train_machine)
y_pred_machine = model.predict(X_test_machine)
accuracy_rf_m = accuracy_score(y_test_machine, y_pred_machine)
y_pred_prob_machine = model.predict_proba(X_test_machine)[:, 1]
roc_auc = roc_auc_score(y_test_machine, y_pred_prob_machine)
roc_rf_m = roc_auc
fpr, tpr, thresholds = roc_curve(y_test_machine, y_pred_prob_machine)
auc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds})
auc_data.to_csv('auc_data_rf_machine.csv', index=False)

model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_machine, y_train_machine)
y_pred_machine = model.predict(X_test_machine)
accuracy_gb_m = accuracy_score(y_test_machine, y_pred_machine)
y_pred_prob_machine = model.predict_proba(X_test_machine)[:, 1]
roc_auc = roc_auc_score(y_test_machine, y_pred_prob_machine)
roc_gb_m = roc_auc
fpr, tpr, thresholds = roc_curve(y_test_machine, y_pred_prob_machine)
auc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds})
auc_data.to_csv('auc_data_gb_machine.csv', index=False)

model = SVC(probability=True, kernel='rbf', random_state=42)
model.fit(X_train_machine, y_train_machine)
y_pred_machine = model.predict(X_test_machine)
accuracy_svm_m = accuracy_score(y_test_machine, y_pred_machine)
y_pred_prob_machine = model.predict_proba(X_test_machine)[:, 1]
roc_auc = roc_auc_score(y_test_machine, y_pred_prob_machine)
roc_auc_svm_m = roc_auc
fpr, tpr, thresholds = roc_curve(y_test_machine, y_pred_prob_machine)
auc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds})
auc_data.to_csv('auc_data_svm_machine.csv', index=False)

auc_data_lr_m = pd.read_csv('auc_data_lg_machine.csv')  # Logistic Regression
auc_data_rf_m = pd.read_csv('auc_data_rf_machine.csv')  # Random Forest
auc_data_xgb_m = pd.read_csv('auc_data_gb_machine.csv')  # XGBoost
auc_data_svm_m = pd.read_csv('auc_data_svm_machine.csv')  # SVM

fpr_lr_m, tpr_lr_m = auc_data_lr_m['FPR'], auc_data_lr_m['TPR']
fpr_rf_m, tpr_rf_m = auc_data_rf_m['FPR'], auc_data_rf_m['TPR']
fpr_xgb_m, tpr_xgb_m = auc_data_xgb_m['FPR'], auc_data_xgb_m['TPR']
fpr_svm_m, tpr_svm_m = auc_data_svm_m['FPR'], auc_data_svm_m['TPR']

fig2, ax = plt.subplots(figsize = (10, 8))
plt.plot(fpr_lr_m, tpr_lr_m, label='Logistic Regression', color='blue', lw=2)
plt.plot(fpr_rf_m, tpr_rf_m, label='Random Forest', color='green', lw=2)
plt.plot(fpr_xgb_m, tpr_xgb_m, label='XGBoost', color='orange', lw=2)
plt.plot(fpr_svm_m, tpr_svm_m, label='SVM', color='red', lw=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Random guess')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Comparison of ROC Curves for All Models', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)

st.title('Comparing ROC curves of all the models (per Category)')
st.subheader('Select category')
cat = st.selectbox('', ('Man', 'Machine', 'Both'))

if cat == 'Man':
    st.pyplot(fig1)
elif cat == 'Machine':
    st.pyplot(fig2)
elif cat == 'Both':
    st.pyplot(fig1)
    st.pyplot(fig2)


st.title('Model Comparison Man vs Machine 🏋🏻‍♀️')
'\n'
st.subheader('Select the models to compare')
sel_lr = st.checkbox('Logistic Regression')
sel_rf = st.checkbox('Random Forest Regression')
sel_gb = st.checkbox('Gradient Boosting')
sel_svm = st.checkbox('SVM')

fig_com, ax = plt.subplots(figsize=(10, 8))
ax.clear()  # Ensure the plot is cleared before starting a new one

# Add ROC curves for selected models
if sel_lr:
    ax.plot(fpr_lr, tpr_lr, label='Logistic Regression (Man)', color='blue', lw=2)
    ax.plot(fpr_lr_m, tpr_lr_m, label='Logistic Regression (Machine)', color='red', lw=2)
if sel_rf:
    ax.plot(fpr_rf, tpr_rf, label='Random Forest (Man)', color='green', lw=2)
    ax.plot(fpr_rf_m, tpr_rf_m, label='Random Forest (Machine)', color='olive', lw=2)
if sel_gb:
    ax.plot(fpr_xgb, tpr_xgb, label='Gradient Boosting (Man)', color='pink', lw=2)
    ax.plot(fpr_xgb_m, tpr_xgb_m, label='Gradient Boosting (Machine)', color='purple', lw=2)
if sel_svm:
    ax.plot(fpr_svm, tpr_svm, label='SVM (Man)', color='orange', lw=2)
    ax.plot(fpr_svm_m, tpr_svm_m, label='SVM (Machine)', color='brown', lw=2)

# Add common elements to the plot
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Random guess')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.legend(loc="lower right", fontsize=12)
ax.grid(alpha=0.3)

# Display the plot in Streamlit
if st.button('Press to compare'):
    st.pyplot(fig_com)


st.title('Conclusion')
'Feature selection is an important process as features might have direct correlations with the target variable but multiple features can have a correlation with each other.'
'It is also important to test for feature importance as some feature might have a larger effect than others.'
'\n'
'In our case, the feature selection has lead to certain features being added and other features being left out.'
'Furthermore, it turns out that for all models having a higher accuracy than those with hand-selected features.'
'Important to note is that the only metric in which the hand-selected features prevailed is in the ROC-AUC score of the SVM model with the "Man-made" model having a score of 0.62 and the "Machine" model a score of 0.60'
'\n'
'This project has highlighted the importance of feature selection and testing for feature importance to get the best models'