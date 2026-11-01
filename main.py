import pandas as pd
from sklearn import tree

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    balanced_accuracy_score,
    roc_auc_score, confusion_matrix
)
from scipy.stats import chi2_contingency
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data/hotel_bookings.csv')

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())


df = df.drop([
    "reservation_status",
    "reservation_status_date",
    "reserved_room_type", #analiz et
    "arrival_date_week_number",
    "days_in_waiting_list",
    "arrival_date_day_of_month"
], axis=1
)

df.drop_duplicates(inplace=True)


print(df['hotel'].value_counts())

print(df.groupby('hotel')['is_canceled'].value_counts())


features = [
    'lead_time',
    'stays_in_weekend_nights',
    'stays_in_week_nights',
    'adr',
    'total_of_special_requests'
]

for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]


print(df['is_canceled'].value_counts())

print(df.head())
print(df.info())
print(df.describe())

num_features = [
    "lead_time",
    "arrival_date_year",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests"
]

cat_features = [
    "hotel",
    "arrival_date_month",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
]
for col in cat_features:
    print(f"{col:30} unique : ", df[col].nunique())

corr = df[num_features + ['is_canceled']].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('results/graphs/correlation.png')

cramers_results = {}

for col in cat_features:
    table = pd.crosstab(
        df[col],
        df["is_canceled"]
    )

    chi2, p_value, dof, expected = chi2_contingency(table)

    n = table.sum().sum()
    r, c = table.shape

    cramers_v = np.sqrt(chi2 / (n * min(r - 1, c-1)))

    cramers_results[col] = cramers_v

cramers_df = pd.DataFrame(cramers_results, index=['is_canceled'])

plt.figure(figsize=(14, 3))

sns.heatmap(
    cramers_df,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    vmin=0,
    vmax=1
)

plt.title("Cramér's V: Categorical Features vs Is Canceled")
plt.xlabel("Categorical Features")
plt.ylabel("Target")

plt.tight_layout()
plt.savefig('results/graphs/cramers_v_heatmap.png')

#Minimum -> reserved_room_type (0.06)

X = df[num_features + cat_features]
y = df['is_canceled']

#arrival_date_week_number, arrival_date_day_of_month, days_in_waiting_list are useless


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown= 'infrequent_if_exist', min_frequency= 10))
])


num_scaled = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


pre_scaled = ColumnTransformer([
    ('num', num_scaled, num_features),
    ('cat', cat_pipeline, cat_features),
])


num_tree = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
])


pre_tree = ColumnTransformer([
    ('num', num_tree, num_features),
    ('cat', cat_pipeline, cat_features),
])


cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


knn = Pipeline([
    ("pre", pre_scaled),
    ("knn", KNeighborsClassifier())
])

knn_grid = {
    'knn__n_neighbors': [5, 15, 25], #3 5 7 9
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]
}

knn_search = RandomizedSearchCV( #gridden ferqlenmedi
        knn,
        knn_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        refit=True,
        n_iter=12,
        random_state = 42
)

#votingcls yarat her birine  modellerin hard voting ve soft voting et stacking

knn_search.fit(X_train, y_train)
print("Best params")  #th deyis oynat yoxla
print(knn_search.best_params_)
print("Mean ROC-AUC score: ", knn_search.best_score_)

knn_model = knn_search.best_estimator_

knn_predict = knn_model.predict(X_test)
knn_score = knn_model.predict_proba(X_test)[:, 1]

knn_accuracy = accuracy_score(y_test, knn_predict)
knn_roc_auc = roc_auc_score(y_test, knn_score)
knn_f1 = f1_score(y_test, knn_predict)
knn_recall = recall_score(y_test, knn_predict)
knn_precision = precision_score(y_test, knn_predict)
knn_balanced_accuracy = balanced_accuracy_score(y_test, knn_predict)
 #cls_rep
print("Accuracy: ", knn_accuracy)
print("AUC: ", knn_roc_auc)
print("F1: ", knn_f1)
print("Recall: ", knn_recall)
print("Precision: ", knn_precision)
print("Balanced Accuracy: ", knn_balanced_accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_test, knn_predict))


svm = Pipeline([
    ("pre", pre_scaled),
    ("svm", SVC(class_weight='balanced'))
])

svm_grid = [
    {
        'svm__kernel': ['linear'],
        'svm__C': [1, 10]
    },
    {
        'svm__kernel': ['rbf'],
        'svm__C': [0.1, 1, 10],
        'svm__gamma': ['scale', 0.01]
    }
]

svm_search = RandomizedSearchCV(
        svm,
        svm_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        refit=True,
        n_iter= 8,
        random_state = 42
)

svm_search.fit(X_train, y_train)
print("Best params")
print(svm_search.best_params_)
print("Mean ROC-AUC score: ", svm_search.best_score_)

svm_model = svm_search.best_estimator_

svm_predict = svm_model.predict(X_test)
svm_score = svm_model.decision_function(X_test)

svm_accuracy = accuracy_score(y_test, svm_predict)
svm_roc_auc = roc_auc_score(y_test, svm_score)
svm_f1 = f1_score(y_test, svm_predict)
svm_recall = recall_score(y_test, svm_predict)
svm_precision = precision_score(y_test, svm_predict)
svm_balanced_accuracy = balanced_accuracy_score(y_test, svm_predict)

print("Accuracy: ", svm_accuracy)
print("AUC: ", svm_roc_auc)
print("F1: ", svm_f1)
print("Recall: ", svm_recall)
print("Precision: ", svm_precision)
print("Balanced Accuracy: ", svm_balanced_accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_predict))


dt = Pipeline([
    ("pre", pre_tree),
    ("dt", DecisionTreeClassifier(
        class_weight='balanced',
        random_state=42
    ))
])

dtree_grid = {
    'dt__criterion': ['gini', 'entropy'],
    'dt__max_depth': [5, 10, None],
    'dt__min_samples_split': [2, 10],
}

dtree_search = RandomizedSearchCV(
        dt,
        dtree_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        refit=True,
        n_iter= 12,
        random_state = 42
)

dtree_search.fit(X_train, y_train)
print("Best params")
print(dtree_search.best_params_)
print("Mean ROC-AUC score: ", dtree_search.best_score_)

dtree_model = dtree_search.best_estimator_

dtree_predict = dtree_model.predict(X_test)
dtree_score = dtree_model.predict_proba(X_test)[:, 1]

dtree_accuracy = accuracy_score(y_test, dtree_predict)
dtree_roc_auc = roc_auc_score(y_test, dtree_score)
dtree_f1 = f1_score(y_test, dtree_predict)
dtree_recall = recall_score(y_test, dtree_predict)
dtree_precision = precision_score(y_test, dtree_predict)
dtree_balanced_accuracy = balanced_accuracy_score(y_test, dtree_predict)

print("Accuracy: ", dtree_accuracy)
print("AUC: ", dtree_roc_auc)
print("F1: ", dtree_f1)
print("Recall: ", dtree_recall)
print("Precision: ", dtree_precision)
print("Balanced Accuracy: ", dtree_balanced_accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_test, dtree_predict))

rf = Pipeline([
    ("pre", pre_tree),
    ("rf", RandomForestClassifier(
        class_weight='balanced',
        random_state=42
    ))
])

rf_grid = {
    'rf__n_estimators': [50, 100],
    'rf__max_depth': [5, 10, None],
    'rf__min_samples_split': [2, 10],
}


rf_search = RandomizedSearchCV(
        rf,
        rf_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        refit=True,
        n_iter = 12,
        random_state = 42
)

rf_search.fit(X_train, y_train)
print("Best params")
print(rf_search.best_params_)
print("Mean ROC-AUC score: ", rf_search.best_score_)

rf_model = rf_search.best_estimator_

rf_predict = rf_model.predict(X_test)
rf_score = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_predict)
rf_roc_auc = roc_auc_score(y_test, rf_score)
rf_f1 = f1_score(y_test, rf_predict)
rf_recall = recall_score(y_test, rf_predict)
rf_precision = precision_score(y_test, rf_predict)
rf_balanced_accuracy = balanced_accuracy_score(y_test, rf_predict)

print("Accuracy: ", rf_accuracy)
print("AUC: ", rf_roc_auc)
print("F1: ", rf_f1)
print("Recall: ", rf_recall)
print("Precision: ", rf_precision)
print("Balanced Accuracy: ", rf_balanced_accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_predict))

results = pd.DataFrame({
    "Model": ["KNN", "SVM", "DT", "RF"],
    "Accuracy": [knn_accuracy, svm_accuracy, dtree_accuracy, rf_accuracy],
    "Precision": [knn_precision, svm_precision, dtree_precision, rf_precision],
    "Recall": [knn_recall, svm_recall, dtree_recall, rf_recall],
    "F1": [knn_f1, svm_f1, dtree_f1, rf_f1],
    "Balanced Accuracy": [knn_balanced_accuracy, svm_balanced_accuracy, dtree_balanced_accuracy, rf_balanced_accuracy],
    "ROC-AUC" : [knn_roc_auc, svm_roc_auc, dtree_roc_auc, rf_roc_auc]
})

results = results.sort_values(
    by="ROC-AUC",
    ascending=False
)

print(results)

model_predictions = {
    'KNN' : knn_predict,
    'SVM' : svm_predict,
    'DT' : dtree_predict,
    'RF' : rf_predict
}

for name, prediction in model_predictions.items():
    cm = confusion_matrix(y_test, prediction)

    plt.figure(figsize = (5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    filename = name.lower().replace(' ', '_')
    plt.savefig(f'results/graphs/{filename}_confusion_matrix.png')

    plt.close()


#modelleri save et
#modelleri save et