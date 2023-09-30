from time import time

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.utils.extmath import density
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd

np.random.seed(42)

#load dataset
data_train = pd.read_csv('data_train.csv')
data_test = pd.read_csv('data_test.csv')

#extract features with vectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.15, min_df=3, stop_words="english", ngram_range=(1,1), norm="l1", smooth_idf=False)

X_train = vectorizer.fit_transform(data_train["truncated"].apply(lambda x: np.str_(x)))
X_test = vectorizer.transform(data_test['truncated'].apply(lambda x: np.str_(x)))

y_train, y_test = data_train["target"], data_test['target']

#extract feature names and target names (labels)
feature_names = vectorizer.get_feature_names_out()
target_names = fetch_20newsgroups(subset="train",shuffle=True,random_state=42).target_names



def benchmark(clf):
    #fit model
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print(f"train time: {train_time:.3}s")

    #test model
    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print(f"test time:  {test_time:.3}s")

    score = metrics.accuracy_score(y_test, pred)
    print(f"accuracy:   {score:.7}")

    print(f"dimensionality: {clf.coef_.shape[1]}")
    print(f"density: {density(clf.coef_)}")
    print()

    return clf

#train and test model
print("Linear SVC")
clf = benchmark(LinearSVC(C=30, dual=True, max_iter=3000))

print(clf)

#extract coefficients
coef = clf.coef_

#normalize coefficients to range [-1,1]
min_val = coef.min()
max_val = coef.max()
normalized_coef = 2 * (coef - min_val) / (max_val - min_val) - 1

#print(normalized_coef.min())
#print(normalized_coef.max())



class_dict = {"feature":feature_names}
target_names.sort()

for i in range(20):
    class_dict[target_names[i]]=normalized_coef[i]

# get and store feature coefficients over whole vocab
all_coef = pd.DataFrame(class_dict)
all_coef.to_csv("vocab_coef_svm.csv")

# get and store predictions for test data
pred_test = clf.predict(X_test)
pred_test_names = []
for i in pred_test:
    pred_test_names.append(target_names[i])

y_test_names = []
for i in y_test:
    y_test_names.append(target_names[i])

new_dict = {"true class no":y_test, "true class name":y_test_names, "pred class no": pred_test, "pred class name":pred_test_names}
coefs_test = pd.DataFrame(new_dict)


#print(X_test.shape)

# transform coefficient matrix format
X_test = X_test.tocoo()
# get number of test instances
num_docs = X_test.shape[0]

feature_indices_per_doc =  [[] for _ in range(num_docs)]
tfidf_per_doc = [[] for _ in range(num_docs)]

# get feature indices and tfidf scores per document
for doc_index, feature_index, tfidf in zip(X_test.row, X_test.col, X_test.data):
    feature_indices_per_doc[doc_index].append(feature_index)
    tfidf_per_doc[doc_index].append(tfidf)


coefs_test["feature ind"] = feature_indices_per_doc
coefs_test["tfidf"] = tfidf_per_doc

feature_names_per_doc = []
coef_true_per_doc = []
coef_pred_per_doc = []

#iterate over all docs
for doc in range(num_docs):
    names = []
    coefs_true = []
    coefs_pred = []
    #get name of true and pred class for doc
    true = y_test_names[doc]
    pred = pred_test_names[doc]
    #iterate over all features in doc
    for feature in range(len(feature_indices_per_doc[doc])):
        #get index of feature in vocab
        index = feature_indices_per_doc[doc][feature]
        #get feature name and add to list
        name = feature_names[index]
        names.append(name)
        #get feature coef for true and pred class, add to list
        coef_true= all_coef.at[index, true]
        coefs_true.append(coef_true)
        coef_pred = all_coef.at[index, pred]
        coefs_pred.append(coef_pred)
    feature_names_per_doc.append(names)
    coef_true_per_doc.append(coefs_true)
    coef_pred_per_doc.append(coefs_pred)

coefs_test["feature names"] = feature_names_per_doc
coefs_test["coef true"] = coef_true_per_doc
coefs_test["coef pred"] = coef_pred_per_doc

def multiply_lists(list1, list2):
    return [a * b for a, b in zip(list1, list2)]

# multiply coefs and tfidf scores
coefs_test['coef true*tfidf'] = coefs_test.apply(lambda row: multiply_lists(row['coef true'], row['tfidf']), axis=1)
coefs_test['coef pred*tfidf'] = coefs_test.apply(lambda row: multiply_lists(row['coef pred'], row['tfidf']), axis=1)
print(coefs_test)

# save coefficients for test data documents
coefs_test.to_csv("coefs_test.csv")




