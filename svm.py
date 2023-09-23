from time import time

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

np.random.seed(42)


def load_dataset(verbose=False, remove=()):
    """Load and vectorize the 20 newsgroups dataset."""

    data_train = fetch_20newsgroups(
        subset="train",
        shuffle=True,
        random_state=42,
        remove=remove,
    )

    data_test = fetch_20newsgroups(
        subset="test",
        shuffle=True,
        random_state=42,
        remove=remove,
    )

    # order of labels in `target_names` can be different from `categories`
    target_names = data_train.target_names

    # split target in a training set and a test set
    y_train, y_test = data_train.target, data_test.target

    # Extracting features from the training data using a sparse vectorizer
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=0.15, min_df=3, stop_words="english", ngram_range=(1,1), norm="l1", smooth_idf=False
    )
    X_train = vectorizer.fit_transform(data_train.data)

    # Extracting features from the test data using the same vectorizer
    X_test = vectorizer.transform(data_test.data)

    feature_names = vectorizer.get_feature_names_out()

    return X_train, X_test, y_train, y_test, feature_names, target_names

# Load data with metadata stripping
'''(    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
    target_names,
) = load_dataset(remove=("headers", "footers", "quotes"))'''

data_train = pd.read_csv('data_train.csv')
data_test = pd.read_csv('data_test.csv')

# Extracting features from the training data using a sparse vectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.15, min_df=3, stop_words="english", ngram_range=(1,1), norm="l1", smooth_idf=False)

X_train = vectorizer.fit_transform(data_train["truncated"].apply(lambda x: np.str_(x)))
X_test = vectorizer.transform(data_test['truncated'].apply(lambda x: np.str_(x)))

y_train, y_test = data_train["target"], data_test['target']

feature_names = vectorizer.get_feature_names_out()
target_names = fetch_20newsgroups(subset="train",shuffle=True,random_state=42).target_names


from sklearn import metrics
from sklearn.utils.extmath import density


def benchmark(clf, custom_name=False):
    print("_" * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print(f"train time: {train_time:.3}s")

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print(f"test time:  {test_time:.3}s")

    score = metrics.accuracy_score(y_test, pred)
    print(f"accuracy:   {score:.3}")

    if hasattr(clf, "coef_"):
        print(f"dimensionality: {clf.coef_.shape[1]}")
        print(f"density: {density(clf.coef_)}")
        print()

    print()
    if custom_name:
        clf_descr = str(custom_name)
    else:
        clf_descr = clf.__class__.__name__
    return clf, clf_descr, score, train_time, test_time

from sklearn.svm import LinearSVC

print("=" * 80)
print("Linear SVC")
results = benchmark(LinearSVC(C=30, dual=True, max_iter=3000), "Linear SVC")

clf = results[0]
print(clf)

coef = clf.coef_
#mi, ma = -1, 1
#coef_std = (coef - coef.min()) / (coef.max() - coef.min())
#normalized_coef = coef_std * (ma - mi) + mi

min_val = coef.min()
max_val = coef.max()
normalized_coef = 2 * (coef - min_val) / (max_val - min_val) - 1

print(normalized_coef.min())
print(normalized_coef.max())

# Calculate the maximum absolute coefficient value
#max_abs_coef = np.max(np.abs(coef))
#print(max_abs_coef)
# Normalize the coefficients
#normalized_coef = coef / max_abs_coef

class_dict = {"feature":feature_names}

target_names.sort()

for i in range(20):
    class_dict[target_names[i]]=normalized_coef[i]

all_coef = pd.DataFrame(class_dict)
#print(all_coef)
all_coef.to_csv("vocab_coef_svm.csv")

pred_test = clf.predict(X_test)
pred_test_names = []
for i in pred_test:
    pred_test_names.append(target_names[i])

y_test_names = []
for i in y_test:
    y_test_names.append(target_names[i])

new_dict = {"true class no":y_test, "true class name":y_test_names, "pred class no": pred_test, "pred class name":pred_test_names}
coefs_test = pd.DataFrame(new_dict)
#print(test_coef)

print(X_test.shape)



#feature_index = X_test[0,:].nonzero()[1]
#tfidf_scores = zip(feature_index, [X_test[0, x] for x in feature_index])
#for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
  #print(i, w, s)


X_test = X_test.tocoo()
#nonzero_mask = X_test.data != 0
# Get the number of rows in the sparse matrix
num_docs = X_test.shape[0]

# Initialize lists to store feature indices and tfidf scores for each doc
feature_indices_per_doc = [[] for _ in range(num_docs)]
tfidf_per_doc = [[] for _ in range(num_docs)]

# Accessing elements in the sparse matrix and organizing them by doc
for doc_index, feature_index, tfidf in zip(X_test.row, X_test.col, X_test.data):
    feature_indices_per_doc[doc_index].append(feature_index)
    tfidf_per_doc[doc_index].append(tfidf)


coefs_test["feature ind"] = feature_indices_per_doc
coefs_test["tfidf"] = tfidf_per_doc

feature_names_per_doc = []
coef_true_per_doc = []
coef_pred_per_doc = []

#TODO Unterscheidung predicted und true class

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

coefs_test['coef true*tfidf'] = coefs_test.apply(lambda row: multiply_lists(row['coef true'], row['tfidf']), axis=1)
coefs_test['coef pred*tfidf'] = coefs_test.apply(lambda row: multiply_lists(row['coef pred'], row['tfidf']), axis=1)
print(coefs_test)
#print(X_test.col[nonzero_mask])
#print(X_test.data[nonzero_mask])

coefs_test.to_csv("coefs_test.csv")


import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

#fig, ax = plt.subplots(figsize=(10, 10))
#ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
#ax.xaxis.set_ticklabels(target_names)
#ax.yaxis.set_ticklabels(target_names)
#_ = ax.set_title(
#    f"Confusion Matrix for {clf.__class__.__name__}\non the original documents"
#)
#plt.show()

#feature_weights = weights.dot(X_test[0])
#print(feature_weights[0].shape)


# Initialize an empty DataFrame to store the results
#df = pd.DataFrame(columns=[f"Class_{i}" for i in range(len(weights))])

# Calculate feature weights for each sample in X_test and add them to the DataFrame
#for i, sample in enumerate(X_test):
    #feature_weights = weights.dot(sample)
    #feature_weights = feature_weights.reshape(1, -1)
    #df.loc[i] = feature_weights

# Add feature names as column names
#df.columns = feature_names

# Export the DataFrame to a CSV file
#csv_filename = "feature_weights.csv"
#df.to_csv(csv_filename, index=False)

#print(f"Feature weights exported to {csv_filename}")
