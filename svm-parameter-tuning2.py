from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


data_train = fetch_20newsgroups(
    subset="train",
    shuffle=True,
    random_state=42,
    remove=("headers", "footers", "quotes"),
)

data_test = fetch_20newsgroups(
    subset="test",
    shuffle=True,
    random_state=42,
    remove=("headers", "footers", "quotes"),
)

print(f"Loading 20 newsgroups dataset:")
print(data_train.target_names)
print(f"{len(data_train.data)} documents")


pipeline = Pipeline(
    [
        ("tf_id", TfidfVectorizer()),
        ("svm", LinearSVC(max_iter=5000)),
    ]
)
pipeline

parameter = {'tf_id__max_df' : (0.15, 0.25, 0.35, 0.4),
             "tf_id__min_df": (3, 4, 5, ),
             'tf_id__smooth_idf' : (True, False),
             'tf_id__sublinear_tf' : (True, False),
             "tf_id__ngram_range": ((1, 2), (1, 3)),  # unigrams or bigrams
             "tf_id__norm": ("l1", "l2"),
             'tf_id__stop_words': [None, 'english'],
             'svm__C': [0.001, 0.01, 0.1, 1, 10, 20, 30, 40],
             "svm__dual": (True, False)
}


grid_search = GridSearchCV(pipeline, parameter,cv = 3, verbose=1, n_jobs=5)
grid_search.fit(data_train.data, data_train.target)

print(grid_search.best_params_)
print(grid_search.best_score_)