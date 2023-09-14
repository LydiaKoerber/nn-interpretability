import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from _stop_words import ENGLISH_STOP_WORDS


def extract_dataset_statistics(split, path, exclude_stopwords=False):
    df = pd.DataFrame(split)
    unigram_counts = {}
    bigram_counts = {}
    for d in df:
        pass
    # df_bylabels = df.group_by()
    df.to_csv(f'{path}.csv', encoding='utf-8')

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