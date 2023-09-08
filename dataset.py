from datasets import load_dataset
import numpy as np
import pandas as pd

from _stop_words import ENGLISH_STOP_WORDS


def extract_dataset_statistics(split, path, exclude_stopwords=False):
    df = split.to_pandas()
    unigram_counts = {}
    bigram_counts = {}
    for d in df:
        pass
    # df_bylabels = df.group_by()
    df.to_csv(f'{path}.csv')

if __name__ == '__main__':
    data = load_dataset("SetFit/20_newsgroups")
    extract_dataset_statistics(data['test'], 'outputs/20newsgroups_test')
