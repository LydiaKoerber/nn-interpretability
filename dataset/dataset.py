import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from transformers import AutoTokenizer


# load dataset, remove headers, footers, quotes
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

# truncate dataset
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def truncate(text):
    ids = tokenizer(text, truncation=True)
    # decode, convert ids to tokens
    decoded = tokenizer.decode(ids['input_ids'], skip_special_tokens=True)
    return decoded


data_test.pop('target_names')
df_test = pd.DataFrame.from_dict(data_test)[["data", "target"]]
for i, data in df_test.iterrows():
    df_test.loc[i, 'truncated'] = truncate(data["data"])

data_train.pop('target_names')
df_train = pd.DataFrame.from_dict(data_train)[["data", "target"]]
for i, data in df_train.iterrows():
    df_train.loc[i, 'truncated'] = truncate(data["data"])

# save to csv files
df_train.drop('data', axis=1).to_csv('data_train.csv', encoding='utf-8')
df_test.drop('data', axis=1).to_csv('data_test.csv', encoding='utf-8')
