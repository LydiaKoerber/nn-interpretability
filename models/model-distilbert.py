from datasets import Dataset, load_dataset
import evaluate
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import time
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    create_optimizer,
    DataCollatorWithPadding,
    TFAutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

run = 2

# load dataset from sklearn
# dicts of lists of strings, data and target_names
train = fetch_20newsgroups(
    subset="train",
    shuffle=True,
    random_state=42,
    remove=("headers", "footers", "quotes"),
)

test = fetch_20newsgroups(
    subset="test",
    shuffle=True,
    random_state=42,
    remove=("headers", "footers", "quotes"),
)

train.pop("target_names")
train_df = pd.DataFrame.from_dict(train)[["data","target"]]
test.pop("target_names")
test_df = pd.DataFrame.from_dict(test)[["data","target"]]

train_ds = Dataset.from_pandas(train_df)
test_ds  = Dataset.from_pandas(test_df)
train_ds = train_ds.class_encode_column("target")
test_ds = test_ds.class_encode_column("target")

# create validation split
train_dsd = train_ds.train_test_split(test_size=0.1,seed=19,stratify_by_column="target")
train_dsd['validation'] = train_dsd['test']
train_dsd['test'] = test_ds

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

MAX_LEN = 256
def preprocess_function(examples):
    #return tokenizer(examples['data'], truncation=True, padding=True, max_length=MAX_LEN)
    return tokenizer(examples['data'], truncation=True, max_length=MAX_LEN)

tokenized_text = train_dsd.map(preprocess_function,batched=True)
data_collator = DataCollatorWithPadding(tokenizer)

# convert to tf dataset
BATCH_SIZE = 16

tf_train_set = tokenized_text["train"].to_tf_dataset(
    columns=['input_ids', 'attention_mask'],
    label_cols=["target"],
    shuffle= True,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
)
tf_validation_set = tokenized_text["validation"].to_tf_dataset(
    columns=['input_ids', 'attention_mask'],
    label_cols=["target"],
    shuffle= True,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
    )
tf_test_set = tokenized_text["test"].to_tf_dataset(
    columns=['input_ids', 'attention_mask'],
    label_cols=["target"],
    shuffle= True,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
    )

# training parameters
EPOCHS = 5
batches_per_epoch = len(tokenized_text["train"]) // BATCH_SIZE
total_train_steps = int(batches_per_epoch * EPOCHS)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

# load model
my_bert = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint,num_labels=20)
# compile
my_bert.compile(optimizer=optimizer,loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


# fine tune
es = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=1,
)

# reduce learning rate when metric stops evolving
rlp = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    patience=3,
    verbose=1,
)

start_time = time.time()
history = my_bert.fit(tf_train_set,
    validation_data=tf_validation_set,
    epochs=5
)
end_time = time.time()
print(f"Elapsed time: {end_time-start_time:.4f} seconds")

bert_loss, bert_acc = my_bert.evaluate(tf_test_set)
print(bert_loss, bert_acc)
my_bert.save_pretrained(f"distilbert-{run}")

