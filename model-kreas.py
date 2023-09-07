# wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
# unzip -o uncased_L-12_H-768_A-12.zip

import codecs
import tensorflow as tf
from tqdm import tqdm
from chardet import detect
import numpy as np
import keras
from keras_radam import RAdam
from keras import backend as K
from keras_bert import load_trained_model_from_checkpoint, Tokenizer

# https://www.kaggle.com/code/sharmilaupadhyaya/20newsgroup-classification-using-keras-bert-in-gpu

# PARAMETERS
SEQ_LEN = 128
BATCH_SIZE = 50
EPOCHS = 7
LR = 1e-4

# MODEL
pretrained_path = 'uncased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

model = load_trained_model_from_checkpoint(
      config_path,
      checkpoint_path,
      training=True,
      trainable=True,
      seq_len=SEQ_LEN,
  )

print(model.summary())

token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

print(token_dict.items()[:20])

# DATASET
dataset = tf.keras.utils.get_file(
    fname="20news-18828.tar.gz", 
    origin="http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz", 
    extract=True,
)

tokenizer = Tokenizer(token_dict)
# label-index pairs
datapath = ".".join(dataset.split(".")[:-2])
txtfiles = os.listdir(datapath)
labels = [(x, i) for i,x in enumerate(txtfiles)]
def get_label(index):
    for each in labels:
        if index == each[1]:
            return each[0]

# create data splits
def load_data(path, labels):
    global tokenizer
    indices, sentiments = [], []
    for folder, sentiment in labels:
        folder = os.path.join(path, folder)
        for name in tqdm(os.listdir(folder)):
            with open(os.path.join(folder, name), 'r', encoding="utf-8", errors='ignore') as reader:
                  text = reader.read()
            ids, segments = tokenizer.encode(text, max_len=SEQ_LEN)
            indices.append(ids)
            sentiments.append(sentiment)
    items = list(zip(indices, sentiments))
    
    np.random.shuffle(items)
    test_items = items[int(0.8*len(items)):]
    train_items = items[:int(0.8*len(items))]
    indices_test, sentiments_test = zip(*test_items)
    indices_train, sentiments_train = zip(*train_items)
    indices_train = np.array(indices_train)
    indices_test = np.array(indices_test)
    mod_train = indices_train.shape[0] % BATCH_SIZE
    mod_test = indices_test.shape[0] % BATCH_SIZE
    if mod_train > 0:
        indices_train, sentiments_train = indices_train[:-mod_train], sentiments_train[:-mod_train]
    if mod_test > 0:
      indices_test, sentiments_test = indices_test[:-mod_test], sentiments_test[:-mod_test]

    return [indices_train, np.zeros_like(indices_train)], np.array(sentiments_train),[indices_test, np.zeros_like(indices_test)], np.array(sentiments_test)
  
train_path = os.path.join(os.path.dirname(dataset), '20news-bydate')
train_x, train_y, test_x, test_y = load_data(train_path, labels)
print(pd.Series(train_y).value_counts(), pd.Series(test_y).value_counts())

# add news classification layer
inputs = model.inputs[:2]
dense = model.get_layer('NSP-Dense').output
outputs = keras.layers.Dense(units=20, activation='softmax')(dense)

model = keras.models.Model(inputs, outputs)
model.compile(
  RAdam(learning_rate =LR),
  loss='sparse_categorical_crossentropy',
  metrics=['sparse_categorical_accuracy'],
)

# initialize variables
sess = K.get_session()
uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])
init_op = tf.variables_initializer(
    [v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables]
)
sess.run(init_op)

# TRAIN
model.fit(
    train_x,
    train_y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)

# predict & compute accuracy
predicts = model.predict(test_x, verbose=True).argmax(axis=-1)
print(np.sum(test_y == predicts) / test_y.shape[0])
