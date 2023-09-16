from datasets import load_dataset
import evaluate
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)


# load dataset and create id-label-mapping
data = load_dataset("SetFit/20_newsgroups")
id2label = dict()
label2id = dict()

for d in data["test"]:
    id2label[d['label']] = d['label_text']
    label2id[d['label_text']] = d['label']

# preprocessing
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_data = data.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# define metrics
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred, metric=accuracy):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# define model, start training
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=20,
    id2label=id2label,
    label2id=label2id)

training_args = TrainingArguments(
    output_dir="distilbert-4",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

results1 = trainer.evaluate()
print(results1)

trainer.train()

results2 = trainer.evaluate()
print(results2)
trainer.save_model(training_args.output_dir)
