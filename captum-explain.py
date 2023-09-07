from transformers import pipeline, DistilBertTokenizer, BertTokenizer
import explainer


model_repo = 'bert-20news-0'
if model_repo.startswith('distil'):
    tok = DistilBertTokenizer.from_pretrained(model_repo)
else:
    tok = BertTokenizer.from_pretrained(model_repo)

clf = pipeline(task= 'text-classification',
                      model= f"{model_repo}",
                      tokenizer = tok)

def predict(clf, sample):
    return clf.predict(sample)


if __name__ == '__main__':
    # data["test"][0]
    example = {
        'text': 'I am a little confused on all of the models of the 88-89 bonnevilles.\nI have heard of the LE SE LSE SSE SSEI. Could someone tell me the\ndifferences are far as features or performance. I am also curious to\nknow what the book value is for prefereably the 89 model. And how much\nless than book value can you usually get them for. In other words how\nmuch are they in demand this time of year. I have heard that the mid-spring\nearly summer is the best time to buy.',
        'label': 7,
        'label_text': 'rec.autos'
        }
    pred = predict(clf, example['text'])
    print(pred)
    device = 'cpu'
    exp_model = explainer.ExplainableTransformerPipeline(clf, device)
    exp_model.explain(example['text'])

