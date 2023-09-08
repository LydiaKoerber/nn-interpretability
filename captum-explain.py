from transformers import pipeline, DistilBertTokenizer, BertTokenizer
from datasets import load_dataset
import explainer


def setup_bert():
    model_repo = 'bert-20news-0'
    tok = BertTokenizer.from_pretrained(model_repo)
    clf = pipeline(task= 'text-classification',
                        model= f"{model_repo}",
                        tokenizer = tok)
    return clf

def setup_distilbert():
    model_repo_dist = 'distilbert-20news-0'
    tok_dist = DistilBertTokenizer.from_pretrained(model_repo_dist)
    clf_dist = pipeline(task= 'text-classification',
                        model= f"{model_repo_dist}",
                        tokenizer = tok_dist)

def explain_all(test_data, exp_model):
    for i, d in enumerate(data):
        exp_model.explain(d['text'])

if __name__ == '__main__':
    # data["test"][0]
    example = {
        'text': 'I am a little confused on all of the models of the 88-89 bonnevilles.\nI have heard of the LE SE LSE SSE SSEI. Could someone tell me the\ndifferences are far as features or performance. I am also curious to\nknow what the book value is for prefereably the 89 model. And how much\nless than book value can you usually get them for. In other words how\nmuch are they in demand this time of year. I have heard that the mid-spring\nearly summer is the best time to buy.',
        'label': 7,
        'label_text': 'rec.autos'
        }
    device = 'cpu'
    clf = setup_bert()
    exp_model_bert = explainer.ExplainableTransformerPipeline(clf, device, 'output', algorithms=['lig', 'ig'], model='bert')
    exp_model_bert.explain(example['text'])
    # data = load_dataset("SetFit/20_newsgroups")
    # explain_all(data['test'], exp_model)

