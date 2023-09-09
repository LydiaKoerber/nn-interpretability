from transformers import pipeline, DistilBertTokenizer, BertTokenizer
from datasets import load_dataset
import pandas as pd
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
    return clf_dist

def explain_all(test_data, exp_model, subsplit_size=500):
    for j in range(0, len(test_data), subsplit_size):
        print(f'split {j}')
        df = pd.DataFrame(columns=['label_pred', 'score', 'tokens', 'attributions'])
        subsplit = test_data[j:j + subsplit_size]
        for i, d in enumerate(subsplit):
            try:
                a, pred = exp_model.explain(d['text'])
                new_row = pd.DataFrame({
                    'label_pred': pred[0]['label'],
                    'score': pred[0]['score'],
                    'tokens': [a.index.tolist()],
                    'attributions': [a.tolist()]
                })
                df = df.append(new_row, ignore_index=True)
            except Exception as e:
                print(i, e)
                df = df.append([None, None, None, None], ignore_index=True)
        df.to_csv(f'outputs/{exp_model.model}_attributions_{j}.csv')

if __name__ == '__main__':
    # data["test"][0]
    example = {
        'text': 'I am a little confused on all of the models of the 88-89 bonnevilles.\nI have heard of the LE SE LSE SSE SSEI. Could someone tell me the\ndifferences are far as features or performance. I am also curious to\nknow what the book value is for prefereably the 89 model. And how much\nless than book value can you usually get them for. In other words how\nmuch are they in demand this time of year. I have heard that the mid-spring\nearly summer is the best time to buy.',
        'label': 7,
        'label_text': 'rec.autos'
        }
    device = 'cpu'
    clf = setup_bert()
    exp_model_bert = explainer.ExplainableTransformerPipeline(clf, device, 'output', algorithms=['lig', 'lrp'], model='bert')
    #exp_model_bert.explain(example['text'])
    data = load_dataset("SetFit/20_newsgroups")
    explain_all(data['test'], exp_model_bert)

