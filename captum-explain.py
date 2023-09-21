from transformers import (
    AutoTokenizer,
    BertTokenizer,
    DistilBertTokenizer,
    pipeline
    )
from datasets import load_dataset
import pandas as pd
import explainer


def setup(model_repo):
    try:
        tok = AutoTokenizer.from_pretrained(model_repo)
    except Exception as e:
        if 'distilbert' in model_repo:
            tok = DistilBertTokenizer('models/distilbert-4/vocab.txt')
        else:
            tok = BertTokenizer('models/bert-4/vocab.txt')
    clf = pipeline(task='text-classification',
                        model=model_repo,
                        tokenizer=tok)
    return clf


def explain_all(test_data, exp_model, subsplit_size=500):
    df = pd.DataFrame(columns=['label_pred', 'score', 'tokens',
                               'attributions'])
    for i, d in test_data.iterrows():
        if i > 1000:
            break
        try:
            a, pred = exp_model.explain(d['truncated'])
            new_row = pd.DataFrame({
                'label_pred': pred[0]['label'],
                'score': pred[0]['score'],
                'tokens': [a.index.tolist()],
                'attributions': [a.tolist()]
            })
            df = df.append(new_row, ignore_index=True)
            none_row = False
        except Exception as e:
            print(i, e)
            # avoid several rows for one index if several error messages
            if not none_row:
                df = df.append([None, None, None, None], ignore_index=True)
                none_row = True
        if (i+1) % subsplit_size == 0:  # export split to dataframe
            df.to_csv(f'outputs/{exp_model.model}/{
                exp_model.model}_attributions_{int(i/subsplit_size)}.csv')
            df = pd.DataFrame(columns=['label_pred', 'score', 'tokens',
                                       'attributions'])
    df.to_csv(f'outputs/{exp_model.model}/{exp_model.model}_attributions_{
        int(i/subsplit_size)}.csv')


def demo():
    example1 = {
        'text': 'I am a little confused on all of the models of the 88-89 bonnevilles.\nI have heard of the LE SE LSE SSE SSEI. Could someone tell me the\ndifferences are far as features or performance. I am also curious to\nknow what the book value is for prefereably the 89 model. And how much\nless than book value can you usually get them for. In other words how\nmuch are they in demand this time of year. I have heard that the mid-spring\nearly summer is the best time to buy.',
        'label': 7,
        'label_text': 'rec.autos'
        }
    example2 = {
        'text': " They were attacking the Iraqis to drive them out of Kuwait, a country whose citizens have close blood and business ties to Saudi citizens. And me thinks if the US had not helped out the Iraqis would have swallowed Saudi Arabia, too (or at least the eastern oilfields). And no Muslim country was doing much of anything to help liberate Kuwait and protect Saudi Arabia; indeed, in some masses of citizens were demonstrating in favor of that butcher Saddam (who killed lotsa Muslims), just because he was killing, raping, and looting relatively rich Muslims and also thumbing his nose at the West. So how would have *you* defended Saudi Arabia and rolled back the Iraqi invasion, were you in charge of Saudi Arabia??? I think that it is a very good idea to not have governments have an official religion (de facto or de jure), because with human nature like it is, the ambitious and not the pious will always be the ones who rise to power. There are just too many people in this world (or any country) for the citizens to really know if a leader is really devout or if he is just a slick operator. You make it sound like these guys are angels, Ilyess. (In your clarinet posting you edited out some stuff; was it the following???) Friday's New York Times reported that this group definitely is more conservative than even Sheikh Baz and his followers (who think that the House of Saud does not rule the country conservatively enough). The NYT reported that, besides complaining that the government was not conservative enough, they have: - asserted that the (approx. 500,000) Shiites in the Kingdom are apostates, a charge that under Saudi (and Islamic) law brings the death penalty. Diplomatic guy (Sheikh bin Jibrin), isn't he Ilyess? - called for severe punishment of the 40 or so women who drove in public a while back to protest the ban on women driving. The guy from the group who said this, Abdelhamoud al-Toweijri, said that these women should be fired from their jobs, jailed, and branded as prostitutes. Is this what you want to see happen, Ilyess? I've heard many Muslims say that the ban on women driving has no basis in the Qur'an, the ahadith, etc. Yet these folks not only like the ban, they want these women falsely called prostitutes? If I were you, I'd choose my heroes wisely, Ilyess, not just reflexively rally behind anyone who hates anyone you hate. - say that women should not be allowed to work. - say that TV and radio are too immoral in the Kingdom. Now, the House of Saud is neither my least nor my most favorite government on earth; I think they restrict religious and political reedom a lot, among other things. I just think that the most likely replacements for them are going to be a lot worse for the citizens of the country. But I think the House of Saud is feeling the heat lately. In the last six months or so I've read there have been stepped up harassing by the muttawain (religious police---*not* government) of Western women not fully veiled (something stupid for women to do, IMO, because it sends the wrong signals about your morality). And I've read that they've cracked down on the few, home-based expartiate religious gatherings, and even posted rewards in (government-owned) newspapers offering money for anyone who turns in a group of expartiates who dare worship in their homes or any other secret place. So the government has grown even more intolerant to try to take some of the wind out of the sails of the more-conservative opposition. As unislamic as some of these things are, they're just a small taste of what would happen if these guys overthrow the House of Saud, like they're trying to in the long run. Is this really what you (and Rached and others in the general west-is-evil-zionists-rule-hate-west-or-you-are-a-puppet crowd) want, Ilyess? ",
        'label': 17,
        'label_text': 'talk.politics.mideast'
        }
    device = 'cpu'
    clf = setup('models/distilbert-2/')
    exp_model_bert = explainer.ExplainableTransformerPipeline(clf,
                                                              device,
                                                              'output',
                                                              algorithms=['lig'],
                                                              model='distilbert')
    print(exp_model_bert.explain(example1['text']))
    print(exp_model_bert.explain(example2['text']))


if __name__ == '__main__':
    data_path = 'dataset/data_test.csv'
    data_df = pd.read_csv(data_path)
    device = 'cpu'
    dist = False
    if dist:
        # distilbert setup
        clf = setup('models/distilbert-4/')
        exp_model_distilbert = explainer.ExplainableTransformerPipeline(clf,
                                                                        device,
                                                                        'output/distilbert',
                                                                        algorithms=['lig'],
                                                                        model='distilbert')
        explain_all(data_df, exp_model_distilbert)
    else:
        # bert setup
        clf = setup('models/bert-4/')
        exp_model_bert = explainer.ExplainableTransformerPipeline(clf,
                                                                  device,
                                                                  'output/bert',
                                                                  algorithms=['lig'],
                                                                  model='bert')
        explain_all(data_df, exp_model_bert)

