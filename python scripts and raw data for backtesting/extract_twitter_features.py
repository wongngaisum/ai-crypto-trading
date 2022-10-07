import json
import re

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig

device = "cuda:0" if torch.cuda.is_available() else "cpu"


misspell_data = pd.read_csv("./aspell.txt", sep=":", names=["correction", "misspell"])
misspell_data.misspell = misspell_data.misspell.str.strip()
misspell_data.misspell = misspell_data.misspell.str.split(" ")
misspell_data = misspell_data.explode("misspell").reset_index(drop=True)
misspell_data.drop_duplicates("misspell", inplace=True)
miss_corr = dict(zip(misspell_data.misspell, misspell_data.correction))

# Sample of the dict
{v: miss_corr[v] for v in [list(miss_corr.keys())[k] for k in range(20)]}


def misspelled_correction(val):
    for x in val.split():
        if x in miss_corr.keys():
            val = val.replace(x, miss_corr[x])
    return val


contractions = pd.read_csv("./contractions.csv")
cont_dic = dict(zip(contractions.Contraction, contractions.Meaning))


def cont_to_meaning(val):
    for x in val.split():
        if x in cont_dic.keys():
            val = val.replace(x, cont_dic[x])
    return val


def punctuation(val):
    punctuations = '''()-[]{};:'"\,<>./@#$%^&_~'''

    for x in val.lower():
        if x in punctuations:
            val = val.replace(x, " ")
    return val


def clean_text(val):
    val = val.lower()
    val = re.sub("@[A-Za-z0-9_]+","", val)
    val = re.sub("#[A-Za-z0-9_]+","", val)
    val = re.sub(r"http\S+", "", val)
    val = re.sub(r"www.\S+", "", val)
    val = re.sub('[()!?]', ' ', val)
    val = re.sub('\[.*?\]',' ', val)
    # val = emoji.demojize(val)
    val = re.sub("[^a-z0-9]"," ", val)

    val = misspelled_correction(val)
    val = cont_to_meaning(val)
    # val = p.clean(val)
    # val = punctuation(val)
    val = ' '.join(val.split())
    return val

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)


def predict(text):
    text = clean_text(text)
    encoded_input = tokenizer(text, return_tensors='pt').to(device)
    output = model(**encoded_input)
    scores = output[0][0].detach().cpu().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    result = [None, None, None]
    for i in range(scores.shape[0]):
        l = config.id2label[ranking[i]]
        s = scores[ranking[i]]
        if l == 'Negative':
            result[0] = np.round(float(s), 4)
        elif l == 'Positive':
            result[2] = np.round(float(s), 4)
        elif l == 'Neutral':
            result[1] = np.round(float(s), 4)
    return result


datetime = []
content_buckets = []
predict_negative_buckets = []
predict_neutral_buckets = []
predict_positive_buckets = []

skipped_datetime = []
skipped_content_buckets = []

with open("./combined_grouped.json", "r") as f:
    count = 0
    for line in f:
       # if count < 4124:
       #     count += 1
       #     continue

        json_obj = json.loads(line)
        neg = 0.00000
        pos = 0.00000
        neu = 0.00000
        this_skiped = []
        handled = 0
        for i in range(len(json_obj['content'])):
            try:
                res = predict(json_obj['content'][i])
                neg += res[0]
                pos += res[2]
                neu += res[1]
                handled += 1
                count += 1
            except:
                this_skiped.append(json_obj['content'][i])
        if handled > 0:
            size = float(handled)
            neg = np.round(float(neg / size), 4)
            pos = np.round(float(pos / size), 4)
            neu = np.round(float(neu / size), 4)
            datetime.append(json_obj['date'])
            content_buckets.append(json_obj['content'])
            predict_negative_buckets.append(neg)
            predict_positive_buckets.append(pos)
            predict_neutral_buckets.append(neu)
        if len(this_skiped) > 0:
            skipped_datetime.append(json_obj['date'])
            skipped_content_buckets.append(this_skiped)

        #if count == 4125:
        #    break
        print(count)
df = pd.DataFrame(
    {'date': datetime, 'positive_score': predict_positive_buckets, 'neutral_score': predict_neutral_buckets,
     'negative_score': predict_negative_buckets})
with open("./predict.json", "a") as myfile:
    for index, row in df.iterrows():
        myfile.write(json.dumps(row.to_dict()))
        myfile.write('\n')
df = pd.DataFrame(
    {'date': skipped_datetime, 'contents': skipped_content_buckets})
with open("./skipped.json", "a") as myfile:
    for index, row in df.iterrows():
        myfile.write(json.dumps(row.to_dict()))
        myfile.write('\n')
