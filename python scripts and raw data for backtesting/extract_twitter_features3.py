import json
import os
import re

import emoji
import numpy as np
import pandas as pd
import preprocessor as p
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pd.options.display.max_colwidth = 150  # show whole tweet's content
pd.options.display.width = 200  # don't break columns

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


def build_model():
    model_name = "ElKulako/cryptobert"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 3).to(device)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=64, device=0, truncation=True, padding = 'max_length')
    return pipe

model = build_model()

datetime = []
bearish_bucket = []
bullish_bucket = []
neutral_bucket = []

skipped_datetime = []
skipped_content_buckets = []

with open("./combined_grouped.json", "r") as f:
    count = 0
    for line in f:
        json_data = json.loads(line)
        bearish = 0.00000
        bullish = 0.00000
        neutral = 0.00000
        
        this_skiped = []
        handled = 0
        cleaned_text = [clean_text(i) for i in json_data['content']]
        sentiments = model(cleaned_text)
        for i in sentiments:
            if i['label'] == 'Bearish':
                bearish += 1
            elif i['label'] == 'Bullish':
                bullish += 1
            else:
                neutral += 1
            handled += 1
            count += 1
            print(handled)
        if handled > 0:
            size = float(handled)
            bearish = np.round(float(bearish / size), 4)
            bullish = np.round(float(bullish / size), 4)
            neutral = np.round(float(neutral / size), 4)

            datetime.append(json_data['date'])
            bearish_bucket.append(bearish)
            bullish_bucket.append(bullish)
            neutral_bucket.append(neutral)

        if len(this_skiped) > 0:
            skipped_datetime.append(json_data['date'])
            skipped_content_buckets.append(this_skiped)
        # if count == 1:
        #    break
        print(count)
    df = pd.DataFrame({"datetime": datetime, 'bearish': bearish_bucket, 'bullish': bullish_bucket, 'neutral': neutral_bucket})
    with open("./predict_rnn_sentiment.json", "a") as myfile:
        for index, row in df.iterrows():
            myfile.write(json.dumps(row.to_dict()))
            myfile.write('\n')
    df = pd.DataFrame(
        {'date': skipped_datetime, 'contents': skipped_content_buckets})
    with open("./skipped_rnn_sentiment.json", "a") as myfile:
        for index, row in df.iterrows():
            myfile.write(json.dumps(row.to_dict()))
            myfile.write('\n')
