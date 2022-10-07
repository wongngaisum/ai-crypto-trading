import json

import keras
import numpy as np
import emoji
import numpy as np
import pandas as pd
import preprocessor as p
import tensorflow as tf
from keras import Input, Model
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.preprocessing import text
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from transformers import TFAutoModel, AutoTokenizer
from datetime import datetime
import re


test = False
datetime = []
empty_bucket = []
sadness_bucket = []
enthusiasm_bucket = []
neutral_bucket = []
worry_bucket = []
surprise_bucket = []
love_bucket = []
fun_bucket = []
hate_bucket = []
happiness_bucket = []
boredom_bucket = []
relief_bucket = []
anger_bucket = []

skipped_datetime = []
skipped_content_buckets = []

with open("./combined_grouped.json", "r") as f:
    count = 0
    for line in f:
        json_data = json.loads(line)
        count += len(json_data['content'])
        print(count)