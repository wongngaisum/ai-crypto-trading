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
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
# sess = tf.compat.v1.Session(config=config)

checkpoint_path = "training_bert/cp-{epoch:04d}.ckpt"
logdir = "logs/bert/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1,
                                                   write_graph=True,
                                                   write_images=True,
                                                   update_freq='epoch',
                                                   profile_batch=2,
                                                   embeddings_freq=1)

data = pd.read_csv("./text_emotion.csv")

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


data["clean_content"] = data.content.apply(lambda x: misspelled_correction(x))
contractions = pd.read_csv("./contractions.csv")
cont_dic = dict(zip(contractions.Contraction, contractions.Meaning))


def cont_to_meaning(val):
    for x in val.split():
        if x in cont_dic.keys():
            val = val.replace(x, cont_dic[x])
    return val


data.clean_content = data.clean_content.apply(lambda x: cont_to_meaning(x))
p.set_options(p.OPT.MENTION, p.OPT.URL)
data["clean_content"] = data.content.apply(lambda x: p.clean(x))


def punctuation(val):
    punctuations = '''()-[]{};:'"\,<>./@#$%^&_~'''

    for x in val.lower():
        if x in punctuations:
            val = val.replace(x, " ")
    return val


data.clean_content = data.clean_content.apply(lambda x: ' '.join(punctuation(emoji.demojize(x)).split()))

def clean_text(val):
    val = val.lower()

    # stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
    #          'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
    #          'by','can','d','did','do',
    #          'does','doing','down','during','each','few','for','from','o','of','on','once',
    #          'only','or','other','our','ours','ourselves','out','own','re','s','same','she','shes'
    #          't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
    #          'to', 'too', 'under', 'until', 'up', 've', 'very', 'was']

    # val = ' '.join([word for word in val if not word in stopwordlist])
    val = emoji.demojize(val)
    val = " ".join(filter(lambda x:x[0]!='#' and x[0]!='0x' and x[0]!='$', val.split()))
    val = re.sub(r'(.)1+', r'1', val)
    val = re.sub('[0-9]+', '', val)
    val = re.sub('((www.[^s]+)|(https?://[^s]+))',' ',val)
    val = re.sub(r"www.\S+", "", val)
    val = re.sub('[()!?]', ' ', val)
    val = re.sub('\[.*?\]',' ', val)
    # val = re.sub("[^a-z0-9]"," ", val)

    val = misspelled_correction(val)
    val = cont_to_meaning(val)
    # val = p.clean(val)
    # val = punctuation(val)
    # val = ' '.join(val.split())
    # print(val.encode('utf-8', 'replace').decode('utf-8'))
    return val


data = data[data.clean_content != ""]

data.sentiment.value_counts()

sent_to_id = {"empty": 0, "sadness": 1, "enthusiasm": 2, "neutral": 3, "worry": 4,
              "surprise": 5, "love": 6, "fun": 7, "hate": 8, "happiness": 9, "boredom": 10, "relief": 11, "anger": 12}

data["sentiment_id"] = data['sentiment'].map(sent_to_id)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data.sentiment_id)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Y = onehot_encoder.fit_transform(integer_encoded)

X_train, X_test, y_train, y_test = train_test_split(data.clean_content, Y, random_state=1995, test_size=0.2,
                                                    shuffle=True)

# using keras tokenizer here
token = text.Tokenizer(num_words=None)
max_len = 160
Epoch = 3
token.fit_on_texts(list(X_train) + list(X_test))
X_train_pad = pad_sequences(token.texts_to_sequences(X_train), maxlen=max_len)
X_test_pad = pad_sequences(token.texts_to_sequences(X_test), maxlen=max_len)

w_idx = token.word_index

embed_dim = 160
lstm_out = 250


def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts.tolist(),
        return_attention_mask=False,
        return_token_type_ids=False,
        padding='max_length',
        max_length=maxlen,
        truncation=True
    )

    return np.array(enc_di['input_ids'])


def build_model(transformer, max_len=160):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(13, activation='softmax')(cls_token)

    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

AUTO = tf.data.experimental.AUTOTUNE
MODEL = 'roberta-base'

tokenizer = AutoTokenizer.from_pretrained(MODEL)

X_train_t = regular_encode(X_train, tokenizer, maxlen=max_len)
X_test_t = regular_encode(X_test, tokenizer, maxlen=max_len)
batch_size = 16

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_train_t, y_train))
    .repeat()
    .shuffle(1995)
    .batch(batch_size)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_test_t, y_test))
    .batch(batch_size)
    .cache()
    .prefetch(AUTO)
)

transformer_layer = TFAutoModel.from_pretrained(MODEL)
model_roberta_base = build_model(transformer_layer, max_len=max_len)
model_roberta_base.summary()

n_steps = X_train.shape[0] // batch_size
# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq='epoch')

es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)

last_checkpoint = tf.train.latest_checkpoint("./training_bert")

if last_checkpoint is None:
    # Save the weights using the `checkpoint_path` format
    model_roberta_base.save_weights(checkpoint_path.format(epoch=0))
    model_roberta_base.fit(train_dataset, steps_per_epoch=n_steps, validation_data=valid_dataset, epochs=Epoch, callbacks=[cp_callback, tensorboard_callback, es_callback])

model_roberta_base.load_weights(last_checkpoint)


def get_sentiment2(model, text):
    text = clean_text(text)
    # tokenize
    x_test1 = regular_encode(np.array([text]), tokenizer, maxlen=max_len)
    test1 = (tf.data.Dataset.from_tensor_slices(x_test1).batch(1))
    # test1
    sentiment = model.predict(test1, verbose=0)
    sent = np.round(np.dot(sentiment, 100).tolist(), 0)[0]
    result = pd.DataFrame([sent_to_id.keys(), sent]).T
    result.columns = ["sentiment", "percentage"]
    result = result[result.percentage != 0]
    return result


test = True
if test:
    test_sentences =  [
        "We know, we know... Soon\ud83d\ude09 #enjin #blockchaingame #ethereum https://t.co/jthHwgdUN8",
        "DogData Airdrop is now Live\ud83d\ude80\ud83d\udcb0\ud83c\udfc6\n\nClick on below link to participate into this amazing #Airdrop\ud83c\udf81\n\nAirdrop link\ud83d\udc49\nhttps://t.co/UY0U9OSTZ7\n\n\ud83c\udf81 Rewards: 250 + 100 ETHBN \ud83c\udf81\n\ud83d\udcb0 Value : $ 2.5\n\ud83d\udc65 Referral: $ 1\n\n#Airdrops #blockchain #cryptocurrency #ICO #bitcoin #Crypto #ETH",
        "@SpartanBlack_1 Their main value is being idea sources/ research chains for #Ethereum to learn from. And for $RUNE to capture their liquidity so it can be traded in for valuable DeFi coins without using a CEX",
        "$ETHUSD: Here are some HTF levels. Overall, the 1M doesn't need any explanation- incredibly bullish. Once we see $365 broken, I think we see $500 in relative short order. IMO #Ethereum will pull the entire market up if this happens. Looking great. $ETH https://t.co/E3OfdtEn6x",
        # "#Ethereum #ETH current price (GBP): \u00a3263.57",
        # "#Ethereum #ETH current price (GBP): \u00a3263.57",
        # "Join the Cardano ADA Revolution, get the latest Cardano updates.. Follow --- \u26a1Cardano \u20b3rmy\u26a1 @CardanoRmy @IOHK_Charles @CardanoStiftung #cardano #Litecoin #Blockchain #to #Mars2020 #cryptocurrency #CryptocurrencyNews #Bitcoin #Ethereum #EthereumClassic #XRPcommunity #TRON https://t.co/ICzM58Ln8y",
        # "U\nS\nD\nC\n\nhttps://t.co/Ceuf4qxTif\n\n#free $USDC #USDC \n\nSo many faucets; this is the only one that dispenses near-U.S. currency!\n\n#crypto #bitcoin #btc #ltc #eth #trx #xrp #htmlcoin #html #altcoin #althash #xem",
        # "#bidcoin social #reward program gives you #FREE bid token to #play right away! NO need to recharge!! \ud83d\ude0d click now https://t.co/FC1A5Uxpq8 for your #gift and #win #crypto like #bitcoin #ethereum #dash #btc #eth #xmr #ltc #zec without any #exchange, only on https://t.co/dAfxkvJPhC https://t.co/c0CUaLrmNB",
        # "Happy Birthday #Ethereum: 5 Years In and the Future of #ETH https://t.co/IaMlXd2ZMY https://t.co/EAJjn9o0lF",
        # "Don't underestimate the power of #FOMO\n\n$BTC #ETH",
        # "Hourly Price Movement:\n\n#edg $edg #Edgeware Loses -9.69% \ud83d\ude1e\n#yfi $yfi #yearn.finance Gains 2.7% \ud83d\udd25\n#qnt $qnt #Quant Gains 2.62% \ud83e\udd11\n\nPlease checkout https://t.co/tmXV05jb4H for more!\n\n#crypto #bitcoin #cryptocurrency #btc #eth #xrp #bch #usdt #ltc #eos #bsv #bnb https://t.co/z3kQ4ttjVk",
        # "Some of the wealthiest families were created out of the Great Depression. . . The same thing will happen in the Great Reset ! \n\u2022\n\u2022\n\u2022\n\n#Wealth #Forex #Chicago #Crypto #Bitcoin #Xrp #Ethereum #Cardano #Success\u2026 https://t.co/E427MrhcDt",
        # "\ud83d\udcb0 - 15,000,000 #USDT (15,100,877 USD) move from unknow wallet to unknow wallet\n \nDate : 2020-08-01 01:07:28 (GMT 0)\nBlockchain : #ETHEREUM\nSymbol : #USDT\n \nDetail: https://t.co/SlnUUI0pQ0",
        # "\ud83d\udcb0 - 3,999,997 #USDT (3,996,085 USD) move from #binance to #binance\n \nDate : 2020-08-01 01:31:43 (GMT 0)\nBlockchain : #ETHEREUM\nSymbol : #USDT\n \nDetail: https://t.co/zatVXwpipi",
        # "The greatest of all Classics: Classic Rock: I've Seen All Good People A. Your Move B. All Good People by Yes is playing right now! Tune in at https://t.co/okWxACLJsq and  get PAID in #Ethereum Tokens then go trade for some old school Bitcoin. Hows that for a Classic Crypto?!",
        # "The greatest of all Classics: Classic Rock: Comfortably Numb by Pink Floyd is playing right now! Tune in at https://t.co/okWxACLJsq and  get PAID in #Ethereum Tokens then go trade for some old school Bitcoin. Hows that for a Classic Crypto?!",
        # "The greatest of all Classics: Classic Rock: Don't Close Your Eyes by Kix is playing right now! Tune in at https://t.co/okWxACLJsq and  get PAID in #Ethereum Tokens then go trade for some old school Bitcoin. Hows that for a Classic Crypto?!",
        # "The greatest of all Classics: Classic Rock: Cherry Pie by Warrant is playing right now! Tune in at https://t.co/okWxACu8AS and  get PAID in #Ethereum Tokens then go trade for some old school Bitcoin. Hows that for a Classic Crypto?!",
        # "The greatest of all Classics: Classic Rock: Foreplay-Long Time by Boston is playing right now! Tune in at https://t.co/okWxACu8AS and  get PAID in #Ethereum Tokens then go trade for some old school Bitcoin. Hows that for a Classic Crypto?!",
        # "The greatest of all Classics: Classic Rock: Werewolves Of London by Warren Zevon is playing right now! Tune in at https://t.co/okWxACLJsq and  get PAID in #Ethereum Tokens then go trade for some old school Bitcoin. Hows that for a Classic Crypto?!",
        # "The greatest of all Classics: Classic Rock: Eric Clapton - Crossroads by Cream is playing right now! Tune in at https://t.co/okWxACLJsq and  get PAID in #Ethereum Tokens then go trade for some old school Bitcoin. Hows that for a Classic Crypto?!",
        # "The greatest of all Classics: Classic Rock: Tush by Zz Top is playing right now! Tune in at https://t.co/okWxACLJsq and  get PAID in #Ethereum Tokens then go trade for some old school Bitcoin. Hows that for a Classic Crypto?!",
        # "The greatest of all Classics: Classic Rock: Slow Ride by Foghat is playing right now! Tune in at https://t.co/okWxACLJsq and  get PAID in #Ethereum Tokens then go trade for some old school Bitcoin. Hows that for a Classic Crypto?!",
        # "The greatest of all Classics: Classic Rock: Slow Ride by Foghat is playing right now! Tune in at https://t.co/okWxACu8AS and  get PAID in #Ethereum Tokens then go trade for some old school Bitcoin. Hows that for a Classic Crypto?!",
        # "$ETH\nTrend reversal on 1h timeframe\nLast Price: 0.03058500 (Binance)\n24h Volume: 17510.21 BTC\n#BTC #Binance #ETH https://t.co/RA1NJl8MKg",
        # "\"Solidity v0.7 is out!  This is a breaking release.  If you're an Eth dev:\n\n\"Most notably, further cleanup of visibility and state mutability has been performed and several unpopular keywords have b...\"\n\nvia @blockfolio\n\n#ETH $ETH\n https://t.co/hrifjmETMc",
        # "Current #Ethereum safe low gas price: 41 Gwei\nChart(7d): https://t.co/oUmmnTmUHt\nChart(24h): https://t.co/C7h6IpqhkZ",
        # "@KriegVII Vitaliks father after taking too much Xtacy #eth $eth #ethereum pump it!  Music to go alone \ud83d\udd25\ud83d\ude06\ud83d\udd25\ud83d\ude06\ud83d\udd25\ud83d\ude06\ud83d\udd25\ud83d\ude06\ud83d\udd25\ud83e\udd23\ud83d\udd25\ud83e\udd23\ud83d\udd25\ud83d\ude0f\ud83d\udd25\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5https://t.co/zh3K2ZtltW #xrp #btc #ada #crypto #music",
        # "@KriegVII Vitaliks father after taking too much Xtacy #eth $eth #ethereum pump it!  Music to go alone \ud83d\udd25\ud83d\ude06\ud83d\udd25\ud83d\ude06\ud83d\udd25\ud83d\ude06\ud83d\udd25\ud83d\ude06\ud83d\udd25\ud83e\udd23\ud83d\udd25\ud83e\udd23\ud83d\udd25\ud83d\ude0f\ud83d\udd25\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5\ud83d\udca5https://t.co/zh3K2ZtltW #xrp #btc #ada #crypto #music",
        # "@KriegVII vitaliks dads really pumping ethereum.  #xrp #btc #ada $ada $xrp $btc #eth $eth #xrpcommunity #cardano \ud83d\udd25\ud83d\udd25\ud83d\udd25\ud83d\udd25\ud83d\udd25\ud83d\udd25\ud83d\udd25 https://t.co/GaO7Afudfd",
        # "#mining Cryptocurrencies Price target for BTC is 1200  #ETH - 0x3BC690c604d170BB53c155cb66074B4DE1172C36",
        # "\ud83d\udea8\u26a0\ufe0fALERT\u26a0\ufe0f\ud83d\udea8\n\ud83d\udc47\ud83c\udffd\nGive Away Crypto Online by Roger.\n#GiveAwayRoger #Giveaway #Giveaways #GiveawayAlert #giveawaycontest #GiveawayEntry #Giveawayalerts #earn #earnmoney #earncrypto #crypto #BTC #ETH #1UP #Swap $Swap #axionnetwork $axion #TrustSwap #Uptrennd \n\n\ud83d\udea8FOLLOW\ud83d\udea8\n@Rtrimini https://t.co/vXGvSowMAf",
        # "#CHAINLLINK price is recently growing up at a very fast rate. Suppose more people are #investing in it and pushing up the price. I had around 25$ assets in LINK and now they have gone upto 28$. Let's hope for its better future \ud83d\ude42 #crypto #LINK #bitcoin #ethereum #CryptoMining",
        # "Can anyone suggest me a good #platform to sell #bitcoin in #Indian at a very good price like around \u20b97,50,000 ?? I want to sell around 0.015 BTC and get a brand new #crypto #wallet. #crypto #bitcoin #ethereum #dogecoin",
        # "Here\u2019s Where the Earliest ETH Ever Mined Ended Up 5 Years Later - https://t.co/YpAaM0BbOX #Crypto #CryptoNews $BTC #ETH #Bitcoin",
        # "Bitcoin Price Soars to $11,400 as Traders Say a \u2018Bull Phase\u2019 Is Igniting - https://t.co/wE4g3AEJ3X #Crypto #CryptoNews $BTC #ETH #Bitcoin",
        # "A 17 Year Old Was Just Arrested in Connection With Twitter's Recent Hack - https://t.co/D4Rzy16o3y #Crypto #CryptoNews $BTC #ETH #Bitcoin",
        # "VeChain Ready to Awaken, VET Poised for Higher Highs - https://t.co/YIqHqEKP3l #Crypto #CryptoNews $BTC #ETH #Bitcoin",
        # "Bitcoin Ends July at Highest Monthly Close Since 2017 Peak - https://t.co/RwCt0MArPG #Crypto #CryptoNews $BTC #ETH #Bitcoin",
        # "Bitcoin on Track for Highest July Price Gain in 8 Years - https://t.co/21cfziK8g4 #Crypto #CryptoNews $BTC #ETH #Bitcoin",
        # "Elrond Launches Onto Mainnet, Reduces Token Supply by 99% - https://t.co/acpMcjIlcp #Crypto #CryptoNews $BTC #ETH #Bitcoin",
        # "Coinbase pushes DeFi narrative by considering listing Balancer, Curve, and AMPL - https://t.co/yD7PTTT4jR #Crypto #CryptoNews $BTC #ETH #Bitcoin",
        # "Bitcoin\u2019s Monthly Candle is About to See Its Most Bullish Close Since 2017 - https://t.co/s6NdJdoS11 #Crypto #CryptoNews $BTC #ETH #Bitcoin",
        # "Ethereum is Touching a Key Resistance Level, and It Could Spark a Drop to $280 - https://t.co/44s2HEjh78 #Crypto #CryptoNews $BTC #ETH #Bitcoin",
        # "Philippines tasks group to explore central bank digital currency - https://t.co/iOWXhe7dqW #Crypto #CryptoNews $BTC #ETH #Bitcoin",
        # "Officials Arrest 3 Allegedly Behind Twitter Hack #ethereum #crypto https://t.co/EFMT6iUZfj",
        # "Market Wrap: Bitcoin Pushes to $11,450, DeFi Value Locked Now at $4B #ethereum #crypto https://t.co/QSOTtEev5G",
        # "Bitcoin\u2019s Option Market Is Now Skewed Bullish #ethereum #crypto https://t.co/BpV4yAZk5E",
        # "Bitcoin Ends July at Highest Monthly Close Since 2017 Peak #ethereum #crypto https://t.co/UjvN9FBw0k",
        # "What a Professional Trader Thinks of the Fed, Robinhood and Real Estate, Feat. Tony Greer #ethereum #crypto https://t.co/cXFytni8L2",
        # "Dump your #shitcoins for #Ethereum lets make this move!!",
        # "DogData Airdrop is now Live\ud83d\ude80\ud83d\udcb0\ud83c\udfc6\n\nClick on below link to participate into this amazing #Airdrop\ud83c\udf81\n\nAirdrop link\ud83d\udc49\nhttps://t.co/5YUmgRfmmx\n\n\ud83c\udf81 Rewards: 5 + 2 BONE \ud83c\udf81\n\ud83d\udcb0 Value : $ 2.5\n\ud83d\udc65 Referral: $ 1\n\n#Airdrops #blockchain #cryptocurrency #ICO #bitcoin #Crypto #ETH",
        # "DogData Airdrop is now Live\ud83d\ude80\ud83d\udcb0\ud83c\udfc6\n\nClick on below link to participate into this amazing #Airdrop\ud83c\udf81\n\nAirdrop link\ud83d\udc49\nhttps://t.co/vMbfAYq3ve\n\n\ud83c\udf81 Rewards: 250 + 100 ETHBN \ud83c\udf81\n\ud83d\udcb0 Value : $ 2.5\n\ud83d\udc65 Referral: $ 1\n\n#Airdrops #blockchain #cryptocurrency #ICO #bitcoin #Crypto #ETH",
        # "-  $ETH Five Years of #Ethereum: From a Teenage Dream to a $38B Blockchain https://t.co/WdgLEmaaKm",
        # "I had 4 different people come ask me about cryptocurrency again randomly today. The last time that happened was just before bitcoin skyrocketed to 20k since then it\u2019s been crickets.\n\nThe big difference I see is there are many more options now.\n\nInvest wisely.\n\n#btc #eth #xrp #ada",
        # "#Overbit #SaturdaySitdown:\nBitcoin fundamentals continue to grow stronger\ud83d\udcaa, #Ethereum 2.0 is expecting updates in August\u2b06\ufe0f. Gold continues to glitter\u2728and the dollar loses its shine!\ud83d\ude1f\n\nhttps://t.co/Vpc28uxBEW\n#Bitcoin #MarginTrading #Forex #Crypto https://t.co/1SaAMly755",
        # "Just purchased F U T U R E // collectible from Danil Pan!! Bro The FUTURE is OURS !! \ud83d\ude42 https://t.co/zbHvoM3mVi #Rarible #Ethereum #ERC721 #Collectible #NFT via @rariblecom",
        # "#Ethereum short-term Price #Analysis: 31 July - #Altcoins #News #Social #TradingView - https://t.co/x5nYYv7WUF https://t.co/OznD7HNGs3",
        # "Utrust (UTK) - price action analysis, subscriber request. \ud83d\udcc8\ud83d\udcc9Subscribe for real time updates!!\n\nhttps://t.co/m5faTduuMU\n\n#Cryptocurrency #cryptotrading #ETH #Bitcoin #ERD #BTC #Bitcoin  #Binance #BNB #thepriceactionanalyst #priceanalyst #XVG #BancorNetwork #BNT #UTK #Utrust",
        # "Over the past 24 hours, the #ethereum price has renewed its annual maximum, reaching $345. The last time the asset price reached this level was in June 2019. https://t.co/qTBLvwlYqa",
        # "I\u2019ve lived through this tragedy, but some don\u2019t \ud83d\ude14 . Try $Hex out before you knock it. I don\u2019t even trade(I mean gamble) anymore and am so much happier. Thanks @RichardHeartWin \ud83d\ude4f #hexican #trustlessintrest #defi #eth #btc @tradingview https://t.co/0N58z3zLAM",
        # "DogData Airdrop is now Live\ud83d\ude80\ud83d\udcb0\ud83c\udfc6\n\nClick on below link to participate into this amazing #Airdrop\ud83c\udf81\n\nAirdrop link\ud83d\udc49\nhttps://t.co/8pPmJVssiC\n\n\ud83c\udf81 Rewards: 5 + 2 BONE \ud83c\udf81\n\ud83d\udcb0 Value : $ 2.5\n\ud83d\udc65 Referral: $ 1\n\n#Airdrops #blockchain #cryptocurrency #ICO #bitcoin #Crypto #ETH",
        # "@issaAfrican #VET -supply chain \n#BTC - first entry \n#LINK - Google supported \n#ETH - cheap buys",
        # "Fu@k it.. I\u2019m taking my profits and laughing to the the bank with a Cake! What\u2019s for dinner tonight? #BTC #ETH https://t.co/iP5u1A9s9S",
        # "Here's something that no one is talking about:\n\n#Cardano just beat #Ethereum to proof of stake.\n\nAfter multi-asset and smart contracts (coming out this year), Cardano will be the most capable #cryptocurrency in this industry! ETH fanboys will be waiting years for 2.0 to come out.",
        # "if my predictions are right, then Ethereum Classic (ETC, currently $7.36/share) is expected to rise to over $10 per share by the end of the year, and will be at over $23/share by 2022.\n\n#Ethereum #Bitcoin",
        # "#Ethereum hourly net stats.\n0am - 1am UTC\nblocks:257\navgDiff:2.4 PH\nhashRate:182.78 TH/s\navgSize:34.04 KB\ntxns:41518\n#ETH #BTC #crypto",
        # "#Ethereum hourly net stats.\n0am - 1am UTC\nblocks:257\navgDiff:2.4 PH\nhashRate:182.78 TH/s\navgSize:34.04 KB\ntxns:41518\n#ETH #BTC #crypto",
        # "I think $eth still has some room to climb tonight - spotting a Hidden Bullish Divergence on the hourly chart...\n\n#Ethereum #Bitcoin #Crypto",
        # "Indeed, the price of #ETH is growing rapidly recently! What do you think the $ETH price will be in this week? I'm joining the ETH price prediction on #OXBTC to win 1 ETH!Join NOW: https://t.co/v7rnqHNuNj Btw, sign up to earn FREE BTC! https://t.co/KeqpEgUXQp #ToTheMoon #BTC",
        # "Ethereum ETH Current Price:\n$344.48\n1 Hour: -0.28 % | 24 Hours: 3.36 % | 7 Days: 22.05 %\n#eth #ethereum",
        # "Ethereum ETH Current Price:\n$344.48\n1 Hour: -0.28 % | 24 Hours: 3.36 % | 7 Days: 22.05 %\n#eth #ethereum",
        # "Hellooow August, may the Corrections aren't too Deep\ud83d\ude05\n#xrp #eth #btc #ltc https://t.co/SB2jDyiH6T",
        # "#BTC &amp; #Ethereum is going to high. Lucky for those who had bought this cyrpto at starting price below Rp 2 Mio last year. \n\nThey're smiling now. \n\nDon't forget to topup other crypto currencies like Tether who has a low price 0,63%. It seems they're waiting to gain more profit. https://t.co/9jzgjq2isw",
        # "Five Years of #ethereum: From a Teenage Dream to a $38B #blockchain $ETH https://t.co/vLpcPJNrS7 https://t.co/bYFe0ZSF5L",
        # "Analyst: Watch Which Altcoins Hold Against Bitcoin For Surprising Springboard Recovery https://t.co/Zk0QDLFNw8 #Ethereum #XRP",
        # "@SkrillCrypto #Ethereum is definitely my best altcoin because it has better technology and provides more uses than Bitcoin does therefore serves a bigger purpose, and also because Ether transactions tend to be confirmed quicker by the blockchain. \n\n@SwertengMapalad",
        # "Ethereum DEX Processing 1/3 Volume of Coinbase Shows Highly Optimistic Trend #Ethereum via https://t.co/uXU6YZKcWt https://t.co/CYDFJEA91n",
        # "\u24c2\ud83d\udcc8#Binance: #BAT-BTC (#6h)\nMoving Stop-Loss!!!\n\u27a10.00002139 -&gt; 0.00002240\n\n#basicattentiontoken #margin #currency #tradingbot #trading #ethereum",
        # "Market Update: The price of #Ethereum is currently $344.87\n #Crypto $ETH #ETH\n View the coin here here: https://t.co/JCJIcBfJRF",
        # "Market Update: The price of #Ethereum is currently $344.87\n #Crypto $ETH #ETH\n View the coin here here: https://t.co/JCJIcBfJRF",
        # "Order your secure and smart BTC/ETH/Altcoin hardware wallet - Only 94.80 EUR https://t.co/WC9nt6QCb7 #bitcoin #btc #eth #altcoin 21:20 https://t.co/MZ95u2DhA3",
        # "1 Bitcoin ( #BTC )\nDollar: 11,279.76$ \n\n1 Bitcoin Cash ( #BCH )\nDollar: 3,00.37$ \n\n1 Ethereum ( #ETH )\nDollar: 345.48$ \n\n1 Ripple ( #XRP )\nDollar: 0.25722$ \n\nDate: 1 Aug 2020 1:48\n\n#Bitcoin #Ethereum #cryptocurrency #crypto #altcoin #Blockchain #Ripple",
        # "1 Bitcoin ( #BTC )\nDollar: 11,279.76$ \n\n1 Bitcoin Cash ( #BCH )\nDollar: 3,00.37$ \n\n1 Ethereum ( #ETH )\nDollar: 345.48$ \n\n1 Ripple ( #XRP )\nDollar: 0.25722$ \n\nDate: 1 Aug 2020 1:48\n\n#Bitcoin #Ethereum #cryptocurrency #crypto #altcoin #Blockchain #Ripple",
        # "Latium #Freelancing is an online jobs platform where users can work &amp; hire using digital #currencies such as #Bitcoin and #Ethereum. Register now https://t.co/qjCNpfaPIc",
        # "Here\u2019s Where the Earliest ETH Ever Mined Ended Up 5 Years Later (Cointelegraph) \n\nCoinfirm details what happened to the first mined #Ethereum after the asset's launch in 2015.\n\nhttps://t.co/xxCrTZjIHv https://t.co/fH4UqBI6r2",
        # "\ud83d\udd04 Prices update in $USD (1 hour):\n\n$BTC   - 11305.9 $   (+0.28 %)\n$ETH   - 346.66 $    (+0.4 %) \n$XRP   - 0.26 $      (+0.12 %)\n$BCH   - 300.0 $     (+0.17 %)\n$LTC   - 58.13 $     (+0.64 %)\n\n#USD #Dollar #BTC #ETH #XRP #BCH #LTC #Trading #Bitcoin #Ethereum",
        # "\ud83d\udd04 Prices update in $USD (1 hour):\n\n$BTC   - 11305.9 $   (+0.28 %)\n$ETH   - 346.66 $    (+0.4 %) \n$XRP   - 0.26 $      (+0.12 %)\n$BCH   - 300.0 $     (+0.17 %)\n$LTC   - 58.13 $     (+0.64 %)\n\n#USD #Dollar #BTC #ETH #XRP #BCH #LTC #Trading #Bitcoin #Ethereum",
        # "$DGN UniSwap pool is now live!!!\n\nThis is the official pool from the dev team. Be sure to use the right pool as there is a fake pool.\n\n=&gt; https://t.co/RGaYGxfBGo\n\nAll #tokens have been sent!\n\n#DGN #UniSwap #DeFi #UNC #ETH #ERC20 #Crypto #DYOR \n\n@DegeneratePlat1 https://t.co/KZaUbUgAwc",
        # "Good morning hassil hard pa tayo! \ud83d\ude0d #Ethereum #myethjourney #businessopportunity https://t.co/EXI4E6Po5I",
        # "How crypto coin wars begin\n#crypto #btc #bnb #dgb #trx #hex #ltc #eth #bitcoin https://t.co/EySf7OGUfU",
        # "Breathing new life to an old piece as a crypto collectible - high resolution file included so you can print it if you like. :) Presenting: Magic Missy Never Misses! https://t.co/GeWyB7qjeA #Rarible #Ethereum #ERC721 #Collectible #NFT #Cryptoart  via @rariblecom",
        # "The current price of #eth is $343.96, a decrease of -0.8%, a 24 hr volume of $12.04B, and a market cap of $38.25B",
        # "Part 1/2:\n\nACX\n#BTC: $13200\n#BCH: $450\n#ETH: $200\n#LTC: $119\n#XRP: $0.40\n\nBTCMarkets\n#BTC: $15830\n#BCH: $420\n#ETH: $484\n#LTC: $81\n#XRP: $0.36\n\nCoinspot\n#BTC: $15870\n#ETH: $489\n#LTC: $82\n#XRP: $0.36\n\nCointree\n#BTC: $15693\n#BCH: $410\n#ETH: $471\n#LTC: $79\n#XRP: $0.35",
        # "\"Solidity v0.7 is out!  This is a breaking release.  If you're an Eth dev:\n\n\"Most notably, further cleanup of visibility and state mutability has been performed and several unpopular keywords have b...\"\n\nvia @blockfolio\n\n#ETH $ETH\n https://t.co/q1dEPMxWtc",
        # "#Coinbase Considers Listing #Ethereum #DeFi Tokens: Ampleforth, Balancer, Band, Curve, Kava\n\nhttps://t.co/hOrGkJETa3 https://t.co/4vCjcJj0Da",
        # "@RevolutApp  should be banned from crypto #eth #btc #xrp #ltc",
        # "$CBR in biggest gainers @CoinMarketCap !!!\n\n#uniswap #defi #token #eth #btc #Crypto https://t.co/jyuGA5s4bX",
        # "#ETH #Ethereum just getting started https://t.co/qW3KPLxcRI",
        # "Cryptocurrencies second stage bull market has begun. \n$BTC #Bitcoin $ETH #Ethereum $LINK #Chainlink https://t.co/TR8RfYspaj",
        # "Cryptocurrencies second stage bull market has begun. \n$BTC #Bitcoin $ETH #Ethereum $LINK #Chainlink\n\nWeekly chart: https://t.co/1Fin3tILHz",
        # "Find out how making a living during #covid is easy! https://t.co/1YjWxLeVTD #eth #workfromhome",
        # "DogData Airdrop is now Live\ud83d\ude80\ud83d\udcb0\ud83c\udfc6\n\nClick on below link to participate into this amazing #Airdrop\ud83c\udf81\n\nAirdrop link\ud83d\udc49\nhttps://t.co/jNG1jYM2xk\n\n\ud83c\udf81 Rewards: 5 + 2 BONE \ud83c\udf81\n\ud83d\udcb0 Value : $ 2.5\n\ud83d\udc65 Referral: $ 1\n\n#Airdrops #blockchain #cryptocurrency #ICO #bitcoin #Crypto #ETH",
        # "Thinking of what to do once there\u2019s another bull run on crypto will be the most stressful decision ever. #xrp #Ethereum  #Bitcoin #cryptocurency @RealSteveHenry",
        # "#REN/BTC and new #Signal\n\nSL, TP, TPL &amp; more, was published on our telegram channel (link in bio).\n\nFollow &amp; retweet to support #crypto_traders' community\n\n#Stay_at_home and check us every 15 min.\n\n#cryptocurrencies #BTC #ethereum #Trading #Scalping",
        # "#ADA/BTC\n#ZRX/BTC\n#CTSI/BTC\n\nPls get SL,TP,TPL &amp; more, from our telegram channel (link in bio).\n\n#Stay_at_home &amp; check us every 15 min to #scalp &amp; enjoy.\n\n#cryptocurrency #BTC #ethereum #Trading #Binance",
        # "#STRAT/BTC\n#POWR/BTC\n\nPls get SL,TP,TPL &amp; more, from our telegram channel (link in bio).\n\n#Stay_at_home &amp; check us every 15 min to #scalp &amp; enjoy.\n\n#cryptocurrency #BTC #ethereum #Trading #Binance",
        # "#RUNE/BTC and new #Signal\n\nSL, TP, TPL &amp; more, was published on our telegram channel (link in bio).\n\nFollow &amp; retweet to support #crypto_traders' community\n\n#Stay_at_home and check us every 15 min.\n\n#cryptocurrencies #BTC #ethereum #Trading #Scalping",
        # "Short $btc #btc and $eth #eth \ud83e\udd14",
        # "Check all here --&gt; https://t.co/S1I0bD8b2s \nOn #Poloniex #NXT to #BTC -- &gt; 0.00000111, volume --&gt; 6591687 \nOn #Cex.io #ETH to #GBP -- &gt; 264.19, volume --&gt; 78 \nOn #Independent Reserve #ETH to #AUD -- &gt; 485.28, volume --&gt; 2361 \nOn #Livecoin #PPC to #USD -- &gt; 0.30904, volume --&gt; 117 https://t.co/KmsWLyNItu",
        # "Check all here --&gt; https://t.co/S1I0bD8b2s \nOn #YoBit #PPC to #ETH -- &gt; 0.003546, volume --&gt; 31 \nOn #Exmo #LTC to #BTC -- &gt; 0.00512843, volume --&gt; 5716 \nOn #wexnz #DASH to #RUB -- &gt; 16121.813, volume --&gt; 351 \nOn #Coinbase Pro #XRP to #BTC -- &gt; 0.00002279, volume --&gt; 2266145 https://t.co/Wul9olp7XP",
        # "Check all here --&gt; https://t.co/S1I0bD8b2s \nOn #wexnz #DASH to #USD -- &gt; 288.75307, volume --&gt; 1732 \nOn #Coinbase Pro #ETH to #BTC -- &gt; 0.03058, volume --&gt; 14236 \nOn #Kraken #ETH to #GBP -- &gt; 264.3, volume --&gt; 2 \nOn #Coinbase Pro #XRP to #GBP -- &gt; 0.1966, volume --&gt; 30076 https://t.co/S4cFunIcv1",
        # "Where can we get support for the ETH2 launchpad? #reddit #ethereum #ETH #cryptocurrency https://t.co/zFeewEWj4G",
        # "Where can we get support for the ETH2 launchpad? #reddit #ethereum #ETH #cryptocurrency https://t.co/zFeewEWj4G",
        # "Requesting faucet funds into 0x3336696Cf94e90d34FABF2689d247DE829C4D73b on the #Rinkeby #Ethereum test network."
        "to the moon!",
        "to the moon"
    ]
    for s in test_sentences:
        print(s.encode('utf-8', 'replace').decode('utf-8'))
        result = get_sentiment2(model_roberta_base, s)
        print(result)
else:
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
            empty = 0.00000
            sadness = 0.00000
            enthusiasm = 0.00000
            neutral = 0.00000
            worry = 0.00000
            surprise = 0.00000
            love = 0.00000
            fun = 0.00000
            hate = 0.00000
            happiness = 0.00000
            boredom = 0.00000
            relief = 0.00000
            anger = 0.00000
            this_skiped = []
            handled = 0
            cleaned_text = [clean_text(i) for i in json_data['content']]

            x_test1 = regular_encode(np.array(cleaned_text), tokenizer, maxlen=max_len)
            test1 = (tf.data.Dataset.from_tensor_slices(x_test1).batch(128))
            sentiment = model_roberta_base.predict(test1, verbose=0)
            # classes = np.argmax(sentiment, axis=1)
            # print(sentiment)
            # sent = np.round(np.dot(sentiment, 100).tolist(), 0)[0]
            # result = pd.DataFrame([sent_to_id.keys(), sent]).T
            # result.columns = ["sentiment", "percentage"]
            # result = result[result.percentage != 0]

            for j in range(len(sentiment)):
                for i in range(len(sentiment[j])):
                    try:
                        # result = get_sentiment2(model_roberta_base, text)
                        if 0 == i:
                            empty += sentiment[j][0]
                        if 1 == i:
                            sadness += sentiment[j][1]
                        if 2 == i:
                            enthusiasm += sentiment[j][2]
                        if 3 == i:
                            neutral += sentiment[j][3]
                        if 4 == i:
                            worry += sentiment[j][4]
                        if 5 == i:
                            surprise += sentiment[j][5]
                        if 6 == i:
                            love += sentiment[j][6]
                        if 7 == i:
                            fun += sentiment[j][7]
                        if 8 == i:
                            hate += sentiment[j][8]
                        if 9 == i:
                            happiness += sentiment[j][9]
                        if 10 == i:
                            boredom += sentiment[j][10]
                        if 11 == i:
                            relief += sentiment[j][11]
                        if 12 == i:
                            anger += sentiment[j][12]
                    except Exception as e:
                        this_skiped.append(json_data['content'][i])
                handled += 1
                count += 1
                # print(handled)
            if handled > 0:
                size = float(handled)
                empty = np.round(float(empty / size), 4)
                sadness = np.round(float(sadness / size), 4)
                enthusiasm = np.round(float(enthusiasm / size), 4)
                neutral = np.round(float(neutral / size), 4)
                worry = np.round(float(worry / size), 4)
                surprise = np.round(float(surprise / size), 4)
                love = np.round(float(love / size), 4)
                fun = np.round(float(fun / size), 4)
                hate = np.round(float(hate / size), 4)
                happiness = np.round(float(happiness / size), 4)
                boredom = np.round(float(boredom / size), 4)
                relief = np.round(float(relief / size), 4)
                anger = np.round(float(anger / size), 4)

                datetime.append(json_data['date'])
                empty_bucket.append(empty)
                sadness_bucket.append(sadness)
                enthusiasm_bucket.append(enthusiasm)
                neutral_bucket.append(neutral)
                worry_bucket.append(worry)
                surprise_bucket.append(surprise)
                love_bucket.append(love)
                fun_bucket.append(fun)
                hate_bucket.append(hate)
                happiness_bucket.append(happiness)
                boredom_bucket.append(boredom)
                relief_bucket.append(relief)
                anger_bucket.append(anger)

            if len(this_skiped) > 0:
                skipped_datetime.append(json_data['date'])
                skipped_content_buckets.append(this_skiped)
            # if count == 1:
            #    break
            print(count)
        df = pd.DataFrame(
            {'date': datetime, 'empty': empty_bucket, 'sadness': sadness_bucket, 'enthusiasm': enthusiasm_bucket,
             'neutral': neutral_bucket, 'worry': worry_bucket, 'surprise': surprise_bucket, 'love': love_bucket,
             'fun': fun_bucket, 'hate': hate_bucket, 'happiness': happiness_bucket, 'boredom': boredom_bucket,
             'relief': relief_bucket, 'anger': anger_bucket})
        with open("./predict_bert.json", "a") as myfile:
            for index, row in df.iterrows():
                myfile.write(json.dumps(row.to_dict()))
                myfile.write('\n')
        df = pd.DataFrame(
            {'date': skipped_datetime, 'contents': skipped_content_buckets})
        with open("./skipped_bert.json", "a") as myfile:
            for index, row in df.iterrows():
                myfile.write(json.dumps(row.to_dict()))
                myfile.write('\n')
