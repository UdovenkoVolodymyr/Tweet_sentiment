
def main_func(user_screen_name, tweets_count):
    from apiclient import discovery
    from httplib2 import Http
    import oauth2client
    from oauth2client import file, client, tools
    import io
    from googleapiclient.http import MediaIoBaseDownload
    import tweepy
    import csv
    import pandas as pd
    from bs4 import BeautifulSoup
    from nltk.tokenize import WordPunctTokenizer
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.stem.lancaster import LancasterStemmer
    from gensim.models import Word2Vec
    import multiprocessing
    from nltk.corpus import stopwords
    import re
    import pickle
    import numpy as np
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import tensorflow as tf
    from tensorflow.contrib.tensorboard.plugins import projector
    import numpy as np
    import gensim
    import os
    import warnings
    import nltk
    warnings.filterwarnings(action='ignore')
    pd.set_option('display.max_columns', 7)
    nltk.download('stopwords')
    nltk.download('wordnet')

    

    obj = lambda: None
    lmao = {"auth_host_name": 'localhost', 'noauth_local_webserver': 'store_true', 'auth_host_port': [8080, 8090],
            'logging_level': 'ERROR'}
    for k, v in lmao.items():
        setattr(obj, k, v)

    SCOPES = 'https://www.googleapis.com/auth/drive.readonly'
    store = file.Storage('token.json')
    creds = store.get()

    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('client_id.json', SCOPES)
        creds = tools.run_flow(flow, store, obj)

    DRIVE = discovery.build('drive', 'v3', http=creds.authorize(Http()))

    file_id = '1gN9u4zFWfwR5n-LmBwrcwmNGIUKj4Y0F'
    request = DRIVE.files().get_media(fileId=file_id)

    fh = io.FileIO('lemmatization_nolim_all.sav', mode='w')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))

    predict_model = 'lemmatization_nolim_all.sav'
    parent_patch = os.getcwd()
    path = "test"
    stopWords = set(stopwords.words('english'))

    consumer_key = 'NgbszsMy18esxzBRpnS6YJSg5'
    consumer_secret = 'fUlGwElm7B7Q5UUl99TdnMewBA3xW9Cw5xmzBAq1xU9j5O6wUa'
    access_key = '3847979172-1TNy6qbn1DvF2lHuUMpM86hAyRSxN8Uc9WpZzET'  # access_token
    access_secret = 'ZCooGbFqAqxCyFtZGqMPczAhD6IkZW1TfT1hocKVPm8pV'

    tok = WordPunctTokenizer()

    pat1 = r'@[A-Za-z0-9_]+'
    pat2 = r'https?://[^ ]+'
    combined_pat = r'|'.join((pat1, pat2))
    www_pat = r'www.[^ ]+'
    negations_dic = {"isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
                     "haven't": "have not", "hasn't": "has not", "hadn't": "had not", "won't": "will not",
                     "wouldn't": "would not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
                     "can't": "can not", "couldn't": "could not", "shouldn't": "should not", "mightn't": "might not",
                     "mustn't": "must not"}
    neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

    def tweet_cleaner_updated(text, tweet_len=100):
        soup = BeautifulSoup(text, 'lxml')
        souped = soup.get_text()
        try:
            bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
        except:
            bom_removed = souped
        stripped = re.sub(combined_pat, '', bom_removed)
        stripped = re.sub(www_pat, '', stripped)
        lower_case = stripped.lower()
        neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
        letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)

        lema = WordNetLemmatizer()
        lancaster_stemmer = LancasterStemmer()
        words = list()
        for word in tok.tokenize(letters_only):
            if len(word) > 1 and word not in stopWords:
                # print('raw', word)
                lema_word = lema.lemmatize(word)
                # lema_word = lancaster_stemmer.stem(word)
                if len(lema_word) == 1:
                    lema_word = word
                # print('lem', lema_word)
                words.append(lema_word)
        if len(words) <= tweet_len:
            return words, (" ".join(words)).strip()

    def get_api_clean_tweets_df(screen_name, tweet_num=5, predict_model=''):
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_key, access_secret)
        api = tweepy.API(auth)

        columns = ['Screen_Name', 'Time_Stamp', 'raw_tweet', 'text', 'tokens']
        tweet_df = pd.DataFrame(columns=columns)
        tweet_tokeenized = list()
        tweet_tokens = list()
        positive_tokens = list()
        tokens_positiveti_dict = dict()

        index = 0
        for status in tweepy.Cursor(api.user_timeline, screen_name=screen_name, tweet_mode="extended").items():
            clean_tweet = tweet_cleaner_updated(status.full_text)
            tweet_tokens += clean_tweet[0]
            tweet_tokeenized.append(clean_tweet[0])
            tweet_df.loc[index] = [status.user.screen_name, status.created_at,
                                   status.full_text, clean_tweet[1], clean_tweet[0]]
            index += 1
            if index == tweet_num:
                break

        tweet_df['target'] = None

        loaded_model = pickle.load(open(predict_model, 'rb'))

        for index in range(len(tweet_df)):
            if tweet_df['text'][index] is np.nan:
                pass
            else:
                if int(loaded_model.predict(list([tweet_df['text'][index]]))) == 1:
                    tweet_df['target'][index] = 'Postive'
                else:
                    tweet_df['target'][index] = 'Negative'

        positive_tweet_df = tweet_df.loc[tweet_df['target'] == 'Postive']
        for row in positive_tweet_df.index:  # range(len(positive_tweet_df)):
            positive_tokens += positive_tweet_df['tokens'][row]

        tweet_tokens_set = set(tweet_tokens)
        for token in tweet_tokens_set:
            token_all_counter = tweet_tokens.count(token)
            token_pos_counter = positive_tokens.count(token)
            tokens_positiveti_dict[token] = int(token_pos_counter / token_all_counter * 100)

        tweet_df.drop(['Screen_Name', 'Time_Stamp', 'text', 'tokens'], axis=1, inplace=True)
        print(tweet_df)
        return tokens_positiveti_dict, tweet_tokens, tweet_tokeenized, tweet_df

    pos_tokens_dict, donalds_tokens_list, donalds_tokenized_tweets, donalds_df = \
        get_api_clean_tweets_df(user_screen_name, tweets_count, predict_model)

    cores = multiprocessing.cpu_count()

    user_model = Word2Vec(donalds_tokenized_tweets, min_count=1, size=200, workers=cores, )
    user_model.save("user_model")
    model = gensim.models.keyedvectors.KeyedVectors.load("user_model")

    max_size = len(model.wv.vocab) - 1
    w2v = np.zeros((max_size, model.layer1_size))

    with open("test/metadata.tsv", 'w+') as file_metadata:
        meta_word = ('word' + '\t' + 'Sentiment')
        file_metadata.write(meta_word + '\n')

    with open('tensors.tsv', 'w+') as tensors:
        with open("test/metadata.tsv", 'a') as file_metadata:
            for i, word in enumerate(model.wv.index2word[:max_size]):
                w2v[i] = model.wv[word]
                if pos_tokens_dict[word] < 50:
                    meta_word = word + '(' + str(pos_tokens_dict[word]) + ')' + '\t' + str(0)
                    file_metadata.write(meta_word + '\n')
                else:
                    meta_word = word + '(' + str(pos_tokens_dict[word]) + ')' + '\t' + str(100)
                    file_metadata.write(meta_word + '\n')
                vector_row = '\t'.join(map(str, model[word]))
                tensors.write(vector_row + '\n')

    sess = tf.InteractiveSession()

    with tf.device("/cpu:0"):
        embedding = tf.Variable(w2v, trainable=False, name='embedding')
        print(embedding)

    tf.global_variables_initializer().run()

    saver = tf.train.Saver()

    writer = tf.summary.FileWriter(path, sess.graph)

    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'embedding'
    embed.metadata_path = 'metadata.tsv'

    projector.visualize_embeddings(writer, config)

    saver.save(sess, path + '/model.ckpt', global_step=max_size)
