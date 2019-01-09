

def main_func(screen_name, tweet_num=5):
    from watson_developer_cloud import NaturalLanguageUnderstandingV1
    from watson_developer_cloud.natural_language_understanding_v1 \
        import Features, EntitiesOptions, KeywordsOptions, CategoriesOptions, SentimentOptions
    import tweepy
    import pandas as pd
    from IPython.display import display, HTML
    pd.set_option('display.max_columns', 7)

    consumer_key = 'NgbszsMy18esxzBRpnS6YJSg5'
    consumer_secret = 'fUlGwElm7B7Q5UUl99TdnMewBA3xW9Cw5xmzBAq1xU9j5O6wUa'
    access_key = '3847979172-1TNy6qbn1DvF2lHuUMpM86hAyRSxN8Uc9WpZzET'  # access_token
    access_secret = 'ZCooGbFqAqxCyFtZGqMPczAhD6IkZW1TfT1hocKVPm8pV'

    naturalLanguageUnderstanding = NaturalLanguageUnderstandingV1(
        version='2018-11-16',
        iam_apikey='mPjOW939Q8rvnvXfXIVhyvgOhk76aA2PCs_DCqvvUOda',
        url='https://gateway-lon.watsonplatform.net/natural-language-understanding/api')
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    columns2 = ['Tweet', 'Sentiment', 'Keywords', 'Entities', 'Сategory']
    watson_df = pd.DataFrame(columns=columns2)
    all_raw_text = str()

    index = 0
    for status in tweepy.Cursor(api.user_timeline, screen_name=screen_name, tweet_mode="extended").items():
        all_raw_text += str(status.full_text + ' ')

        response = naturalLanguageUnderstanding.analyze(
            text=status.full_text,
            features=Features(
                entities=EntitiesOptions(emotion=True, limit=3),
                keywords=KeywordsOptions(emotion=True, limit=3),
                sentiment=SentimentOptions(),
                categories=CategoriesOptions(limit=1))).get_result()

        keywords = str()
        for word in response['keywords']:  # ключевые слова
            # keywords.append(word['text'])
            if len(str(word['text'])) > 0:
                keywords += (str(word['text']) + ', ')

        entities = str()
        for ent in response['entities']:  # ключевые слова
            if len(str(ent['text'])) > 0:
                entities += (str(ent['text']) + '(' + str(ent['type']) + ')' + ', ')

        watson_df.loc[index] = [status.full_text, response['sentiment']['document']['label'],
                                keywords, entities, response['categories'][0]['label'][1:]]

        '''response_overal = natural_language_understanding.analyze(
            url='www.wsj.com/news/markets',
            features=Features(sentiment=SentimentOptions(targets=['stocks']))).get_result()'''

        index += 1
        if index == tweet_num:
            break

    display(HTML(watson_df.to_html()))
