# chapter 15 offers more of suggestions and code examples then full-blown pipelines - this means that to actually run all the code, one needs to set up appropriate datasets.

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X, labels)

from textblob import TextBlob
analysis = TextBlob(text)
Pos_or_neg = analysis.sentiment.polarity

doc = nlp(text)
sentiment_value = doc.sentiment

def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top10)))


# mining data from reddit

import requests
import json

# Add your username below
hdr = {'User-Agent': ‘:r/news.single.result:v1.0' +
       '(by /u/username)'}
url = 'https://www.reddit.com/r/news/.json'
req = requests.get(url, headers=hdr)
data = json.loads(req.text)


# mining data from twitter

import tweepy

# Authentication and access using keys:
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

# Return API with authentication:
api = tweepy.API(auth)

tweets = api.user_timeline(screen_name="realDonaldTrump", count=20)
For tweet in tweets:
	print(tweet.text)

tweets = api.get_tweets(query = 'Donald Trump', count = 200)

from chatterbot import ChatBot
bot = ChatBot('Stephen')
bot.train([
    'How are you?',
    'I am good.',
    'That is good to hear.',
    'Thank you',
    'You are welcome.',
])
