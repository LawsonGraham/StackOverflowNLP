from imports import *
#THIS FIRST CHUNK OF CODE IS EXPLAINED IN nlp.py
dat = pd.read_csv('Questions.csv')
dat = dat.drop(columns=['Id', 'OwnerUserId'])
positive = dat[dat['Score'] > 0]
positive.sample(frac=1)
positive = positive[:45000]
negative = dat[dat['Score'] < 0]
frames = [positive, negative]
result = pd.concat(frames)
result = result.drop(columns=['CreationDate'])
from collections import Counter
def word_counter(sentence):
    word_count = len(sentence.split())
    return word_count
title_length = result['Title'].apply(lambda x: word_counter(x))
body_length = result['Body'].apply(lambda x: word_counter(x))
result['Word length title'] = title_length.tolist()
result['Word length body'] = body_length.tolist()
from gensim.parsing.preprocessing import remove_stopwords
def stop_word_counter(text):
    filtered_sentence = remove_stopwords(text) 
    filtered_sentence_length = word_counter(filtered_sentence)
    stop_word_count = word_counter(text) - filtered_sentence_length
    return stop_word_count
title_stop_length = result['Title'].apply(lambda x: stop_word_counter(x))
body_stop_length = result['Body'].apply(lambda x: stop_word_counter(x))
result['Stop Word length title'] = title_stop_length.tolist()
result['Stop Word length body'] = body_stop_length.tolist()
title_stop_prop = result['Title'].apply(lambda x: (stop_word_counter(x)/word_counter(x))*100)
body_stop_prop = result['Body'].apply(lambda x: (stop_word_counter(x)/word_counter(x))*100)
title_len = result['Title'].apply(lambda x: len(x))
body_len = result['Body'].apply(lambda x: len(x))
result['Stop Word proportion (title)'] = title_stop_prop.tolist()
result['Stop Word proportion (body)'] = body_stop_prop.tolist()
result['No. of characters (title)'] = title_len.tolist()
result['No. of characters (body)'] = body_len.tolist()
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
title_sentlen = result['Title'].apply(lambda x: len(sent_tokenize(x)))
body_sentlen = result['Body'].apply(lambda x: len(sent_tokenize(x)))
result['No. of sentences (title)'] = title_sentlen.tolist()
result['No. of sentences (body)'] = body_sentlen.tolist()
def convert_to_0and1(num):
    if num>=-0:
        return 1
    else:
        return -1
Scoreclassify = result['Score'].apply(lambda x: convert_to_0and1(x))
result['Scores_classified'] = Scoreclassify.tolist()

# - VISUALIZATIONS - #

#Visualizations begin here. 
plt.scatter(result['Word length title'], result['Score'])

plt.scatter(result['Word length body'], result['Score'])

plt.scatter(result['Stop Word length title'], result['Score'])

plt.scatter(result['Stop Word length body'], result['Score'])

plt.scatter(result['Stop Word proportion (body)'], result['Score'])

plt.scatter(result['Stop Word proportion (title)'], result['Score'])

plt.scatter(result['No. of characters (title)'], result['Score'])

plt.scatter(result['No. of characters (body)'], result['Score'])

plt.scatter(result['No. of sentences (title)'], result['Score'])

plt.scatter(result['No. of sentences (body)'], result['Score'])