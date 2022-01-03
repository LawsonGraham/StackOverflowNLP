from imports import *


### - DATA COLLECTION AND CLEANING - ###


#Importing the data
dat = pd.read_csv('Questions.csv')
dat = dat.drop(columns=['Id', 'OwnerUserId'])

#Sampling 45000 +ve rated questions and storing positive rated questions in 'positive' table.
positive = dat[dat['Score'] > 0]
positive.sample(frac=1)
positive = positive[:45000]

#Storing all negative rated questions in 'negative' table.
negative = dat[dat['Score'] < 0]

#Joining the positive and negative tables in a new table 'result'. Now this new dataset has a good balance of positive and negative rated questions.
frames = [positive, negative]
result = pd.concat(frames)
# Removing Creation Date column as it is not in a format that can be used in the model. 
result = result.drop(columns=['CreationDate'])


### - DECIDING FEATURES - ###


#Deciding features (attributes) for the model begins here. 
#In the next two cells we are adding two features: Number of words in title and Number of words in body.
from collections import Counter

#word counting function

def word_counter(sentence):
    word_count = len(sentence.split())
    return word_count

#title_length is an array which contains the number of words in each entry of the 'Title' column of the result table
#body_length is an array which contains the number of words in each entry of the 'Title' column of the result table

title_length = result['Title'].apply(lambda x: word_counter(x))
body_length = result['Body'].apply(lambda x: word_counter(x))
result['Word length title'] = title_length.tolist()
result['Word length body'] = body_length.tolist()

from gensim.parsing.preprocessing import remove_stopwords
# In the next two cells, we are adding two features. Number of stop words in the title and body respectively.

#Function which counts the number of stop words in a string
def stop_word_counter(text):
    filtered_sentence = remove_stopwords(text) 
    filtered_sentence_length = word_counter(filtered_sentence)
    stop_word_count = word_counter(text) - filtered_sentence_length
    return stop_word_count

#Adding the number of stop words in body and title of the stack overflow question as columns of the table.
title_stop_length = result['Title'].apply(lambda x: stop_word_counter(x))
body_stop_length = result['Body'].apply(lambda x: stop_word_counter(x))
result['Stop Word length title'] = title_stop_length.tolist()
result['Stop Word length body'] = body_stop_length.tolist()

#Adding four more features, proportion of stop words in body, proportion of stop words in title, number of characters in body and title
title_stop_prop = result['Title'].apply(lambda x: (stop_word_counter(x)/word_counter(x))*100)
body_stop_prop = result['Body'].apply(lambda x: (stop_word_counter(x)/word_counter(x))*100)
title_len = result['Title'].apply(lambda x: len(x))
body_len = result['Body'].apply(lambda x: len(x))
result['Stop Word proportion (title)'] = title_stop_prop.tolist()
result['Stop Word proportion (body)'] = body_stop_prop.tolist()
result['No. of characters (title)'] = title_len.tolist()
result['No. of characters (body)'] = body_len.tolist()

#Adding two more features, number of sentences in body and title each
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


### - MACHINE LEARNING - ###


#Machine Learning begins here
#Splitting our dataset into training and testing set. Ratio: 70:30
training_data, test_data = train_test_split(result, test_size=0.3, random_state=2)

# Training and testing 
X_train = training_data.drop(columns='Score').drop(columns='Scores_classified').drop(columns='Title').drop(columns='Body')[['Word length title','Word length body','Stop Word length title','Stop Word length body','Stop Word proportion (title)', 'Stop Word proportion (body)', 'No. of characters (title)', 'No. of sentences (title)', 'No. of sentences (body)']]
y_train = training_data['Scores_classified']

X_test = test_data.drop(columns='Score').drop(columns='Scores_classified').drop(columns='Title').drop(columns='Body')[['Word length title','Word length body','Stop Word length title','Stop Word length body','Stop Word proportion (title)', 'Stop Word proportion (body)', 'No. of characters (title)', 'No. of sentences (title)', 'No. of sentences (body)']]
y_test = test_data['Scores_classified']

#Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

#Printing the accuracy of our model
y_predicted_logistic = logistic_model.predict(X_test)
accuracy = metrics.accuracy_score(y_predicted_logistic,y_test)
print("Accuracy : %s" % "{0:.3%}".format(accuracy))