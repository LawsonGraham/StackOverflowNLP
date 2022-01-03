# StackOverflowNLP
## How to post good questions on Stack Overflow?

We all know that Stack Overflow can be an extremely toxic place for new programmers. When new programmers ask questions on the platform, they get trolled and receive unhelpful answers if the community interprets the post as a 'bad' question. This discourages new coders from learning and raises the entry bar to programming.

We decided to build a Machine Learning model to classify a question as a good question to be posted or not. This model can serve as a great foundation for more advanced applications like giving recommendations to programmers to ask better questions so that they are less likely to have a bad experience and more importantly they get their questions answered without receiving unnecessary trolling and negativity.

In this project, we will be using Logistic Regression to predict whether a question asked on the well known Stack Overflow platform will receive positive or negative votes.

## Setup:
You must install a few libraries to have the scripts run properly - run these installation commands:
pip install datascience
pip install gensim
pip install nltk
pip install sklearn

Also, you must download the dataset from kaggle:
https://www.kaggle.com/stackoverflow/pythonquestions?select=Questions.csv
(for our project you only need the Questions.csv file)
Insert this file into the workspace and you are set!

## Data Collection and Cleaning
We took a dataset about analytics of questions and comments posted on Stack Overflow. The table below is the original dataset which only excludes the 'ID' columns, the unique ID to idenfity each unique post, because we will not be needing it in our data analysis. The dataset is incredibly large and consists of hundreds of thousands of rows which represent unique posts on stack overflow.
We produced a histogram showing frequency of number of likes that questions posted on stack overflow recieve. Vast majority of posts get a score between -100 and 100. The dataset however, is so large and the range of scores is so vast that a histogram of the raw data does not give us a good visual of what is going on.
We found that the dataset had a heavy bias towards posts with positive upvotes as opposed to posts with negative upvotes. Positive upvotes meaning posts with more upvotes than downvotes and negative upvotes meaning more downvotes than upvotes. 
The dataset refers to the integer of upvotes/downvotes as 'scores' so we will use that language.

## Visualizations 
Before constructing and training our model, we visualized our data to get a sense of what distribution of data we are working with. Further, our visualizations  helped us identify which features we used in our model, what associations they had with their scores, corrupt data, and outliers, hence indicating further data cleaning. Through running Visualizations.py, you can see we have multiple different data plots, which helped us gain a well rounded understanding of our data and most importantly, told us how their scores are related to the various features defined previously. 

## Machine Learning
We used a Logistic Regression model to analyze our final data set given its ease of use. After all of our testing, we found a moderately high rate of success (62%), which means that any stackoverflow question put through our model will have a better than random chance to classify a question correctly.

## Conclusion
Afterthought: Increasingly, Stackoverflow has been on of the most toxic QnA coding sites on the internet. Whereas sites such as quora is lenient towards newbie questions, Stackoverflow fails to educate its users on how to ask "quality questions" and continues allowing verbal abuse towards its users and marginalization of minority groups. This not only creates a online culture of discriminiation but also can discourage new coders and coders from marginalized groups from entering the industry, continuing a cycle of systemic racism. Just as a good answer can help coders get unstuck, a harsh comment can make them give up completely.

Imrovements we could have made in the project:

Obviously,our presentation is not perfect. There are multiple ways in which we could have improved our project. Notably, we could have added another attribute like a grammar evaluation score for each question. Something else we could consider is to check how generic the questions. If questions have been repeatedly asked before it would tend to have a lower sccore and is more likely to be trolled. Incorporating these attributes would improve our modest 62% accuracy to something higher!
