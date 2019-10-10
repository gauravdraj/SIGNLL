import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv('global_warming_tweets.csv',encoding = "utf-8-sig",engine='python',error_bad_lines=False)

# play around with the dataset to see what data it contains
print(df)
print('Tweet:', df['tweet'][16], '\nExistence:', df['existence'][16])
print()
print('Tweet:', df['tweet'][17], '\nExistence:', df['existence'][17])


train_df, test_df = train_test_split(df, test_size=0.1)

vectorizer = CountVectorizer()      # Bag-of-words vectorizer; counts occurences of each word in each tweet
vectorizer.fit(train_df['tweet'])   # Fitting the vectorizer = create a vocabulary to index mapping

def tweets_to_tensor(tweet_list):
    
    X = vectorizer.transform(tweet_list)

    X = X.todense()

    X = torch.tensor(X, dtype=torch.float)
    return X

X_train = tweets_to_tensor(train_df['tweet'])
Y_train = train_df['existence'].values

print("Shape of X_train:", X_train.shape)

from model import TweetClassifier

classifier = TweetClassifier(X_train.shape[1]) # len(X_train[0])

optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01)

for epoch in range(50):
    print("Epoch:", epoch)

    for i in range(len(X_train)):
        prediction = classifier.forward(X_train[i])

        if prediction == Y_train[i]:
            loss = torch.log(1+torch.exp(-prediction))
        else:
            loss = torch.log(1+torch.exp(prediction))

        loss.backward()         # calculate the gradients
        optimizer.step()        # take step down the gradient
        optimizer.zero_grad()   # reset the gradients to zero


correct = 0
file = open('testpredictions.txt', 'w')
X_test = tweets_to_tensor(test_df['tweet'])
Y_test = test_df['existence'].values

for i in range(len(X_test)):
    prediction = classifier.forward(X_test[i])

    print(test_df['tweet'].values[i], file=file)
    print("Prediction:", float(prediction), "\tActual:", Y_test[i], file=file)
    print("----------------------------", file=file)

    if prediction == Y_test[i]:
        correct += 1

print("Accuracy", correct / len(X_test))
words_to_examine = ["conspiracy", "environment"]

for word in words_to_examine:
    word_index = vectorizer.get_feature_names().index(word)
    print("Word:", word, "\tCorresponding weight:", classifier.weight_vec[word_index])