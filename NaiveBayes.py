import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def createDataFrame(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'Email text': message, 'Class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)


data = DataFrame({'Email text': [], 'Class': []})

# Load from datasets
data = data.append(createDataFrame('emails/spam', 'spam'))
data = data.append(createDataFrame('emails/ham', 'ham'))

# Model
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['Email text'].values)

classifier = MultinomialNB()
targets = data['Class'].values
classifier.fit(counts, targets)
classifier.score(counts, targets)

# Evaluation
x = data["Email text"]
y = data["Class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state = 1)

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(x_train.values)

classifier = MultinomialNB() 
targets = y_train.values

classifier.fit(counts, targets)
prediction = classifier.predict(counts)
classifier.score(counts, targets)