import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
# convert to core ml with coremltools
import coremltools

# Reading in and parsing data
raw_data = open('data-set/SMSSpamCollection.txt', 'r')
sms_data = []
for line in raw_data:
    split_line = line.split("\t")
    sms_data.append(split_line)

# Splitting the data into messages and labels and training and test sets
sms_data = np.array(sms_data)
X = sms_data[:, 1]
y = sms_data[:, 0]

# Build a svc model
vectorizer = TfidfVectorizer()
vectorized_text = vectorizer.fit_transform(X)
text_clf = LinearSVC()
text_clf = text_clf.fit(vectorized_text, y)

# Cross validation
cross_score = cross_val_score(text_clf, vectorized_text, y, cv=10)

# Convert to CoreML model
coreml_model = coremltools.converters.sklearn.convert(text_clf, "message", "spam_or_not")

# Add descriptions for model parameters
coreml_model.short_description = "Classify if a message is spam"
coreml_model.input_description["message"] = "TFIDF of message"
coreml_model.output_description["spam_or_not"] = "Classification"

# Save the CoreML model
coreml_model.save("SpamClassifier.mlmodel")
