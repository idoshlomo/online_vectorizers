"""Simple examples of how to apply the online vectorizers
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from online_vectorizers import OnlineTfidfVectorizer
import logging
logging.basicConfig(level=logging.INFO)

# train TfidfVectorizer on some texts
raw_texts = ['cat and dog', 'a dog', 'a cat']
sklearn_tfidf = TfidfVectorizer()
sklearn_tfidf.fit(raw_texts)

# in real world usage texts with new unrecognized words come along
new_text_1 = 'cat and fox'
new_text_1_vector = sklearn_tfidf.transform([new_text_1])

new_text_2 = 'cat and wolf'
new_text_2_vector = sklearn_tfidf.transform([new_text_2])

# new texts are different but only in out of core text, so vectors are identical
print("\ntfidf vectorizer: out of core texts")
print(new_text_1, ": ", new_text_1_vector.A)
print(new_text_2, ": ", new_text_2_vector.A)

# solution 1: refit vectorizer on old texts and the new texts before using
sklearn_tfidf = TfidfVectorizer()
sklearn_tfidf.fit(raw_texts + [new_text_1] + [new_text_2])

new_text_1_vector = sklearn_tfidf.transform([new_text_1])
new_text_2_vector = sklearn_tfidf.transform([new_text_2])

print("\ntfidf vectorizer: out of core texts + full refit")
print(new_text_1, ": ", new_text_1_vector.A)
print(new_text_2, ": ", new_text_2_vector.A)

# this can be a problem for use in production, since it may take a long time to retrain on large dataset
# its also wasteful since the out of core texts are usually tiny compared to the core training texts
# solution 2: perform a partial fit to only on the out of core texts

online_tfidf = OnlineTfidfVectorizer()  # initial fit on core texts
online_tfidf.fit(raw_texts)

online_tfidf.partial_refit([new_text_1] + [new_text_2])  # partial fit on out of core texts
new_text_1_vector = online_tfidf.transform([new_text_1])
new_text_2_vector = online_tfidf.transform([new_text_2])

print("\nonline tfidf vectorizer: out of core texts + partial refit")
print(new_text_1, ": ", new_text_1_vector.A)
print(new_text_2, ": ", new_text_2_vector.A)
