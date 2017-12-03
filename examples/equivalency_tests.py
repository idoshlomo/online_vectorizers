from sklearn.feature_extraction.text import TfidfVectorizer
from online_vectorizers import OnlineTfidfVectorizer
import time
import os

# load some sample data
with open(os.path.join("data_files", "hansard.36.2.house.debates.073.txt")) as f:
    lines = f.readlines()

raw_text = [x.rstrip("\n").strip() for x in lines]
print("\nfirst 5 entries in text sample:\n", '\n'.join(raw_text[:5]))

short_text = raw_text[:10]

# split into core and out of core samples
core_text = raw_text[:2500]
out_of_core_text = raw_text[2500:]

# train regular tfidf vectorizer on core sample
start = time.clock()
sklearn_tfidf = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
sklearn_tfidf.fit(core_text)
print("\nregular tfidf core texts train time: ", time.clock() - start)

# train online tfidf vectorizer on core sample
start = time.clock()
online_tfidf = OnlineTfidfVectorizer(ngram_range=(1, 2), lowercase=True)
online_tfidf.fit(core_text)
print("online tfidf core texts train time (sec): ", time.clock() - start)

# check equivalency of representation: dictionary of all tokens and corresponding idf values
sklearn_token_to_idf_dict = {k: sklearn_tfidf.idf_[v] for (k, v) in sklearn_tfidf.vocabulary_.items()}
online_token_to_idf_dict = {k: online_tfidf.idf_[v] for (k, v) in online_tfidf.vocabulary_.items()}
print("is representation equivalent: ", sklearn_token_to_idf_dict == online_token_to_idf_dict)

# to incorporate out of core texts regular tifdf vectorizer must be fit on entire text sample
start = time.clock()
sklearn_tfidf = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
sklearn_tfidf.fit(raw_text)
print("\nregular tfidf all texts fit time (sec): ", time.clock() - start)

# to incorporate out of core texts lonline tifdf vectorizer can be partially refit on just out of core sample
start = time.clock()
online_out_of_core_text_result = online_tfidf.partial_refit(out_of_core_text)
print("online tfidf out of core texts update time (sec): ", time.clock() - start)

# check equivalency of representation: dictionary of all tokens and corresponding idf values
# only difference should be the numeric index associated with the out of core tokens. regular tfidf vectorizer will
# number them in ascending lexicographic order, whereas online vectorizer will not.
sklearn_token_to_idf_dict = {k: sklearn_tfidf.idf_[v] for (k, v) in sklearn_tfidf.vocabulary_.items()}
online_token_to_idf_dict = {k: online_tfidf.idf_[v] for (k, v) in online_tfidf.vocabulary_.items()}
print("is representation equivalent: ", sklearn_token_to_idf_dict == online_token_to_idf_dict)
