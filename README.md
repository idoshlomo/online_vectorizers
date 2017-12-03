# Online Vectorizers
Extension of scikit-learn TfidfVectorizer and CountVectorizer that allows for online learning / partial fit.

### The problem
A big challenges in using TF-IDF vector representations in production environment is how to deal with out of core textual data. What usually happens is that the existing vectorization will ignore all OOV (out of vocabulary) tokens. This then causes inaccurate decision making later on in the pipelines. This is a real problem when the process is sensitive to unique tokens (e.g. cosine string similarity on short strings).


The ideal solution is to fit the vectorizer on the out of core data so that it recognizes all tokens (this is usually called partial fit). However Scikit Learn's existing TfidfVectorizers and CountVectorizer don't support partial fit methods, and instead one must fit the vectorizer both on the original data and the new data (this can be totally unfeasible in production environment).

This repo contains an extension of these two classes which supports partial fitting. The idea is that partial fitting allows yields the same representation that a fit on the entire data would yield but **much faster**. How much faster is related to the size of the core text data VS the out of core text data.

###  Some important notes

* The classical "partial fit" in Scikit Learn requires that it play nice with other objects in a pipeline that support partial fitting. This means that the vector representation dimension cannot change after partial fit is called (see discussion in [PR #9014](https://github.com/scikit-learn/scikit-learn/pull/9014/files)).

* This implementation definitely **does** change the dimension, since the whole point is to support new tokens. This is why I called these methods "partial **refit**" as opposed to "partial **fit**" to make clear this distinction.

* If your pipeline specifically doesn't care about the vector dimension (e.g. if you immediately apply a transformation like cosine similarity) than you shouldn't care about this.

### In this repo

* Core code in online_vectorizers.py

* Example usage and script validation equivalency of online to "regular" methods in the examples directory

* Examples use a bit of sample data taken from Hansards "Parliament of Canada" dataset ([here](https://www.isi.edu/natural-language/download/hansard/)).