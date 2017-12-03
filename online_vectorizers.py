"""Extension of scikit-learn TfidfVectorizer and CountVectorizer that allows for online learning
"""
import logging
from collections import defaultdict
from itertools import chain

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, _make_int_array, _document_frequency

logger = logging.getLogger(__name__)


class OnlineCountVectorizer(CountVectorizer):
    """Scikit-learn CountVectorizer with online learning
    """
    def partial_refit(self, raw_documents):
        """Update existing vocabulary dictionary with all oov (out of vocabulary) tokens in the raw documents.

         Parameters
         ----------
         raw_documents : iterable
             An iterable which yields either str, unicode or file objects.

         Returns
         -------
         self: OnlineCountVectorizer
         """
        self.partial_refit_transform(raw_documents)
        return self

    def partial_refit_transform(self, raw_documents):
        """Update the exiting vocabulary dictionary and return term-document matrix.

        This is equivalent to partial_refit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : array, [n_samples, n_features]
            Document-term matrix.
        """
        analyzer = self.build_analyzer()
        analyzed_documents = [analyzer(doc) for doc in raw_documents]
        new_tokens = set(chain.from_iterable(analyzed_documents))
        oov_tokens = new_tokens.difference(set(self.vocabulary_.keys()))

        if oov_tokens:
            logger.info("adding {} tokens".format(len(oov_tokens)))
            max_index = max(self.vocabulary_.values())
            oov_vocabulary = dict(zip(oov_tokens, list(range(max_index + 1, max_index + 1 + len(oov_tokens), 1))))
            self.vocabulary_.update(oov_vocabulary)

        _, X = self._count_analyzed_vocab(analyzed_documents, True)
        return X

    def _count_analyzed_vocab(self, analyzed_documents, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False.

        For consistency this is a slightly edited version of feature_extraction.text._count_vocab with the document
        analysis factored out.
        """
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        j_indices = []
        indptr = _make_int_array()
        values = _make_int_array()
        indptr.append(0)
        for analyzed_doc in analyzed_documents:
            feature_counter = {}
            for feature in analyzed_doc:
                try:
                    feature_idx = vocabulary[feature]
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = 1
                    else:
                        feature_counter[feature_idx] += 1
                except KeyError:
                    # Ignore out-of-vocabulary items for fixed_vocab=True
                    continue

            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))

        j_indices = np.asarray(j_indices, dtype=np.intc)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        values = np.frombuffer(values, dtype=np.intc)

        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(vocabulary)),
                          dtype=self.dtype)
        X.sort_indices()
        return vocabulary, X


class OnlineTfidfVectorizer(OnlineCountVectorizer):
    """Scikit-learn TfidfVectorizer with online learning
    """
    def __init__(self, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, stop_words=None, token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1),
                 analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False,
                 dtype=np.int64, norm='l2', smooth_idf=True, sublinear_tf=False):
        """All of the regular kwargs accepted by the TfidfVectorizer __init__ and their default values except use_idf
        which is hard set to True.
        """
        super().__init__(input, encoding, decode_error, strip_accents, lowercase, preprocessor, tokenizer, stop_words,
                         token_pattern, ngram_range, analyzer, max_df, min_df, max_features, vocabulary, binary, dtype)
        self._tfidf = TfidfTransformer(norm=norm, use_idf=True, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
        self.n_samples = None
        self.n_features = None
        self.df = None

    def fit(self, raw_documents, y=None):
        """Standard TfidfVectorizer fit method plus storing some meta-data needed for partial_refit methods
        (document frequency vector and number of samples).
        """
        X = super().fit_transform(raw_documents)
        self.n_samples, self.n_features = X.shape
        self.df = _document_frequency(X)
        self.n_samples += int(self._tfidf.smooth_idf)
        self._update_idf()
        return self

    def fit_transform(self, raw_documents, y=None):
        """Standard TfidfVectorizer fit method plus storing some meta-data needed for partial_refit methods
        (document frequency vector and number of samples).
        """
        X = super().fit_transform(raw_documents)
        self.n_samples, self.n_features = X.shape
        self.df = _document_frequency(X)
        self.n_samples += int(self._tfidf.smooth_idf)
        self._update_idf()
        return X

    def partial_refit(self, raw_documents):
        """Update the exiting vocabulary dictionary and idf

        This is equivalent to partial_refit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        self: OnlineTfidfVectorizer
        """
        self.partial_refit_transform(raw_documents)
        return self

    def partial_refit_transform(self, raw_documents):
        """Update the exiting vocabulary dictionary and idf and return term-document matrix.

        This is equivalent to partial_refit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : array, [n_samples, n_features]
            Document-term matrix.
        """
        logger.info("validate: checking {} records for new tokens".format(len(raw_documents)))
        current_vocabulary_size = len(self.vocabulary_)
        X = super().partial_refit_transform(raw_documents)
        vocabulary_size_change = len(self.vocabulary_) - current_vocabulary_size
        if vocabulary_size_change > 0:
            df = _document_frequency(X)
            self.n_features += vocabulary_size_change
            self.n_samples += X.shape[0]
            self.df = np.vstack((np.hstack((self.df, np.zeros(vocabulary_size_change))), df))
            self.df = self.df.sum(0)
            self._update_idf()

        return self._tfidf.transform(X)

    def transform(self, raw_documents):
        """Standard TfidfVectorizer transform.
        """
        X = super().transform(raw_documents)
        return self._tfidf.transform(X)

    @property
    def idf_(self):
        return np.ravel(self._tfidf._idf_diag.sum(axis=0))

    def _update_idf(self):
        """Update the idf and _idf_diag attributes given updated values for document-frequency matrix, n_samples and
        n_features.
        """
        self._tfidf.idf = np.log(float(self.n_samples) / (self.df + int(self._tfidf.smooth_idf))) + 1.0
        self._tfidf._idf_diag = sp.spdiags(self._tfidf.idf, diags=0, m=self.n_features, n=self.n_features,
                                           format='csr')
