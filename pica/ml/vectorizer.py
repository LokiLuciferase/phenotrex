import array
from collections import defaultdict

import numpy as np
from scipy import sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.fixes import sp_version


class CustomVectorizer(CountVectorizer):
    """
    modified from CountVectorizer to override the _validate_vocabulary function, which invoked an error because
    multiple indices of the dictionary contained the same feature index. However, this is we intend with the
    compress_vocabulary function.
    Other functions had to be adopted: get_feature_names (allow decompression), _count_vocab (reduce the matrix size)
    """

    def _validate_vocabulary(self):
        """
        overriding the validation which does not accept multiple feature-names to encode for one feature
        """
        if self.vocabulary:
            self.vocabulary_ = dict(self.vocabulary)
            self.fixed_vocabulary_ = True
        else:
            self.fixed_vocabulary_ = False

    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name"""
        if not hasattr(self, 'vocabulary_'):
            self._validate_vocabulary()

        self._check_vocabulary()

        # return value is different from normal CountVectorizer output: maintain dict instead of returning a list
        return sorted(self.vocabulary_.items(), key=lambda x: x[1])

    def _count_vocab(self, raw_documents, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False
        Modified to reduce the actual size of the matrix returned if compression of vocabulary is used
        """
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        analyze = self.build_analyzer()
        j_indices = []
        indptr = []

        values = array.array(str("i"))
        indptr.append(0)
        for doc in raw_documents:
            feature_counter = {}
            for feature in analyze(doc):
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

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulary; perhaps the documents only"
                                 " contain stop words")

        if indptr[-1] > 2147483648:  # = 2**31 - 1
            if sp_version >= (0, 14):
                indices_dtype = np.int64
            else:
                raise ValueError(f'sparse CSR array has {indptr[-1]} non-zero '
                                 f'elements and requires 64 bit indexing, '
                                 f' which is unsupported with scipy {".".join(sp_version)}. '
                                 f'Please upgrade to scipy >=0.14')

        else:
            indices_dtype = np.int32

        j_indices = np.asarray(j_indices, dtype=indices_dtype)
        indptr = np.asarray(indptr, dtype=indices_dtype)
        values = np.frombuffer(values, dtype=np.intc)

        # modification here:
        vocab_len = vocabulary[max(vocabulary, key=vocabulary.get)] + 1

        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, vocab_len),
                          dtype=self.dtype)
        X.sort_indices()
        return vocabulary, X
