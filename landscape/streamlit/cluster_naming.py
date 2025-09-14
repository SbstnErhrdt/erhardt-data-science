import numpy as np
import scipy.sparse as sp
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
from sklearn.utils import check_array
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from sklearn.feature_extraction.text import CountVectorizer
from cluster_naming_pluralize import pluralize


class CTFIDFVectorizer(TfidfTransformer):
    def __init__(self, *args, **kwargs):
        super(CTFIDFVectorizer, self).__init__(*args, **kwargs)
        self._idf_diag = None

    def fit(self, x: sp.csr_matrix, n_samples: int):
        """Learn the idf vector (global term weights)
        Parameters
        ----------
        x : sparse matrix of shape n_samples, n_features)
            A matrix of term/token counts.
        n_samples : int
        """

        # Prepare input
        x = check_array(x, accept_sparse=('csr', 'csc'))
        if not sp.issparse(x):
            x = sp.csr_matrix(x)
        dtype = x.dtype if x.dtype in FLOAT_DTYPES else np.float64

        # Calculate IDF scores
        _, n_features = x.shape
        df = np.squeeze(np.asarray(x.sum(axis=0)))
        avg_nr_samples = int(x.sum(axis=1).mean())
        idf = np.log(avg_nr_samples / df)
        self._idf_diag = sp.diags(idf, offsets=0,
                                  shape=(n_features, n_features),
                                  format='csr',
                                  dtype=dtype)
        return self

    def transform(self, x: sp.csr_matrix, copy=True) -> sp.csr_matrix:
        """Transform a count-based matrix to c-TF-IDF
        Parameters
        ----------
        x : sparse matrix of (n_samples, n_features)
            a matrix of term/token counts
        Returns
        -------
        vectors : sparse matrix of shape (n_samples, n_features)
        :param copy:
        """

        # Prepare input
        x = check_array(x, accept_sparse='csr', dtype=FLOAT_DTYPES, copy=copy)
        if not sp.issparse(x):
            x = sp.csr_matrix(x, dtype=np.float64)

        n_samples, n_features = x.shape

        # idf_ being a property, the automatic attributes detection
        # does not work as usual and we need to specify the attribute
        # name:
        check_is_fitted(self, attributes=["idf_"],
                        msg='idf vector is not fitted')

        # Check if expected nr features is found
        expected_n_features = self._idf_diag.shape[0]
        if n_features != expected_n_features:
            raise ValueError("Input has n_features=%d while the model"
                             " has been trained with n_features=%d" % (
                                 n_features, expected_n_features))

        x = x * self._idf_diag

        if self.norm:
            x = normalize(x, axis=1, norm='l1', copy=False)

        return x


def generate_cluster_names(df,
                           cluster_id_column="cluster_id",
                           cluster_name_column="cluster_name",
                           content_columns=["text"],
                           stop_word_language="english") -> pd.DataFrame:
    """
    Generate cluster names based on the top words in the cluster
    """
    print("generate_cluster_names: ", cluster_id_column, content_columns, stop_word_language)

    # stop words
    stops = set(stopwords.words(stop_word_language))

    df[cluster_id_column] = df[cluster_id_column].astype(int)

    cluster_id_2_text = {}

    # iterate over rows
    for word_index, row in df.iterrows():
        # print(row[cluster_id_column], row[content_columns])
        cluster_id_str = row[cluster_id_column]
        cluster_id = int(cluster_id_str)
        if cluster_id == -1:
            continue
        # get all text of the columns in the row
        text = " ".join(row[content_columns])

        if cluster_id not in cluster_id_2_text:
            cluster_id_2_text[cluster_id] = ""

        # tokenize text
        tokens = nltk.word_tokenize(text)
        # remove stop words
        tokens = [token for token in tokens if token not in stops]
        # remove punctuation
        tokens = [token for token in tokens if token.isalpha()]
        # remove single characters
        tokens = [token for token in tokens if len(token) > 1]
        # remove numbers
        tokens = [token for token in tokens if not token.isnumeric()]
        # pluralize tokens
        tokens = [pluralize(token) for token in tokens]
        # join tokens
        text = " ".join(tokens)

        cluster_id_2_text[cluster_id] = cluster_id_2_text[cluster_id] + " " + text

    # get all ids
    cluster_ids = list(cluster_id_2_text.keys())
    # sort ids
    cluster_ids.sort()

    # create dataframe
    docs_per_class = pd.DataFrame(
        {"Class": cluster_ids, "Document": [cluster_id_2_text[cluster_id] for cluster_id in cluster_ids]})

    # Create bag of words
    count_vectorizer = CountVectorizer(
        # tokenizer=textblob_tokenizer, # this did not work well
        # tokenizer=wordnet_tokenizer,
        stop_words=stop_word_language,
        max_df=0.80,  # ignore terms that appear in 95% of the documents
        min_df=1,  # ignore terms that appear in less than 2 documents
    ).fit(docs_per_class.Document)
    count = count_vectorizer.transform(docs_per_class.Document)
    words = count_vectorizer.get_feature_names_out()

    # Extract top 10 words
    c_tf_idf = CTFIDFVectorizer().fit_transform(count, n_samples=len(df)).toarray()

    cluster_names = {}
    for cluster_id in cluster_ids:
        if cluster_id == -1:
            cluster_names[-1] = "no cluster"
            continue

        cluster_names[cluster_id] = ""
        j = 0
        sorted_indices = np.argsort(c_tf_idf[cluster_id])[::-1]
        for i, word_index in enumerate(sorted_indices):
            # check if greater than 0
            if c_tf_idf[cluster_id][word_index] > 0 and j < 5:
                cluster_names[cluster_id] = cluster_names[cluster_id] + words[word_index] + " "
                j += 1
        cluster_names[cluster_id] = cluster_names[cluster_id].strip(" ")

    df[cluster_name_column] = df[cluster_id_column].map(cluster_names)
    # replace nan with no cluster
    df[cluster_name_column] = df[cluster_name_column].fillna("no cluster")

    return df
