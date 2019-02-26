import jellyfish
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def jaro_winkler_similarity(word_matrix):
    '''
    Compute the Jaro-Winkler similarities among all word pairs

    :param word_matrix: Word matrix
    :return: float: Computed similarity score
    '''
    similarity_matrix = np.zeros(word_matrix.shape)

    for i in similarity_matrix.shape[0]:
        for j in similarity_matrix.shape[j]:
            similarity_matrix[i, j] = 1.0 - jellyfish.jaro_winkler(w1, w2)

    return similarity_matrix


def cosine_similarity(word_matrix):
    '''
    Compute the cosine similarities among all word pairs

    :param word_matrix: Word matrix
    :return: float: Computed similarity score
    '''
    vectorizer = CountVectorizer(word_matrix)
    vectorizer.fit(word_matrix)
    vectors = [t for t in vectorizer.transform(text).toarray()]
    return cosine_similarity(vectors)
