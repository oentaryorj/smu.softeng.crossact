from abc import ABC, abstractmethod


class Embedding:
    '''
    Abstract class for embedding
    '''

    @abstractmethod
    def fit(self, objects):
        '''
        Abstract method for training/fitting the embedding vector

        :param objects:
        :return:
        '''
        pass

    @abstractmethod
    def transform(self, objects):
        '''
        Abstract method for computing the embedding vector

        :param objects:
        :return:
        '''
        pass


class ObjectEmbedding(Embedding):
    '''
    Class for Object Embedding
    '''

    def __init__(self, similarity_func):
        '''
        Constructor for object embedding

        :param similarity_func:
        '''
        super().__init__()
        self.similarity_func = similarity_func

    def fit(self, objects):
        '''
        Learn the embedding vectors of objects X based on a specified similarity function

        :param objects:
        :return:
        '''
        # TO DO: Implement logic here
        return self

    def transform(self, objects):
        '''
        Transform objects into embedding vectors

        :param objects:
        :return:
        '''
        # TO DO: Implement logic here
        return object_embeddings


class FactoidEmbedding(Embedding):
    '''
    Class for Factoid Embedding
    '''

    def __init__(self):
        '''
        Constructor for factoid embedding
        '''
        super().__init__()

    def fit(self, factoids, object_embeddings):
        '''
        Learn the factoid embedding vectors based object embeddings and generated factoids

        :param object_embeddings:
        :param factoids:
        :return:
        '''
        # TO DO: Implement logic here
        return self

    def transform(self, factoids):
        '''
        Transform factoids into embedding vectors

        :param factoids:
        :return:
        '''
        # TO DO: Implement logic here
        return factoid_embeddings
