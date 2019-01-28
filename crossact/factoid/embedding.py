class ObjectEmbedding:
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

        :param O:
        :return:
        '''
        # TO DO: Implement logic here
        return self

    def transform(self, objects):
        '''
        Compute the embedding vector for object X

        :param O:
        :return:
        '''
        # TO DO: Implement logic here
        return object_embeddings


class FactoidEmbedding:
    '''
    Class for Factoid Embedding
    '''

    def __init__(self):
        '''
        Constructor for factoid embedding
        '''

    def fit(self, object_embeddings, factoids):
        '''
        Learn the factoid embedding vectors based object embeddings and generated factoids

        :param object_embeddings:
        :param factoids:
        :return:
        '''
        # TO DO: Implement logic here
        return self

    def transform(selfself, factoids):
        '''
        Compute the factoid embedding vectors

        :param factoids:
        :return:
        '''
        # TO DO: Implement logic here
        return factoid_embeddings
