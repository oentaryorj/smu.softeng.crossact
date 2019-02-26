from abc import ABC, abstractmethod


class FactoidFactory(ABC):
    '''
    Abstract class for generating factoids from data
    '''

    @abstractmethod
    def run(self, data):
        '''
        Generate factoids from data

        :param data:
        :return:
        '''
        pass


class GitHubFactoidFactory(FactoidFactory):
    def __init__(self):
        '''
        Constructor for Github factoid factory
        '''
        pass

    def run(self, data):
        '''
        Generate factoids from GitHub data

        :param data:
        :return:
        '''
        return factoids


class StackOverflowFactoidFactory(FactoidFactory):
    def __init__(self):
        '''
        Constructor for StackOverflow factoid factory
        '''
        pass

    def run(self, data):
        '''
        Generate factoids from StackOverflow data

        :param data:
        :return:
        '''
        return factoids
