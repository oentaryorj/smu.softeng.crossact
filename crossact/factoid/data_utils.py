import os
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from time import time


class DataLoader:
    """
    Utility class for loading various types of data from CSV files
    """

    def __init__(self, data_type):
        """
        Construct a DataLoader instance

        Args:
            data_type (str): Data type (currently 'so' or 'gh')
        """
        self.data_type = data_type

    def load_object_tag(self, path_file):
        """
        Load object tag data from file

        Args:
            path_file (str): Data location path

        Returns:
            dict: Object tag dictionary
        """
        t0 = time()
        tag_dict = {}

        with open(path_file, 'r') as f:
            for line in f:
                tokens = line.strip().split(',')
                obj_id = '{}_{}'.format(self.data_type, tokens[0])
                obj_tags = [t.strip() for t in tokens[1].split(';')]
                tag_dict[obj_id] = obj_tags

        logging.info('Loading object tag data with {} rows from {} takes {} secs'.format(len(tag_dict),
                                                                                         path_file, time() - t0))
        return tag_dict

    def load_user_object(self, path_file):
        """
        Load user object data from file

        Args:
            path_file (str): Data location path

        Returns:
            dict: User object dictionary
        """
        t0 = time()
        user_object_dict = defaultdict(list)

        with open(path_file, 'r') as f:
            for line in f:
                tokens = line.strip().split(',')
                user_id = '{}_{}'.format(self.data_type, tokens[0])
                obj_id = '{}_{}'.format(self.data_type, tokens[1])
                user_object_dict[user_id].append(obj_id)

        logging.info('Loading user object data with {} ros from {} takes {} secs'.format(len(user_object_dict),
                                                                                         path_file, time() - t0))
        return user_object_dict

    @staticmethod
    def load_label(path_file):
        """
        Load label data from file

        Args:
            path_file (str): Data location path

        Returns:
            dict: Label dictionary
        """
        t0 = time()
        label_dict = {}

        with open(path_file, 'r') as f:
            # TODO: Implement the logic once the format is finalised
            pass

        logging.info('Loading label data with {} rows from {} takes {} secs'.format(len(label_dict),
                                                                                    path_file, time() - t0))
        return label_dict


class DataProcessor:
    @staticmethod
    def compute_user_tag(user_obj_dict, obj_tag_dict):
        """
        Compute user tags from user object and object tag dictionaries

        Args:
            user_obj_dict (dict): User object dictionary
            obj_tag_dict (dict): Object tag dictionary

        Returns:
            dict: Computed user tag dictionary
        """
        user_tag_dict = defaultdict(list)

        for uid, objs in user_obj_dict.items():
            for obj in objs:
                user_tag_dict[uid].extend(obj_tag_dict.get(obj, []))

        return user_tag_dict

    @staticmethod
    def compute_user_features(user_tag_dicts, vectorizer=TfidfVectorizer()):
        """
        Computes user features by merging user tags from multiple platforms

        Args:
            user_tag_dicts (List): List of user tag dictionaries

        Kwargs:
            vectorizer: Feature vectorizer (Default: TfidfVectorizer)

        Returns:
            user_ids: List of user ids
            user_features: sparse matrix representation of features with shape of (n_users, n_features)
        """
        t0 = time()
        merged_dict = defaultdict(list)
        all_user_ids, all_user_tags = [], []

        for user_tag_dict in user_tag_dicts:
            for user_id, user_tags in user_tag_dict.items():
                merged_dict[user_id].extend(user_tags)

        for user_id, user_tags in merged_dict.items():
            all_user_ids.append(user_id)
            all_user_tags.append(' '.join(user_tags))

        user_features = vectorizer.fit_transform(all_user_tags)
        logging.info('Computing user features with shape takes {} secs'.format(user_features.shape, time() - t0))
        return all_user_ids, user_features


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'data/SO_GH')

    question_path_file = os.path.join(data_path, 'question_tag.csv')
    user_question_path_file = os.path.join(data_path, 'user_question.csv')
    repository_path_file = os.path.join(data_path, 'repository_tag.csv')
    user_repository_path_file = os.path.join(data_path, 'user_repository.csv')

    so_loader = DataLoader('so')
    user_question = so_loader.load_user_object(user_question_path_file)
    question_tag = so_loader.load_object_tag(question_path_file)
    so_user_tag = DataProcessor.compute_user_tag(user_question, question_tag)

    gh_loader = DataLoader('gh')
    user_repo = gh_loader.load_user_object(user_repository_path_file)
    repo_tag = gh_loader.load_object_tag(repository_path_file)
    gh_user_tag = DataProcessor.compute_user_tag(user_repo, repo_tag)

    user_ids, user_features = DataProcessor.compute_user_features([so_user_tag, gh_user_tag])

    logging.info('No. of users: {}, Feature dimension: {}'.format(len(user_ids), user_features.shape))
