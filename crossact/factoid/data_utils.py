import os
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from time import time

logging.basicConfig(level=logging.INFO)


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

        logging.info('Loading object tag data from {} takes {} secs'.format(path_file, time() - t0))
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

        logging.info('Loading user object data from {} takes {} secs'.format(path_file, time() - t0))
        return user_object_dict

    @staticmethod
    def load_label(path_file):
        t0 = time()
        label_dict = {}

        with open(path_file, 'r') as f:
            # TODO: Implement the logic once the format is finalised
            pass

        logging.info('Loading label data from {} takes {} secs'.format(path_file, time() - t0))
        return label_dict


class DataTransformer:
    @staticmethod
    def compute_user_tag(user_obj_dict, obj_tag_dict):
        user_tag_dict = defaultdict(list)

        for uid, objs in user_obj_dict.items():
            for obj in objs:
                user_tag_dict[uid].extend(obj_tag_dict.get(obj, []))

        return user_tag_dict

    @staticmethod
    def merge_user_tags(user_tag_dicts):
        merged_dict = defaultdict(list)
        all_user_ids, all_user_tags = [], []

        for user_tag_dict in user_tag_dicts:
            for user_id, user_tags in user_tag_dict:
                merged_dict[user_id].extend(user_tags)

        for user_id, user_tags in merged_dict.items():
            all_user_ids.append(user_id)
            all_user_tags.append(user_tags)

        return all_user_ids, all_user_tags

    @staticmethod
    def compute_user_features(user_tag_dicts, vectorizer=TfidfVectorizer()):
        t0 = time()
        merged_dict = defaultdict(list)
        all_user_ids, all_user_tags = [], []

        for user_tag_dict in user_tag_dicts:
            for user_id, user_tags in user_tag_dict.items():
                merged_dict[user_id] += user_tags

        for user_id, user_tags in merged_dict.items():
            all_user_ids.append(user_id)
            all_user_tags.append(' '.join(user_tags))

        user_features = vectorizer.fit_transform(all_user_tags)
        logging.info('Compute user features takes {} secs'.format(time() - t0))
        return all_user_ids, user_features


if __name__ == '__main__':
    base_path = '../../data/SO_GH'
    question_path_file = os.path.join(base_path, 'question_tag.csv')
    user_question_path_file = os.path.join(base_path, 'user_question.csv')
    repository_path_file = os.path.join(base_path, 'repository_tag.csv')
    user_repository_path_file = os.path.join(base_path, 'user_repository.csv')

    so_loader = DataLoader('so')
    user_question = so_loader.load_user_object(user_question_path_file)
    question_tag = so_loader.load_object_tag(question_path_file)
    so_user_tag = DataTransformer.compute_user_tag(user_question, question_tag)

    gh_loader = DataLoader('gh')
    user_repo = gh_loader.load_user_object(user_repository_path_file)
    repo_tag = gh_loader.load_object_tag(repository_path_file)
    gh_user_tag = DataTransformer.compute_user_tag(user_repo, repo_tag)

    user_ids, user_features = DataTransformer.compute_user_features([so_user_tag, gh_user_tag])

    logging.info('#users: {}, #feature dimension: {}'.format(len(user_ids), user_features.shape))
    # print(len(question_tag), len(repo_tag), len(user_question))
