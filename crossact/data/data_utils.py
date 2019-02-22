import os
import logging
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from time import time


class DataLoader:
    """
    Utility class for loading various types of data from CSV files
    """

    def __init__(self, id_prefix='', header_file=None):
        """
        Construct a DataLoader instance

        Keyword args:
            id_prefix (str): Prefix for user and object ID (currently 'so' or 'gh')
        """
        self.id_prefix = id_prefix

    def load_object_tag(self, path_file):
        """
        Load object tag data from file

        Args:
            path_file (str): Data location path

        Returns:
            pd.DataFrame: Object tag dataframe
        """
        if '.csv' not in path_file:
            raise FileNotFoundError('Only CSV format is supported currently')

        t0 = time()
        df = pd.read_csv(path_file, sep=',', header=None)

        if df.shape[1] != 2:
            raise RuntimeError('Object tag data should only consist of object ID and its tags (separated by ;)')

        df.columns = ['object_id', 'object_tags']
        df['object_id'] = df['object_id'].map(lambda x: '{}_{}'.format(self.id_prefix, x))
        df['object_tags'] = df['object_tags'].map(lambda tags: [t.strip() for t in tags.split(';')])

        logging.info('Loading object tag data with {} rows from {} takes {} secs'.format(df.shape[0],
                                                                                         path_file, time() - t0))
        return df

    def load_user_object(self, path_file):
        """
        Load user object data from file

        Args:
            path_file (str): Data location path

        Returns:
            pd.DataFrame: User object dataframe
        """
        if '.csv' not in path_file:
            raise FileNotFoundError('Only CSV format is supported currently')

        t0 = time()
        df = pd.read_csv(path_file, sep=',', header=None)

        if df.shape[1] != 2:
            raise RuntimeError('User object data should only consist of user ID and object ID')

        df.columns = ['user_id', 'object_id']
        df['user_id'] = df['user_id'].map(lambda x: '{}_{}'.format(self.id_prefix, x))
        df['object_id'] = df['object_id'].map(lambda x: '{}_{}'.format(self.id_prefix, x))

        logging.info('Loading user object data with {} rows from {} takes {} secs'.format(df.shape[0],
                                                                                          path_file, time() - t0))
        return df

    @staticmethod
    def load_label(path_file):
        """
        Load label data from file

        Args:
            path_file (str): Data location path

        Returns:
            pd.DataFrame: Label dataframe
        """
        if '.csv' not in path_file:
            raise FileNotFoundError('Only CSV format is supported currently')

        t0 = time()
        df = pd.DataFrame()

        with open(path_file, 'r') as f:
            # TODO: Implement the logic once the format is finalised
            pass

        logging.info('Loading label data with {} rows from {} takes {} secs'.format(df.shape[0],
                                                                                    path_file, time() - t0))
        return df


class DataProcessor:
    @staticmethod
    def aggregate_user_tags(user_obj_df, obj_tag_df):
        """
        Aggregate user tags from user-object and object-tag data

        Args:
            user_obj_df (pd.DataFrame): User object dataframe
            obj_tag_df (pd.DataFrame): Object tag dataframe

        Returns:
            pd.DataFrame: Computed user tag dictionary
        """
        t0 = time()
        user_obj_dict = dict(zip(user_obj_df['user_id'], user_obj_df['object_id']))
        obj_tag_dict = dict(zip(obj_tag_df['object_id'], obj_tag_df['object_tags']))
        user_tag_dict = defaultdict(list)

        for user_id, object_id in user_obj_dict.items():
            user_tag_dict[user_id].extend(obj_tag_dict.get(object_id, []))

        df = pd.DataFrame([{'user_id': id, 'user_tags': tags} for id, tags in user_tag_dict.items()])

        logging.info('Aggregating tags for {} users takes {} secs'.format(df.shape[0], time() - t0))
        return df

    @staticmethod
    def compute_user_tag_features(user_tag_df, vectorizer=TfidfVectorizer()):
        """
        Computes user features by merging user tags from multiple platforms

        Args:
            user_tag_df (pd.DataFrame): List of user tag dictionaries

        Keyword args:
            vectorizer: Feature vectorizer (Default: TfidfVectorizer)

        Returns:
            List: List of user ids
            sp.csr_matrix: SciPy sparse matrix representation of features with shape of (n_users, n_features)
        """
        t0 = time()
        user_ids = user_tag_df['user_id'].tolist()
        user_tag_features = vectorizer.fit_transform(user_tag_df['user_tags'].map(lambda x: ' '.join(x)))

        logging.info('Computing user features with shape takes {} secs'.format(user_tag_features.shape, time() - t0))
        return user_ids, user_tag_features


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(root_path, 'data', 'SO_GH')

    so_question_tag_file = os.path.join(data_path, 'question_tag.csv.gz')
    so_user_question_file = os.path.join(data_path, 'user_question.csv.gz')
    gh_repo_tag_file = os.path.join(data_path, 'repository_tag.csv.gz')
    gh_user_repo_file = os.path.join(data_path, 'user_repository.csv.gz')

    so_loader = DataLoader(id_prefix='so')
    so_user_question_df = so_loader.load_user_object(so_user_question_file)
    so_question_tag_df = so_loader.load_object_tag(so_question_tag_file)
    so_user_tag_df = DataProcessor.aggregate_user_tags(so_user_question_df, so_question_tag_df)
    so_user_ids, so_user_tag_features = DataProcessor.compute_user_tag_features(so_user_tag_df)

    gh_loader = DataLoader(id_prefix='gh')
    gh_user_repo_df = gh_loader.load_user_object(gh_user_repo_file)
    gh_repo_tag_df = gh_loader.load_object_tag(gh_repo_tag_file)
    gh_user_tag_df = DataProcessor.aggregate_user_tags(gh_user_repo_df, gh_repo_tag_df)
    gh_user_ids, gh_user_tag_features = DataProcessor.compute_user_tag_features(gh_user_tag_df)

    logging.info('StackOverflow has {} users and each user has {} features'.format(len(so_user_ids),
                                                                                   so_user_tag_features.shape[1]))
    logging.info('GitHub has {} users and each user has {} features'.format(len(gh_user_ids),
                                                                            gh_user_tag_features.shape[1]))
