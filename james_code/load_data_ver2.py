from Merlot_Model_FM import loadInteractions, loadUserFeatures, loadItemFeatures, loadTagItems, loadUserTags, \
    loadUserFeatures_CO
import pickle
import scipy.sparse
import csv
from sklearn.preprocessing import normalize


def save_data(path, data):
    print('Saving data...')
    with open(path, 'wb') as fle:
        pickle.dump(data, fle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(path):
    print('Loading data...')
    return pickle.load(open(path, 'rb'))


def load_interactions_without_activities(i_train, n_users, n_items):
    print('Loading Interaction without activities...')
    row = n_users
    column = n_items
    A = scipy.sparse.lil_matrix((row, column), dtype=int)
    users, items, labels = list(), list(), list()
    with open(i_train, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        for row in reader:
            src = int(row[0])
            des = int(row[1])
            # make sure it is symmetry
            A[src, des] = 1
            users.append(src)
            items.append(des)
            labels.append(1)
            
    # TODO: Check if normalisation is needed. In my opinion, binary features/matrices need not be normalised at all :)
    return normalize(A), (users, items, labels)


def load_interactions_with_activities_for_ACT(i_train, n_users, n_items, n_activities):
    print('Loading Interaction with activities for ACT model...')
    row = n_users
    column = n_items

    A = scipy.sparse.lil_matrix((row, column), dtype=int)
    print('pass sparse matrix')
    users, items, labels = list(), list(), list()
    with open(i_train, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        for row in reader:
            src = int(row[0])
            des = int(row[1])
            if n_activities == 0:
                act = int(row[2])
            elif n_activities == 1:
                act = int(row[3])
            else:
                print('Wrong activities index')
                exit()
            if act == 1:
                A[src, des] = 1
                users.append(src)
                items.append(des)
                labels.append(1)
                
    # TODO: Check if normalisation is needed. In my opinion, binary features/matrices need not be normalised at all :)
    return normalize(A), (users, items, labels)


def load_item_features_without_activities(i_file, i_neighbors, n_items, n_tags, n_activities):
    print('Loading item features without considering the activities...')
    row = n_items * n_activities
    columns = n_tags
    A = scipy.sparse.lil_matrix((row, columns), dtype=int)
    with open(i_file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            rid = int(row[0])
            tag = int(row[1])
            A[rid, tag] = 1
            
    # TODO: Check if normalisation is needed. In my opinion, binary features/matrices need not be normalised at all :)
    return normalize(A)


if __name__ == '__main__':
    # # load interactions and user observed items
    # i_train = '../data/full_data_v7/so/train.csv'
    # n_users, n_items, n_activities = 23612, 1020809, 2
    # interactions, user_observed_items = loadInteractions(i_train, n_users, n_items, n_activities)
    # interactions_user_observed_items = (interactions, user_observed_items)
    # print(interactions.shape, len(user_observed_items))
    # path_saving = './data_ver1/'
    # save_data(path=path_saving + 'interactions_user_observed_items.pickle', data=interactions_user_observed_items)

    # # # load interactions and user observed items
    # i_train = '../data/full_data_v7/gh/train.csv'
    # n_users, n_items, n_activities = 33453, 461931, 2
    # interactions, user_observed_items = loadInteractions(i_train, n_users, n_items, n_activities)
    # interactions_user_observed_items = (interactions, user_observed_items)
    # print(interactions.shape, len(user_observed_items))
    # path_saving = './data_ver1/gh'
    # save_data(path=path_saving + 'interactions_user_observed_items.pickle', data=interactions_user_observed_items)

    # #####################################################################################################
    # # load interactions and user observed items
    # i_train = '../data/full_data_v7/so/train.csv'
    # n_users, n_items, n_activities = 23612, 1020809, 2
    # interactions, user_observed_items = loadInteractions(i_train, n_users, n_items, n_activities)
    # print(interactions.shape, len(user_observed_items))
    #
    # # load user features
    # i_user_graph = '../data/full_data_v7/so/so_user_user_graph.csv'
    # u_features = loadUserFeatures(i_user_graph, n_users, n_items, n_activities, user_observed_items)
    # print(u_features.shape)
    #
    # # load item features
    # i_item_graph = '../data/full_data_v7/so/so_item_graph.csv'
    # i_item_nearest_neighbours = '../data/full_data_v7/so/so_item_neighbors.csv'
    # n_tags = 30354
    # i_features = loadItemFeatures(i_item_graph, i_item_nearest_neighbours, n_items, n_tags, n_activities)
    # print(i_features.shape)
    #
    # # load tag items
    # i_tag_items = '../data/full_data_v7/so/so_tag_items.csv'
    # tag_items = loadTagItems(i_tag_items)
    # print(len(tag_items))
    #
    # # load user tags
    # i_user_tags = '../data/full_data_v7/so/so_user_tags.csv'
    # user_tags = loadUserTags(i_user_tags)
    # print(len(user_tags))

    # # saving data
    # #####################################################################################################
    # path_saving = './data_ver1/'
    # save_data(path=path_saving + 'interactions.pickle', data=interactions)
    # save_data(path=path_saving + 'user_observed_items.pickle', data=user_observed_items)
    # save_data(path=path_saving + 'u_features.pickle', data=u_features)
    # save_data(path=path_saving + 'i_features.pickle', data=i_features)
    # save_data(path=path_saving + 'tag_items.pickle', data=tag_items)
    # save_data(path=path_saving + 'user_tags.pickle', data=user_tags)
    # exit()

    # # load user features Co-participant
    # i_user_graph = '../data/full_data_v7/so/so_user_user_graph.csv'
    # n_users = 23612
    # u_features_CO = loadUserFeatures_CO(i_user_graph, n_users)
    # print(u_features_CO.shape)
    # path_saving = './data_ver1/'
    # save_data(path=path_saving + 'u_features_CO.pickle', data=u_features_CO)
    # exit()

    # # # #####################################################################################################
    # # # #####################################################################################################
    # # # load user and items without considering any activities
    # # i_train = '../data/full_data_v7/so/train.csv'
    # n_users, n_items = 23612, 1020809
    # # interactions_non_activities = load_interactions_without_activities(i_train=i_train, n_users=n_users,
    # #                                                                    n_items=n_items)
    #
    # # load item features for keen model  (don't care about the activities)
    # i_item_graph = '../data/full_data_v7/so/so_item_graph.csv'
    # i_item_nearest_neighbours = '../data/full_data_v7/so/so_item_neighbors.csv'
    # n_tags, n_activities = 30354, 1
    # i_features_non_activities = load_item_features_without_activities(i_item_graph, i_item_nearest_neighbours, n_items,
    #                                                                   n_tags, n_activities)
    # print(i_features_non_activities.shape)
    # # # #####################################################################################################
    # path_saving = './data_ver1/so/'
    # # save_data(path=path_saving + 'interactions_non_activities.pickle', data=interactions_non_activities)
    # save_data(path=path_saving + 'i_features_non_activities.pickle', data=i_features_non_activities)

    # #####################################################################################################
    # # load user and items without considering any activities -- SMALL TRAIN DATA
    # i_train = '../data/full_data_v7/so/train_small.csv'
    # n_users, n_items = 23612, 1020809
    # interactions_non_activities = load_interactions_without_activities(i_train=i_train, n_users=n_users,
    #                                                                    n_items=n_items)
    # print(interactions_non_activities.shape)
    # path_saving = './data_test/'
    # save_data(path=path_saving + 'interactions_non_activities.pickle', data=interactions_non_activities)
    # exit()

    # # # #####################################################################################################
    # # # #####################################################################################################
    # # # load user and items considering any activities for ACT model
    # i_train = '../data/full_data_v7/so/train.csv'
    # n_users, n_items = 23612, 1020809
    # interactions_with_first_activities_for_ACT = load_interactions_with_activities_for_ACT(i_train, n_users, n_items,
    #                                                                                        n_activities=0)
    # path_saving = './data_ver1/'
    # save_data(path=path_saving + 'interactions_with_first_activities_for_ACT.pickle',
    #           data=interactions_with_first_activities_for_ACT)
    #
    # interactions_with_second_activities_for_ACT = load_interactions_with_activities_for_ACT(i_train, n_users, n_items,
    #                                                                                         n_activities=0)
    # path_saving = './data_ver1/'
    # save_data(path=path_saving + 'interactions_with_second_activities_for_ACT.pickle',
    #           data=interactions_with_second_activities_for_ACT)

    # # # #####################################################################################################
    # # # #####################################################################################################
    # TESTING PHASE
    # i_train = '../data/full_data_v7/so/test.csv'
    # n_users, n_items = 23612, 1020809
    # interactions_non_activities_testing = load_interactions_without_activities(i_train=i_train, n_users=n_users,
    #                                                                            n_items=n_items)
    # # return two elements, iteractions and (users, items, labels)
    # print(len(interactions_non_activities_testing))
    # path_saving = './data_ver1/'
    # save_data(path=path_saving + 'interactions_non_activities_testing.pickle', data=interactions_non_activities_testing)

    # i_train = '../data/full_data_v7/so/test.csv'
    # n_users, n_items = 23612, 1020809
    # interactions_with_first_activities_for_ACT = load_interactions_with_activities_for_ACT(i_train, n_users, n_items,
    #                                                                                        n_activities=0)
    # path_saving = './data_ver1/'
    # save_data(path=path_saving + 'interactions_with_first_activities_for_ACT_testing.pickle',
    #           data=interactions_with_first_activities_for_ACT)
    #
    # interactions_with_second_activities_for_ACT = load_interactions_with_activities_for_ACT(i_train, n_users, n_items,
    #                                                                                         n_activities=0)
    # path_saving = './data_ver1/'
    # save_data(path=path_saving + 'interactions_with_second_activities_for_ACT_testing.pickle',
    #           data=interactions_with_second_activities_for_ACT)

    # # # #####################################################################################################
    # # # #####################################################################################################
    # GH
    # # # #####################################################################################################
    # # # #####################################################################################################
    # # # load user features Co-participant
    # i_user_graph = '../data/full_data_v7/gh/gh_user_user_graph.csv'
    # n_users = 33453
    # u_features_CO = loadUserFeatures_CO(i_user_graph, n_users)
    # print(u_features_CO.shape)
    # path_saving = './data_ver1/gh/'
    # save_data(path=path_saving + 'u_features_CO.pickle', data=u_features_CO)
    # # # #####################################################################################################

    # # # #####################################################################################################
    # # # # load user and items without considering any activities
    # i_train = '../data/full_data_v7/gh/train.csv'
    # n_users, n_items = 33453, 461931
    # interactions_non_activities = load_interactions_without_activities(i_train=i_train, n_users=n_users,
    #                                                                    n_items=n_items)
    # path_saving = './data_ver1/gh/'
    # save_data(path=path_saving + 'interactions_non_activities.pickle', data=interactions_non_activities)
    # # # #####################################################################################################

    # # # #####################################################################################################
    # load item features for keen model  (don't care about the activities)
    # i_item_graph = '../data/full_data_v7/gh/gh_item_graph.csv'
    # i_item_nearest_neighbours = '../data/full_data_v7/gh/gh_item_neighbors.csv'
    # n_users, n_items = 33453, 461931
    # n_tags, n_activities = 12150, 1
    # i_features_non_activities = load_item_features_without_activities(i_item_graph, i_item_nearest_neighbours, n_items,
    #                                                                   n_tags, n_activities)
    # print(i_features_non_activities.shape)
    # path_saving = './data_ver1/gh/'
    # save_data(path=path_saving + 'i_features_non_activities.pickle', data=i_features_non_activities)
    # # # #####################################################################################################

    # # # #####################################################################################################
    # # load user and items considering any activities for ACT model
    # i_train = '../data/full_data_v7/gh/train.csv'
    # n_users, n_items = 33453, 461931
    # interactions_with_first_activities_for_ACT = load_interactions_with_activities_for_ACT(i_train, n_users, n_items,
    #                                                                                        n_activities=0)
    # path_saving = './data_ver1/gh/'
    # save_data(path=path_saving + 'interactions_with_first_activities_for_ACT.pickle',
    #           data=interactions_with_first_activities_for_ACT)
    #
    # interactions_with_second_activities_for_ACT = load_interactions_with_activities_for_ACT(i_train, n_users, n_items,
    #                                                                                         n_activities=0)
    # path_saving = './data_ver1/gh/'
    # save_data(path=path_saving + 'interactions_with_second_activities_for_ACT.pickle',
    #           data=interactions_with_second_activities_for_ACT)
    # # # #####################################################################################################
    # load tag items
    # path_saving = './data_ver1/gh/'
    # i_tag_items = '../data/full_data_v7/gh/gh_tag_items.csv'
    # tag_items = loadTagItems(i_tag_items)
    # print(len(tag_items))
    # save_data(path=path_saving + 'tag_items.pickle', data=tag_items)
    #
    # # # load user tags
    # path_saving = './data_ver1/gh/'
    # i_user_tags = '../data/full_data_v7/gh/gh_user_tags.csv'
    # user_tags = loadUserTags(i_user_tags)
    # print(len(user_tags))
    # save_data(path=path_saving + 'user_tags.pickle', data=user_tags)

    # # # #####################################################################################################
    i_train = '../data/full_data_v7/gh/train.csv'
    n_users, n_items, n_activities = 33453, 461931, 2
    interactions, user_observed_items = loadInteractions(i_train, n_users, n_items, n_activities)
    interactions_user_observed_items = (interactions, user_observed_items)
    print(interactions.shape, len(user_observed_items))
    path_saving = './data_ver1/gh/'
    save_data(path=path_saving + 'interactions_user_observed_items.pickle', data=interactions_user_observed_items)
