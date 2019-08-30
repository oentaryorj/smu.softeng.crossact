from Merlot_Model_FM import loadTest, generateSingleTestRecordSampleUnobserved, evaluateAveragePrecisionAtK
from load_data_ver2 import load_data
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import coverage_error


def modified_items_indexes_for_activities(data, n_items):
    users, items, labels = data
    first_users, first_items, first_labels = list(), list(), list()
    second_users, second_items, second_labels = list(), list(), list()
    for u, i, l in zip(users, items, labels):
        if i < n_items:
            first_users.append(u)
            first_items.append(i)
            first_labels.append(l)
        else:
            second_users.append(u)
            second_items.append(i - n_items)
            second_labels.append(l)
    return (np.array(first_users), np.array(first_items), np.array(first_labels)), (np.array(second_users),
                                                                                    np.array(second_items),
                                                                                    np.array(second_labels))


if __name__ == '__main__':
    # i_test = '../data/full_data_v7/so/test.csv'
    # i_user_tags = './data_ver1/so/user_tags.pickle'
    # i_tag_items = './data_ver1/so/tag_items.pickle'
    # n_users, n_items, n_activities = 23612, 1020809, 2
    # query_users = loadTest(i_test, n_items, n_activities)
    # start, end = 0, 24000
    #
    # i_interaction = './data_ver1/so/interactions_user_observed_items.pickle'
    # _, user_observed_items = load_data(i_interaction)
    # user_tags = load_data(path=i_user_tags)
    # tag_items = load_data(path=i_tag_items)
    #
    # i_act_interactions_model_first_activities = './model/so/act_interactions_model_first_activities.pickle'
    # act_interactions_model_first_activities = load_data(path=i_act_interactions_model_first_activities)
    #
    # i_act_interactions_model_second_activities = './model/so/act_interactions_model_second_activities.pickle'
    # act_interactions_model_second_activities = load_data(path=i_act_interactions_model_second_activities)
    #
    # i_user_features_CO = './data_ver1/so/u_features_CO.pickle'
    # user_features_CO = load_data(path=i_user_features_CO)
    #
    # path_i_features = './data_ver1/so/i_features_non_activities.pickle'
    # i_features = load_data(path=path_i_features)

    i_test = '../data/full_data_v7/gh/test.csv'
    i_user_tags = './data_ver1/gh/user_tags.pickle'
    i_tag_items = './data_ver1/gh/tag_items.pickle'
    n_users, n_items, n_activities = 33453, 461931, 2
    query_users = loadTest(i_test, n_items, n_activities)
    start, end = 0, 35000

    i_interaction = './data_ver1/gh/interactions_user_observed_items.pickle'
    _, user_observed_items = load_data(i_interaction)
    user_tags = load_data(path=i_user_tags)
    tag_items = load_data(path=i_tag_items)

    i_act_interactions_model_first_activities = './model/gh/act_interactions_model_first_activities.pickle'
    act_interactions_model_first_activities = load_data(path=i_act_interactions_model_first_activities)

    i_act_interactions_model_second_activities = './model/gh/act_interactions_model_second_activities.pickle'
    act_interactions_model_second_activities = load_data(path=i_act_interactions_model_second_activities)

    i_user_features_CO = './data_ver1/gh/u_features_CO.pickle'
    user_features_CO = load_data(path=i_user_features_CO)

    path_i_features = './data_ver1/gh/i_features_non_activities.pickle'
    i_features = load_data(path=path_i_features)

    # Prediction and Compute LRAP
    sum_AP = 0
    sum_CE = 0
    K = [5, 10, 20, 50, 100, 200]
    sum_APK = [0, 0, 0, 0, 0, 0]
    print('No. of Query Users: ' + str(len(query_users)))
    for uid, pos_items in query_users.items():
        if uid >= start and uid < end:
            test_users, test_items, labels = generateSingleTestRecordSampleUnobserved(uid, pos_items,
                                                                                      user_observed_items[uid],
                                                                                      user_tags, tag_items, n_items,
                                                                                      n_activities)
            data = (test_users, test_items, labels)
            first_data, second_data = modified_items_indexes_for_activities(data, n_items)
            first_users, first_items, first_labels = first_data
            second_users, second_items, second_labels = second_data
            first_y_scores = np.array(act_interactions_model_first_activities.predict(first_users, first_items,
                                                                                      user_features=user_features_CO,
                                                                                      item_features=i_features,
                                                                                      num_threads=20))
            second_y_scores = np.array(act_interactions_model_first_activities.predict(second_users, second_items,
                                                                                       user_features=user_features_CO,
                                                                                       item_features=i_features,
                                                                                       num_threads=20))
            y_scores = np.concatenate((first_y_scores, second_y_scores), axis=0)
            y_label = np.concatenate((first_labels, second_labels), axis=0)

            sum_AP += label_ranking_average_precision_score(np.array([labels]), np.array([y_scores]))
            sum_CE += coverage_error(np.array([labels]), np.array([y_scores]))
            results = evaluateAveragePrecisionAtK(labels, y_scores, K)
            print(uid, results)
            for i in range(len(results)):
                sum_APK[i] += results[i]

    print('Results for ' + i_test)
    print('Start uid ' + str(start) + ' ' + 'End uid ' + str(end))
    print('No. of Query Users: ' + str(len(query_users)))
    print('LARP: ' + str(sum_AP))
    print('Coverage Error: ' + str(sum_CE))
    for i in range(len(sum_APK)):
        print('AP@' + str(K[i]) + ': ' + str(sum_APK[i]))
