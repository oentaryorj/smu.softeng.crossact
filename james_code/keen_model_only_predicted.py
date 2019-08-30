from Merlot_Model_FM import loadTest, generateSingleTestRecordSampleUnobserved, evaluateAveragePrecisionAtK
from load_data_ver2 import load_data
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import coverage_error


def modified_items_index(data, n_items):
    users, items, labels = data
    new_users, new_items, new_labels = list(), list(), list()
    for u, i, l in zip(users, items, labels):
        if i >= n_items:
            i = i - n_items
        new_users.append(u)
        new_items.append(i)
        new_labels.append(l)
    return np.array(new_users), np.array(new_items), np.array(new_labels)


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
    # i_keen_interactions_model = './model/so/keen_interactions_model.pickle'
    # keen_interactions_model = load_data(path=i_keen_interactions_model)
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

    i_keen_interactions_model = './model/gh/keen_interactions_model.pickle'
    keen_interactions_model = load_data(path=i_keen_interactions_model)

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
            test_users, test_items, test_labels = modified_items_index(data, n_items)
            y_scores = np.array(
                keen_interactions_model.predict(test_users, test_items,
                                                user_features=user_features_CO,
                                                item_features=i_features,
                                                num_threads=20))
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
