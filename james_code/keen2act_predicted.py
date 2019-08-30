from Merlot_Model_FM import loadTest, generateSingleTestRecordSampleUnobserved, evaluateAveragePrecisionAtK, \
    evaluateAveragePrecisionAtK_ver2
from Merlot_Model_FM import loadInteractions
from load_data_ver2 import load_data
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import coverage_error


def distinct_items(items, n_items):
    new_items = list()
    for item in items:
        if item >= n_items:
            new_items.append(item - n_items)
        else:
            new_items.append(item)
    new_items = list(sorted(list(set(new_items))))
    return new_items


def keen_model_predicted_items(keen_interactions_model, keen_items_threshold, users_features, items_features,
                               data, n_items):
    # return a list of items used for a user
    users, items, labels = data
    users, items = list(set(list(users))), distinct_items(items=items, n_items=n_items)
    users = users * len(items)
    users, items = np.array(users), np.array(items)
    users_items_pred_rank = keen_interactions_model.predict(user_ids=np.array(users), item_ids=np.array(items),
                                                            user_features=users_features,
                                                            item_features=items_features,
                                                            num_threads=20)
    users_items_pred_rank = list(users_items_pred_rank)
    items_dict, cnt = dict(), 0
    for item in items:
        items_dict[item] = users_items_pred_rank[cnt]
        cnt += 1
    selected_items = list()
    for item in items_dict.keys():
        if item in keen_items_threshold.keys():
            ranking, threshold = items_dict[item], keen_items_threshold[item]
            if ranking >= threshold:
                selected_items.append(item)
    return selected_items, (users, items, list(users_items_pred_rank))


def act_model_ranking_predicted(act_model, activities, users_features, items_features, data, selected_items, n_items):
    users, items, labels = data
    users, items, labels = list(users), list(items), list(labels)
    if activities == 1:
        indexes = [items.index(item) for item in selected_items if item in items]
        users = [users[index] for index in indexes]
        items = [items[index] for index in indexes]
        labels = [labels[index] for index in indexes]
    elif activities == 2:
        indexes = [items.index(item + n_items) for item in selected_items if (item + n_items) in items]
        users = [users[index] for index in indexes]
        items = [items[index] - n_items for index in indexes]
        labels = [labels[index] for index in indexes]
    else:
        print('Input correct activities number')
        exit()
    users_items_pred_rank = act_model.predict(user_ids=np.array(users), item_ids=np.array(items),
                                              user_features=users_features, item_features=items_features,
                                              num_threads=20)
    return users_items_pred_rank, np.array(labels), (users, items, list(users_items_pred_rank), labels)


def keen_and_act_scores(act_data, keen_data):
    act_users, act_items, act_scores, act_labels = act_data
    keen_users, keen_items, keen_scores = keen_data

    act_dict_scores = dict()
    act_dict_labels = dict()
    keen_dict_scores = dict()

    for u, i, s in zip(act_users, act_items, act_scores):
        act_dict_scores[str(u) + '-' + str(i)] = s

    for u, i, s in zip(act_users, act_items, act_labels):
        act_dict_labels[str(u) + '-' + str(i)] = s

    for u, i, s in zip(keen_users, keen_items, keen_scores):
        keen_dict_scores[str(u) + '-' + str(i)] = s

    y_keen, y_act, y_label = list(), list(), list()
    for key in act_dict_scores.keys():
        if key in keen_dict_scores.keys():
            y_keen.append(keen_dict_scores[key])
            y_act.append(act_dict_scores[key])
            y_label.append(act_dict_labels[key])
    return np.array(y_keen), np.array(y_act), np.array(y_label)


def keen2Act_evaluation(data, selected_items, users_items_keen_model, users_features, items_features,
                        n_items,
                        act_model_first_activities,
                        act_model_second_activities):
    first_y_pred, first_y_label, first_data = act_model_ranking_predicted(act_model_first_activities, activities=1,
                                                                          users_features=users_features,
                                                                          items_features=items_features,
                                                                          data=data, selected_items=selected_items,
                                                                          n_items=n_items)
    second_y_pred, second_y_label, second_data = act_model_ranking_predicted(act_model_second_activities, activities=2,
                                                                             users_features=users_features,
                                                                             items_features=items_features,
                                                                             data=data, selected_items=selected_items,
                                                                             n_items=n_items)
    first_keen_score, first_act_score, first_label = keen_and_act_scores(act_data=first_data,
                                                                         keen_data=users_items_keen_model)
    second_keen_score, second_act_score, second_label = keen_and_act_scores(act_data=second_data,
                                                                            keen_data=users_items_keen_model)
    y_keen = np.concatenate((first_keen_score, second_keen_score), axis=0)
    y_act = np.concatenate((first_act_score, second_act_score), axis=0)
    y_label = np.concatenate((first_label, second_label), axis=0)
    return y_keen, y_act, y_label


if __name__ == '__main__':
    # i_test = '../data/full_data_v7/so/test.csv'
    # i_user_tags = './data_ver1/so/user_tags.pickle'
    # i_tag_items = './data_ver1/so/tag_items.pickle'
    # n_users, n_items, n_activities = 23612, 1020809, 2
    # query_users = loadTest(i_test, n_items, n_activities)
    # start, end = 0, 24000
    #
    # i_interaction = './data_ver1/so/interactions_user_observed_items.pickle'
    # interactions, user_observed_items = load_data(i_interaction)
    # user_tags = load_data(path=i_user_tags)
    # tag_items = load_data(path=i_tag_items)
    # i_keen_interactions_model, i_keen_items_threshold = './model/so/keen_interactions_model.pickle', \
    #                                                     './model/so/keen_items_threshold_epoch_15.pickle'
    # i_user_features_CO = './data_ver1/so/u_features_CO.pickle'
    # i_act_model_first_activities = './model/so/act_interactions_model_first_activities.pickle'
    # i_act_model_second_activities = './model/so/act_interactions_model_second_activities.pickle'
    # keen_interactions_model = load_data(path=i_keen_interactions_model)
    # keen_items_threshold = load_data(path=i_keen_items_threshold)
    # user_features_CO = load_data(path=i_user_features_CO)
    # act_model_first_activities = load_data(path=i_act_model_first_activities)
    # act_model_second_activities = load_data(path=i_act_model_second_activities)
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
    interactions, user_observed_items = load_data(i_interaction)
    user_tags = load_data(path=i_user_tags)
    tag_items = load_data(path=i_tag_items)
    i_keen_interactions_model, i_keen_items_threshold = './model/gh/keen_interactions_model.pickle', \
                                                        './model/gh/keen_items_threshold_epoch_15.pickle'
    i_user_features_CO = './data_ver1/gh/u_features_CO.pickle'
    i_act_model_first_activities = './model/gh/act_interactions_model_first_activities.pickle'
    i_act_model_second_activities = './model/gh/act_interactions_model_second_activities.pickle'
    keen_interactions_model = load_data(path=i_keen_interactions_model)
    keen_items_threshold = load_data(path=i_keen_items_threshold)
    user_features_CO = load_data(path=i_user_features_CO)
    act_model_first_activities = load_data(path=i_act_model_first_activities)
    act_model_second_activities = load_data(path=i_act_model_second_activities)

    path_i_features = './data_ver1/gh/i_features_non_activities.pickle'
    i_features = load_data(path=path_i_features)

    # Prediction and Compute LRAP
    sum_AP = 0
    sum_CE = 0
    K = [5, 10, 20, 50, 100, 200]
    sum_APK = [0, 0, 0, 0, 0, 0]

    for uid, pos_items in query_users.items():
        if uid >= start and uid < end:
            test_users, test_items, labels = generateSingleTestRecordSampleUnobserved(uid, pos_items,
                                                                                      user_observed_items[uid],
                                                                                      user_tags, tag_items, n_items,
                                                                                      n_activities)
            data = (test_users, test_items, labels)
            selected_items, users_items_keen_model = keen_model_predicted_items(
                keen_interactions_model=keen_interactions_model,
                keen_items_threshold=keen_items_threshold,
                users_features=user_features_CO,
                items_features=i_features,
                data=data, n_items=n_items)
            y_keen, y_act, y_label = keen2Act_evaluation(data=data, selected_items=selected_items,
                                                         users_items_keen_model=users_items_keen_model,
                                                         users_features=user_features_CO,
                                                         items_features=i_features,
                                                         n_items=n_items,
                                                         act_model_first_activities=act_model_first_activities,
                                                         act_model_second_activities=act_model_second_activities)
            sum_AP += label_ranking_average_precision_score(np.array([y_label]), np.array([y_act]))
            # results = evaluateAveragePrecisionAtK(y_label, y_pred, K)
            y_keen, y_act, y_label = list(y_keen), list(y_act), list(y_label)
            results = evaluateAveragePrecisionAtK_ver2(y_true=y_label, y_keen=y_keen, y_act=y_act, K=K)
            print(uid, results)
            for i in range(len(results)):
                sum_APK[i] += results[i]
    print('Results for ' + i_test)
    print('Start uid ' + str(start) + ' ' + 'End uid ' + str(end))
    print('No. of Query Users: ' + str(len(query_users)))
    print('LARP: ' + str(sum_AP))
    for i in range(len(sum_APK)):
        print('AP@' + str(K[i]) + ': ' + str(sum_APK[i]))
