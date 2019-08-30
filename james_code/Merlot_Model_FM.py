import os
import csv
import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
import scipy.sparse
import warnings
from scipy.sparse import SparseEfficiencyWarning
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import coverage_error
from sklearn.preprocessing import normalize
from sklearn.metrics import average_precision_score
import math
import random
import sys
import pickle

csv.field_size_limit(sys.maxsize)


def loadUserFeatures(i_file, n_users, n_items, n_activities, user_observed_items):
    print('Loading User Feature...')
    # Sparse Matrix of size U x (U + (IA))
    row = n_users
    column = n_users + (n_items * n_activities)
    # column = n_items * n_activities
    A = scipy.sparse.lil_matrix((row, column), dtype=int)
    with open(i_file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            src = int(row[0])
            des = int(row[1])
            count = int(row[2])
            # make sure it is symmetry
            A[src, des] = count
            A[des, src] = count

    # NN Feature: user activities
    for uid, items in user_observed_items.items():
        for item in items:
            A[uid, n_users + item] = 1
    # A[uid,item] = 1

    return normalize(A)


def loadUserFeatures_CO(i_file, n_users):
    print('Loading User Feature Co Participant...')
    # Sparse Matrix of size U x (U + (IA))
    row = n_users
    column = n_users
    # column = n_items * n_activities
    A = scipy.sparse.lil_matrix((row, column), dtype=int)
    with open(i_file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            src = int(row[0])
            des = int(row[1])
            count = int(row[2])
            # make sure it is symmetry
            A[src, des] = count
            A[des, src] = count
    return normalize(A)


def loadItemFeatures(i_file, i_neighbors, n_items, n_tags, n_activities):
    print('Loading item graph...')
    # Sparse Matrix of size (IA) x (N_i + A)
    row = n_items * n_activities
    columns = n_tags + n_activities + (n_items * n_activities)
    # columns = n_activities + (n_items * n_activities)
    A = scipy.sparse.lil_matrix((row, columns), dtype=int)
    with open(i_file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            rid = int(row[0])
            tag = int(row[1])
            A[rid, tag] = 1
            A[rid + n_items, tag] = 1

    # Activity Indicator
    for rid in range(n_items * n_activities):
        if rid < n_items:
            A[rid, n_tags] = 1
        # A[rid,0] = 1
        else:
            A[rid, n_tags + 1] = 1
    # A[rid,1] = 1

    # NN Feature: BOW k-NN
    with open(i_neighbors, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            src = int(row[0])
            des = int(row[1])
            sim = float(row[2])
            A[src, n_tags + n_activities + des] = sim
            A[src + n_items, n_tags + n_activities + n_items + des] = sim
    # A[src,n_activities+des] = sim
    # A[src+n_items,n_activities+n_items+des] = sim
    return normalize(A)


def loadInteractions(i_file, n_users, n_items, n_activities):
    print('Loading interaction...')
    # Sparse Matrix of size U x (IA)
    row = n_users
    columns = n_items * n_activities
    user_observed_items = {}
    for uid in range(n_users):
        user_observed_items[uid] = []

    A = scipy.sparse.lil_matrix((row, columns), dtype=int)
    with open(i_file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        for row in reader:
            uid = int(row[0])
            rid = int(row[1])
            act1 = int(row[2])
            act2 = int(row[3])
            if act1 == 0:
                A[uid, rid] = -1
            if act2 == 0:
                A[uid, rid + n_items] = -1

    with open(i_file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        for row in reader:
            uid = int(row[0])
            rid = int(row[1])
            act1 = int(row[2])
            act2 = int(row[3])
            if act1 == 1:
                A[uid, rid] = 1
                user_observed_items[uid].append(rid)
            if act2 == 1:
                A[uid, rid + n_items] = 1
                user_observed_items[uid].append(rid + n_items)

    return A, user_observed_items


def loadTest(i_file, n_items, n_activities):
    print('Loading test set...')
    query_users = {}
    with open(i_file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        for row in reader:
            uid = int(row[0])
            query_users[uid] = []

    with open(i_file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        for row in reader:
            uid = int(row[0])
            rid = int(row[1])
            act1 = int(row[2])
            act2 = int(row[3])
            if act1 == 1:
                query_users[uid].append(rid)
            if act2 == 1:
                query_users[uid].append(rid + n_items)

    print(len(query_users))
    result = {}
    for uid, items in query_users.items():
        if len(query_users[uid]) >= 5:
            result[uid] = items

    return result


# return query_users

def loadUserTags(i_file):
    print('Loading user tags...')
    user_tags = {}
    with open(i_file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            uid = int(row[0])
            tags = str(row[1]).split(';')
            user_tags[uid] = list(map(int, tags))
    return user_tags


def loadTagItems(i_file):
    print('Loading tag items...')
    tag_items = {}
    with open(i_file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            tid = int(row[0])
            items = str(row[1]).split(';')
            tag_items[tid] = list(map(int, items))
    return tag_items


def generateSingleTestRecordSampleUnobservedJaccard(uid, pos_items, observed_items, user_tags, tag_items, n_items,
                                                    n_activities):
    threshold = int(n_items * 0.01)

    sample_items = []
    tags = user_tags[uid]
    for tag in tags:
        sample_items.extend(tag_items[tag])

    # Compute the Jaccard Index (numerator only) and sort by this count. We retrieve top % items sorted by Jaccard Index
    item_counts = {}
    for i in sample_items:
        if i not in item_counts:
            item_counts[i] = 1
        else:
            item_counts[i] += 1

    if threshold > len(item_counts) - 1:
        threshold = len(item_counts) - 1

    data = {'rid': list(item_counts.keys()), 'count': list(item_counts.values())}
    dfCounts = pd.DataFrame.from_dict(data)
    sorted_dfCounts = dfCounts.sort_values(by=['count'], ascending=False)
    jaccard_sample_items = []
    for i in range(threshold):
        jaccard_sample_items.append(int(sorted_dfCounts.iloc[i]['rid']))

    unobserved_items = []
    for item in jaccard_sample_items:
        unobserved_items.append(item)
        unobserved_items.append(item + n_items)

    unobserved_items = set(unobserved_items)
    # remove the postive items first
    unobserved_items = unobserved_items.difference(set(pos_items))
    # remove the observed items
    unobserved_items = unobserved_items.difference(set(observed_items))
    items = list(pos_items).copy()
    items.extend(unobserved_items)

    users = [uid] * len(items)
    labels = np.zeros((len(items),), dtype=int)

    for i in range(len(pos_items)):
        labels[i] = 1

    print(len(items))
    return np.array(users), np.array(items), labels


def generateSingleTestRecordSampleUnobserved(uid, pos_items, observed_items, user_tags, tag_items, n_items,
                                             n_activities):
    threshold = int(n_items * 0.01)

    sample_items = []
    tags = user_tags[uid]
    for tag in tags:
        sample_items.extend(tag_items[tag])
    sample_items = list(set(sample_items))

    pick_items = []
    if threshold > len(sample_items):
        pick_items = sample_items.copy()
    else:
        pick_items = random.choices(sample_items, k=threshold)

    unobserved_items = []
    for item in pick_items:
        unobserved_items.append(item)
        unobserved_items.append(item + n_items)

    unobserved_items = set(unobserved_items)
    # remove the postive items first
    unobserved_items = unobserved_items.difference(set(pos_items))
    # remove the observed items
    unobserved_items = unobserved_items.difference(set(observed_items))
    items = list(pos_items).copy()
    items.extend(unobserved_items)

    users = [uid] * len(items)
    labels = np.zeros((len(items),), dtype=int)

    for i in range(len(pos_items)):
        labels[i] = 1
    return np.array(users), np.array(items), labels


def generateSingleTestRecordAllUnobserved(uid, pos_items, n_items, n_activities):
    users = [uid] * (n_items * n_activities)
    items = list(range(n_items * n_activities))
    labels = np.zeros((n_items * n_activities,), dtype=int)

    for rid in pos_items:
        labels[rid] = 1

    return np.array(users), np.array(items), labels


def evaluateAveragePrecisionAtK(y_true, y_score, K):
    records = []
    for i in range(len(y_true)):
        records.append({'y_true': y_true[i], 'y_pred': y_score[i]})

    df = pd.DataFrame(records)
    sorted_df = df.sort_values(by=['y_pred'], ascending=False)

    y_true_k = []
    y_pred_k = []
    results = []

    maxK = max(K)
    if max(K) > len(y_true) - 1:
        maxK = len(y_true) - 1

    for i in range(maxK + 1):

        y_true_k.append(int(sorted_df.iloc[i]['y_true']))
        y_pred_k.append(float(sorted_df.iloc[i]['y_pred']))
        if i in K:
            if sum(y_true_k) == 0:
                results.append(0)
            else:
                results.append(average_precision_score(np.array(y_true_k), np.array(y_pred_k)))

    for i in range(len(results)):
        if math.isnan(results[i]):
            results[i] = 0
    return results


def evaluateAveragePrecisionAtK_ver2(y_true, y_keen, y_act, K):
    records = []
    for i in range(len(y_true)):
        records.append({'y_true': y_true[i], 'y_keen': y_keen[i], 'y_act': y_act[i]})

    df = pd.DataFrame(records)
    sorted_df = df.sort_values(by=['y_keen', 'y_act'], ascending=False)

    y_true_k = []
    y_pred_k = []
    results = []

    maxK = max(K)
    if max(K) > len(y_true) - 1:
        maxK = len(y_true) - 1

    for i in range(maxK + 1):

        y_true_k.append(int(sorted_df.iloc[i]['y_true']))
        y_pred_k.append(float(sorted_df.iloc[i]['y_act']))
        if i in K:
            if sum(y_true_k) == 0:
                results.append(0)
            else:
                results.append(average_precision_score(np.array(y_true_k), np.array(y_pred_k)))

    for i in range(len(results)):
        if math.isnan(results[i]):
            results[i] = 0
    return results


def recommendActivity(i_train, i_test, i_user_graph, i_item_graph, i_item_nearest_neighbours, i_user_tags, i_tag_items,
                      n_users, n_items, n_tags, n_activities, start, end):
    print(i_test)

    L_RATE = 0.05
    L_LOSS = 'bpr'
    ALPHA = 1E-07
    NUM_COMPONENTS = 100
    NUM_THREADS = 20
    EPOCH = 100
    MAX_SAMPLE = int(n_items * n_activities * 0.001)

    interactions, user_observed_items = loadInteractions(i_train, n_users, n_items, n_activities)

    u_features = loadUserFeatures(i_user_graph, n_users, n_items, n_activities, user_observed_items)

    i_features = loadItemFeatures(i_item_graph, i_item_nearest_neighbours, n_items, n_tags, n_activities)

    tag_items = loadTagItems(i_tag_items)

    user_tags = loadUserTags(i_user_tags)

    model = LightFM(no_components=NUM_COMPONENTS,
                    learning_rate=L_RATE,
                    loss=L_LOSS,
                    item_alpha=ALPHA,
                    user_alpha=ALPHA,
                    max_sampled=MAX_SAMPLE)

    print('---------------------------------------')
    print('L_RATE: User_ALL, Item_ALL')
    print('L_RATE:' + str(L_RATE))
    print('L_LOSS:' + str(L_LOSS))
    print('ALPHA:' + str(ALPHA))
    print('NUM_COMPONENTS:' + str(NUM_COMPONENTS))
    print('EPOCH:' + str(EPOCH))
    print('MAX_SAMPLE:' + str(MAX_SAMPLE))
    print('---------------------------------------')

    # # Fitting model
    # model.fit(interactions, user_features=u_features, item_features=i_features, epochs=EPOCH, verbose=True,
    #           num_threads=NUM_THREADS)
    # # model.fit(interactions, item_features=i_features, epochs=EPOCH, verbose=True, num_threads=NUM_THREADS)
    # # model.fit(interactions, user_features=u_features, epochs=EPOCH, verbose=True, num_threads=NUM_THREADS)
    #
    # # Saving model
    # pickle.dump(model, open(
    #     'so/so_model_User_ALL_Item_ALL_' + str(L_LOSS) + '_' + str(L_RATE) + '_' + str(NUM_COMPONENTS) + '_' + str(
    #         EPOCH) + '_' + str(ALPHA) + '.sav', 'wb'))
    # # pickle.dump(model, open('so/so_model_'+str(L_LOSS)+'_'+str(NUM_COMPONENTS)+'_'+str(EPOCH)+'.sav', 'wb'))
    # # model = pickle.load(open('so/so_model_User_ALL_Item_ALL_'+str(L_LOSS)+'_'+str(L_RATE)+'_'+str(NUM_COMPONENTS)+'_'+str(EPOCH)+'_'+str(ALPHA)+'.sav', 'rb'))
    # # model.item_biases *= 0.0
    # # model.user_biases *= 0.0
    #
    query_users = loadTest(i_test, n_items, n_activities)

    # Prediction and Compute LRAP
    sum_AP = 0
    sum_CE = 0
    K = [5, 10, 20, 50, 100, 200]
    sum_APK = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    for uid, pos_items in query_users.items():
        if uid >= start and uid < end:
            test_users, test_items, labels = generateSingleTestRecordSampleUnobserved(uid, pos_items,
                                                                                      user_observed_items[uid],
                                                                                      user_tags, tag_items, n_items,
                                                                                      n_activities)

            print('Predicting for user ' + str(uid))
            # print(pos_items)
            y_scores = np.array(
                model.predict(test_users, test_items, item_features=i_features, user_features=u_features,
                              num_threads=NUM_THREADS))
            # y_scores = np.array(model.predict(test_users,test_items,item_features=i_features,num_threads=NUM_THREADS))
            # y_scores = np.array(model.predict(test_users,test_items,user_features=u_features,num_threads=NUM_THREADS))
            sum_AP += label_ranking_average_precision_score(np.array([labels]), np.array([y_scores]))
            sum_CE += coverage_error(np.array([labels]), np.array([y_scores]))
            results = evaluateAveragePrecisionAtK(labels, y_scores, K)
            print(results)
            for i in range(len(results)):
                sum_APK[i] += results[i]

    print('---------------------------------------')
    print('L_RATE:User_ALL, Item_ALL')
    print('L_RATE:' + str(L_RATE))
    print('L_LOSS:' + str(L_LOSS))
    print('ALPHA:' + str(ALPHA))
    print('NUM_COMPONENTS:' + str(NUM_COMPONENTS))
    print('EPOCH:' + str(EPOCH))
    print('MAX_SAMPLE:' + str(MAX_SAMPLE))
    print('---------------------------------------')

    print('Results for ' + i_test)
    print('Start uid ' + str(start) + ' ' + 'End uid ' + str(end))
    print('No. of Query Users: ' + str(len(query_users)))
    print('LARP: ' + str(sum_AP))
    print('Coverage Error: ' + str(sum_CE))
    for i in range(len(sum_APK)):
        print('AP@' + str(K[i]) + ': ' + str(sum_APK[i]))


if __name__ == "__main__":
    recommendActivity('so/train.csv',
                      'so/test.csv',
                      'so/so_user_user_graph.csv',
                      'so/so_item_graph.csv',
                      'so/so_item_neighbors.csv',
                      'so/so_user_tags.csv',
                      'so/so_tag_items.csv',
                      23612, 1020809, 30354, 2,
                      0, 24000)

# recommendActivity('gh/train.csv',
#					'gh/test.csv',
#					'gh/gh_user_user_graph.csv',
#					'gh/gh_item_graph.csv',
#					33453,461931,12150,2)

# recommendSOFavorites('so_neg/train_favorite_new.csv',
#					'so_neg/test_favorite_new.csv',
#					'so/so_user_user_graph.csv',
#					'so/so_item_graph.csv',
#					23612,1020809,30354)

# recommendGHFork('gh/train_fork_new.csv',
#					'gh/test_fork_new.csv',
#					'gh/gh_user_user_graph.csv',
#					'gh/gh_item_graph.csv',
#					33453,461931,12150)

# recommendGHWatch('gh/train_watch_new.csv',
#					'gh/test_watch_new.csv',
#					'gh/gh_user_user_graph.csv',
#					'gh/gh_item_graph.csv',
#					33453,461931,12150)
