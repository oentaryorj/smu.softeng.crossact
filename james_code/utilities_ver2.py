import csv
import numpy as np
from lightfm import LightFM
import scipy.sparse
from sklearn.metrics import label_ranking_average_precision_score
import math
import random
import pickle


def loadUserFeatures(i_file, n_users):
    print('Loading user graph...')
    A = scipy.sparse.lil_matrix((n_users, n_users), dtype=int)
    with open(i_file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        for row in reader:
            src = int(row[0])
            des = int(row[1])
            count = int(row[2])
            A[src, des] = count
    return A


def loadItemFeatures(i_file, n_items, n_tags):
    print('Loading item graph...')
    A = scipy.sparse.lil_matrix((n_items, n_tags), dtype=int)
    with open(i_file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        for row in reader:
            rid = int(row[0])
            tag = int(row[1])
            A[rid, tag] = 1
    return A


def loadInteractions(i_file, n_users, n_items):
    print('Loading interaction...')
    A = scipy.sparse.lil_matrix((n_users, n_items), dtype=int)
    with open(i_file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        for row in reader:
            uid = int(row[0])
            rid = int(row[1])
            A[uid, rid] = 1
    return A


def loadInteractions_keenfunc_thres(i_file, n_users, n_items):
    pos_users, pos_items, pos_labels = [], [], []
    print('Loading interactions threshold model...')
    with open(i_file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        for row in reader:
            pos_users.append(int(row[0]))
            pos_items.append(int(row[1]))
            pos_labels.append(1)

    total_users = [u for u in range(n_users)]
    total_items = [i for i in range(n_items)]
    neg_users, neg_items, neg_labels = [], [], []
    set_neg_items = list(set(total_items) - set(pos_items))
    for item in set_neg_items:
        users = list(random.sample(total_users, 2))
        for u in users:
            neg_users.append(u)
            neg_items.append(item)
            neg_labels.append(0)
    users = pos_users + neg_users
    items = pos_items + neg_items
    labels = pos_labels + neg_labels
    print(len(users), len(items), len(labels))
    return np.array(users), np.array(items), np.array(labels)


def loadTest(i_file):
    users = []
    items = []
    labels = []
    print('Loading test set...')
    with open(i_file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        for row in reader:
            users.append(int(row[0]))
            items.append(int(row[1]))
            labels.append(int(row[2]))

    return np.array(users), np.array(items), np.array(labels)


def evaluateLRAP(test_users, test_items, labels, pred):
    pos_users = []

    for i in range(len(test_users)):
        if labels[i] == 1:
            pos_users.append(test_users[i])

    pos_users = set(pos_users)

    sub_users = []
    sub_items = []
    sub_labels = []
    sub_pred = []
    for i in range(len(test_users)):
        if test_users[i] in pos_users:
            sub_users.append(test_users[i])
            sub_items.append(test_items[i])
            sub_labels.append(labels[i])
            sub_pred.append(pred[i])

    y_score = np.array([sub_pred])
    y_true = np.array([labels])
    print(len(y_score))
    print(len(y_true))

    score = label_ranking_average_precision_score(y_true, y_score)


def recommendSOAnswers(i_train, i_test, i_user_graph, i_item_graph, n_users, n_items, n_tags):
    interactions = loadInteractions(i_train, n_users, n_items)

    u_features = loadUserFeatures(i_user_graph, n_users)

    i_features = loadItemFeatures(i_item_graph, n_items, n_tags)

    test_users, test_items, labels = loadTest(i_test)

    model = LightFM(learning_rate=0.05, loss='logistic')

    model.fit(interactions, user_features=u_features, item_features=i_features, epochs=5, verbose=True, num_threads=10)

    result = model.predict(test_users, test_items, item_features=i_features, user_features=u_features, num_threads=10)
    y_score = np.array([result])
    y_true = np.array([labels])
    print(result)
    print(len(y_score))
    print(len(y_true))

    score = label_ranking_average_precision_score(y_true, y_score)
    print(score)


def mini_batch(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation]

    # Step 1: No Shuffle (X, Y)
    # shuffled_X = X[:, :]
    # shuffled_Y = Y[:]

    # Step 2: Partition (X, Y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size)))
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batch_no_shuffle_X(X, mini_batch_size=64, seed=0):
    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    shuffled_X = X
    # Step 2: Partition (X, Y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size)))
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batches.append(mini_batch_X)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        mini_batches.append(mini_batch_X)
    return mini_batches


def save_data(path, data):
    print('Saving data...')
    with open(path, 'wb') as fle:
        pickle.dump(data, fle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(path):
    print('Loading data...')
    return pickle.load(open(path, 'rb'))


if __name__ == "__main__":
    path_interaction = '../data/full_data_v5/so/train.csv'
    path_user_features = '../data/full_data_v5/so/so_user_user_graph.csv'
    path_item_features = '../data/full_data_v5/so/so_item_graph.csv'
    n_users, n_items, n_tags = 23612, 1020809, 30354
    path_inter_thres = './data/so_interactions_threshold.pickle'
    path_save_interactions = './data/so_interactions.pickle'
    path_save_users_features = './data/so_user_features.pickle'
    path_save_items_features = './data/so_item_features.pickle'

    # # interactions_thres = loadInteractions_keenfunc_thres(i_file=path_interaction, n_users=n_users, n_items=n_items)
    # # save_data(path=path_inter_thres, data=interactions_thres)
    # interactions_thres = load_data(path=path_inter_thres)
    # print(len(interactions_thres))

    # interactions = loadInteractions(i_file=path_interaction, n_users=n_users, n_items=n_items)
    # save_data(path=path_save_interactions, data=interactions)

    # user_features = loadUserFeatures(i_file=path_user_features, n_users=n_users)
    # save_data(path=path_save_users_features, data=user_features)

    item_features = loadItemFeatures(i_file=path_item_features, n_items=n_items, n_tags=n_tags)
    save_data(path=path_save_items_features, data=item_features)


