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


def loadUserFeatures(i_file, n_users):
    print('Loading user graph...')
    # Sparse Matrix of size U x N_u
    row = n_users
    column = n_users
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


def loadItemFeatures(i_file, n_items, n_tags, n_activities):
    print('Loading item graph...')
    # Sparse Matrix of size (IA) x (N_i + A)
    row = n_items * n_activities
    columns = n_tags + n_activities
    A = scipy.sparse.lil_matrix((row, columns), dtype=int)
    with open(i_file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            rid = int(row[0])
            tag = int(row[1])
            A[rid, tag] = 1
            A[rid + 1020809, tag] = 1
    for rid in range(n_items * n_activities):
        if rid < n_items:
            A[rid, n_tags] = 1
        else:
            A[rid, n_tags + 1] = 1
    return A


def loadInteractions(i_file, n_users, n_items, n_activities):
    print('Loading interaction...')
    # Sparse Matrix of size U x (IA)
    row = n_users
    columns = n_items * n_activities
    A = scipy.sparse.lil_matrix((row, columns), dtype=int)
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
            if act2 == 1:
                A[uid, rid + n_items] = 1
    return A


def loadTest(i_file, n_users, n_items, n_activities):
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

    return query_users


def generateSingleTestRecord(uid, pos_items, n_items, n_activities):
    users = [uid] * (n_items * n_activities)
    items = range(n_items * n_activities)
    labels = np.zeros((n_items * n_activities,), dtype=int)
    for rid in pos_items:
        labels[rid] = 1
    return np.array(users), np.array(items), labels


'''
def filterPostiveOnly(test_users, labels, pred):
	pos_users = []

	for i in range(len(test_users)):
		if labels[i] == 1:
			pos_users.append(test_users[i])

	pos_users = set(pos_users)

	sub_users = {}
	for uid in pos_users:
		sub_users[uid] = [[],[]]
	
	for i in range(len(test_users)):
		uid = test_users[i]
		if uid in pos_users:
			sub_users[uid][0].append(labels[i])
			sub_users[uid][1].append(pred[i])

	return sub_users

def evaluateLRAP(user_results):
	sum_lrap = 0
	for uid, results in user_results.items():
		y_true = np.array([results[0]])
		y_score = np.array([results[1]])
		sum_lrap += label_ranking_average_precision_score(y_true, y_score)

	return 	sum_lrap/len(user_results)

def evaluateRankingLoss(user_results):
	sum_rl = 0
	for uid, results in user_results.items():
		y_true = np.array([results[0]])
		y_score = np.array([results[1]])
		sum_rl += label_ranking_loss(y_true, y_score)

	return 	sum_rl/len(user_results)

def evaluatePrecisionAtK(user_results, k):
	records = []
	for uid, results in user_results.items():
		d_results = {}
		for i in range(len(results[0])):
			records.append({'uid':uid, 'y_true':results[0][i], 'y_pred':results[1][i]})
	
	df = pd.DataFrame(records)
	total_users = 0
	total_precision = 0
	for uid, results in user_results.items():
		user_df = df[df.uid == uid]
		sorted_user_df = user_df.sort_values(by=['y_pred'], ascending=False)
		sum_pos=0
		if k > sorted_user_df.shape[0]:
			k = sorted_user_df.shape[0]
		for i in range(k):
			sum_pos += sorted_user_df.iloc[i]['y_true']
		total_users += 1
		total_precision += sum_pos/k
	return total_precision/total_users 

def recommendSOAnswers(i_train, i_test, i_user_graph, i_item_graph, n_users, n_items, n_tags):

	interactions = loadInteractions(i_train,n_users,n_items)

	u_features = loadUserFeatures(i_user_graph,n_users)

	i_features = loadItemFeatures(i_item_graph,n_items,n_tags)

	test_users, test_items, labels = loadTest(i_test)

	model = LightFM(learning_rate=0.05, loss='bpr')

	model.fit(interactions, user_features=u_features, item_features=i_features, epochs=1, verbose=True, num_threads=10)
	
	result = model.predict(test_users,test_items,item_features=i_features,user_features=u_features,num_threads=10)
	
	user_results = filterPostiveOnly(test_users, labels, result)

	LRAP = evaluateLRAP(user_results)

	#RankLoss = evaluateRankingLoss(user_results)

	prec1 = evaluatePrecisionAtK(user_results, 1)
	prec5 = evaluatePrecisionAtK(user_results, 5)
	prec10 = evaluatePrecisionAtK(user_results, 10)

	print('Answer Activity LRAP:' + str(LRAP))
	#print('Answer Activity RankLoss:' + str(RankLoss))
	print('Answer Activity Prec@1:'+ str(prec1))
	print('Answer Activity Prec@5:'+ str(prec5))
	print('Answer Activity Prec@10:'+ str(prec10))

def recommendSOFavorites(i_train, i_test, i_user_graph, i_item_graph, n_users, n_items, n_tags):

	interactions = loadInteractions(i_train,n_users,n_items)

	u_features = loadUserFeatures(i_user_graph,n_users)

	i_features = loadItemFeatures(i_item_graph,n_items,n_tags)

	test_users, test_items, labels = loadTest(i_test)

	model = LightFM(learning_rate=0.05, loss='logistic')

	model.fit(interactions, user_features=u_features, item_features=i_features, epochs=50, verbose=True, num_threads=10)
	
	result = model.predict(test_users,test_items,item_features=i_features,user_features=u_features,num_threads=10)
	
	user_results = filterPostiveOnly(test_users, labels, result)

	LRAP = evaluateLRAP(user_results)

	RankLoss = evaluateRankingLoss(user_results)

	prec1 = evaluatePrecisionAtK(user_results, 1)
	prec5 = evaluatePrecisionAtK(user_results, 5)
	prec10 = evaluatePrecisionAtK(user_results, 10)

	print('Favorite Activity LRAP:' + str(LRAP))
	print('Favorite Activity RankLoss:' + str(RankLoss))
	print('Favorite Activity Prec@1:'+ str(prec1))
	print('Favorite Activity Prec@5:'+ str(prec5))
	print('Favorite Activity Prec@10:'+ str(prec10))
	
def recommendGHFork(i_train, i_test, i_user_graph, i_item_graph, n_users, n_items, n_tags):

	interactions = loadInteractions(i_train,n_users,n_items)

	u_features = loadUserFeatures(i_user_graph,n_users)

	i_features = loadItemFeatures(i_item_graph,n_items,n_tags)

	test_users, test_items, labels = loadTest(i_test)

	model = LightFM(learning_rate=0.05, loss='bpr')

	model.fit(interactions, user_features=u_features, item_features=i_features, epochs=50, verbose=True, num_threads=10)
	
	result = model.predict(test_users,test_items,item_features=i_features,user_features=u_features,num_threads=10)
	
	user_results = filterPostiveOnly(test_users, labels, result)

	LRAP = evaluateLRAP(user_results)

	RankLoss = evaluateRankingLoss(user_results)

	prec1 = evaluatePrecisionAtK(user_results, 1)
	prec5 = evaluatePrecisionAtK(user_results, 5)
	prec10 = evaluatePrecisionAtK(user_results, 10)

	print('Fork Activity LRAP:' + str(LRAP))
	print('Fork Activity RankLoss:' + str(RankLoss))
	print('Fork Activity Prec@1:'+ str(prec1))
	print('Fork Activity Prec@5:'+ str(prec5))
	print('Fork Activity Prec@10:'+ str(prec10))
	
def recommendGHWatch(i_train, i_test, i_user_graph, i_item_graph, n_users, n_items, n_tags):

	interactions = loadInteractions(i_train,n_users,n_items)

	u_features = loadUserFeatures(i_user_graph,n_users)

	i_features = loadItemFeatures(i_item_graph,n_items,n_tags)

	test_users, test_items, labels = loadTest(i_test)

	model = LightFM(learning_rate=0.05, loss='bpr')

	model.fit(interactions, user_features=u_features, item_features=i_features, epochs=50, verbose=True, num_threads=10)
	
	result = model.predict(test_users,test_items,item_features=i_features,user_features=u_features,num_threads=10)
	
	user_results = filterPostiveOnly(test_users, labels, result)

	LRAP = evaluateLRAP(user_results)

	RankLoss = evaluateRankingLoss(user_results)

	prec1 = evaluatePrecisionAtK(user_results, 1)
	prec5 = evaluatePrecisionAtK(user_results, 5)
	prec10 = evaluatePrecisionAtK(user_results, 10)

	print('Watch Activity LRAP:' + str(LRAP))
	print('Watch Activity RankLoss:' + str(RankLoss))
	print('Watch Activity Prec@1:'+ str(prec1))
	print('Watch Activity Prec@5:'+ str(prec5))
	print('Watch Activity Prec@10:'+ str(prec10))

'''


def recommendActivity(i_train, i_test, i_user_graph, i_item_graph, n_users, n_items, n_tags, n_activities):
    interactions = loadInteractions(i_train, n_users, n_items, n_activities)

    u_features = loadUserFeatures(i_user_graph, n_users)

    i_features = loadItemFeatures(i_item_graph, n_items, n_tags, n_activities)

    model = LightFM(learning_rate=0.05, loss='bpr')

    model.fit(interactions, user_features=u_features, item_features=i_features, epochs=50, verbose=True, num_threads=10)

    query_users = loadTest(i_test, n_users, n_items, n_activities)

    # Prediction and Compute LRAP
    sum_AP = 0
    sum_CE = 0
    for uid, pos_items in query_users.items():
        test_users, test_items, labels = generateSingleTestRecord(uid, pos_items, n_items, n_activities)
        print('Predicting for user ' + str(uid))
        y_scores = np.array(
            model.predict(test_users, test_items, item_features=i_features, user_features=u_features, num_threads=10))
        sum_AP += average_precision_score(labels, y_scores)
        sum_CE += coverage_error(np.array([labels]), np.array([y_scores]))

    print('Results for ' + i_test)
    print('No. of Query Users: ' + len(query_users))
    print('LARP: ' + str(sum_AP / len(query_users)))
    print('Coverage Error: ' + str(sum_CE / len(query_users)))


if __name__ == "__main__":
    # recommendActivity('so/train.csv',
    # 					'so/test.csv',
    # 					'so/so_user_user_graph.csv',
    # 					'so/so_item_graph.csv',
    # 					23612,1020809,30354,2)

    recommendActivity('../data/full_data_v5/gh/train.csv',
                      '../data/full_data_v5/gh/test.csv',
                      '../data/full_data_v5/gh/gh_user_user_graph.csv',
                      '../data/full_data_v5/gh/gh_item_graph.csv', 33453, 461931, 12150, 2)

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
