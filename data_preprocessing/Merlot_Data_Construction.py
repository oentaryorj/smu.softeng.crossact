import os
import csv
import time
import datetime
import urllib
import re
import math
import sys
import io
import random
import pymysql.cursors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import warnings
from sklearn.metrics import label_ranking_average_precision_score
from scipy.sparse import SparseEfficiencyWarning
import networkx as nx
warnings.simplefilter('ignore',SparseEfficiencyWarning)
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels


def generateUserItemGithub(i_user_file, i_user_fork_file, i_user_watch_file, o_user_item_file, o_user_file, min_item):
	print('Loading user data... \n')
	user_item_count={}
	user_fork_count={}
	user_watch_count={}
	user_item={}
	user_fork={}
	user_watch={}
	with open(i_user_file, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			uid = row[0]
			user_item_count[uid] = 0
			user_fork_count[uid]=0
			user_watch_count[uid]=0
			user_item[uid]=[]
			user_fork[uid]=[]
			user_watch[uid]=[]

	print('Loading user-fork-item data... \n')
	with open(i_user_fork_file, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			uid = row[0]
			rid = row[1]
			user_item_count[uid] += 1
			user_fork_count[uid] += 1
			user_item[uid].append(rid)
			user_fork[uid].append(rid)

	print('Loading user-watch-item data... \n')
	with open(i_user_watch_file, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			uid = row[0]
			rid = row[1]
			user_item_count[uid] += 1
			user_watch_count[uid] += 1
			user_item[uid].append(rid)
			user_watch[uid].append(rid)

	print('Generating user_item data... \n')
	user_index = {}
	index=0
	with open(o_user_item_file, 'a+', encoding='utf-8') as fout:
		fout.writelines('uid,rid,fork,watch\n')
		for key,value in user_item_count.items():
			#if user_fork_count[key] >= min_fork and user_watch_count[key]>=min_watch:
			if value >= min_item:
				print('Processing user '+str(index)+'... \n')
				user_index[index] = key
				itemset = set(user_item[key])
				for item in itemset:
					record = str(index) +','+str(item)
					#check if it is forked
					if item in user_fork[key]:
						record = record + ',1'
					else:
						record = record + ',0'
					#check if it is watched
					if item in user_watch[key]:
						record = record + ',1'
					else:
						record = record + ',0'
					fout.writelines(record+'\n')
				index += 1

	#with open(o_user_file, 'a+', encoding='utf-8') as fout:
	#	fout.writelines('index,uid\n')
	#	for key,value in user_index.items():
	#		fout.writelines(str(key)+','+str(value)+'\n')

def generateUserItemStackoverflow(i_user_file, i_user_answer_file, i_user_favorite_file, o_user_item_file, o_user_file, min_item):
	print('Loading user data... \n')
	user_item_count={}
	user_answer_count={}
	user_favorite_count={}
	user_item={}
	user_answer={}
	user_favorite={}
	with open(i_user_file, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			uid = row[0]
			user_item_count[uid] = 0
			user_answer_count[uid]=0
			user_favorite_count[uid]=0
			user_item[uid]=[]
			user_answer[uid]=[]
			user_favorite[uid]=[]

	print('Loading user-answer-item data... \n')
	with open(i_user_answer_file, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			uid = row[0]
			rid = row[1]
			user_item_count[uid] += 1
			user_answer_count[uid] += 1
			user_item[uid].append(rid)
			user_answer[uid].append(rid)

	print('Loading user-favorite-item data... \n')
	with open(i_user_favorite_file, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			uid = row[0]
			rid = row[1]
			user_item_count[uid] += 1
			user_favorite_count[uid] += 1
			user_item[uid].append(rid)
			user_favorite[uid].append(rid)

	print('Generating user_item data... \n')
	user_index = {}
	index=0
	with open(o_user_item_file, 'a+', encoding='utf-8') as fout:
		fout.writelines('uid,rid,answer,favorite\n')
		for key,value in user_item_count.items():
			#if user_answer_count[key] >= min_answer and user_favorite_count[key]>=min_favorite:
			#if value >= min_item:
			print('Processing user '+str(index)+'... \n')
			user_index[index] = key
			itemset = set(user_item[key])
			for item in itemset:
				record = str(index) +','+str(item)
				#check if it is answered
				if item in user_answer[key]:
					record = record + ',1'
				else:
					record = record + ',0'
				#check if it is favoriteed
				if item in user_favorite[key]:
					record = record + ',1'
				else:
					record = record + ',0'
				fout.writelines(record+'\n')
			index += 1

	#with open(o_user_file, 'a+', encoding='utf-8') as fout:
	#	fout.writelines('index,uid\n')
	#	for key,value in user_index.items():
	#		fout.writelines(str(key)+','+str(value)+'\n')

def generateItemTags(i_full_item_file, i_user_item_file, o_item_file):
	print('Loading full item data... \n')
	item_tags={}
	with open(i_full_item_file, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			rid = row[0]
			tags = row[1]
			item_tags[rid] = tags

	print('Generating item data... \n')
	items=[]
	with open(o_item_file, 'a+', encoding='utf-8') as fout:
		fout.writelines('rid,tag\n')
		with open(i_user_item_file, encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			#skip header
			next(reader, None)
			for row in reader:
				rid = row[1]
				items.append(rid)
		items = set(items)

		for item in items:
			fout.writelines(str(item)+','+str(item_tags[item])+'\n')

def genenerateTagIndex(i_items, o_tags):
	l_tags = []
	with open(i_items, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			tags = row[2]
			l_tags.extend(tags.split(' ; '))
	s_tags = set(l_tags)
	with open(o_tags, 'a+', encoding='utf-8') as fout:
		i=0
		for tag in s_tags:
			fout.writelines(str(i)+','+str(tag)+'\n')
			i+=1

def generateItemTagGraph(i_tag,i_item,o_file):
	tag_index = {}
	with open(i_tag, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			tag_index[row[1]] = int(row[0])

	with open(o_file, 'a+', encoding='utf-8') as fout:
		with open(i_item, encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			for row in reader:
				rid = int(row[0])
				tags = row[2]
				for tag in tags.split(' ; '):
					fout.writelines(str(rid)+','+str(tag_index[tag])+'\n')

def generateUserCoParticpationGraphV2(i_file,o_file):
	print('Loading user-item data... \n')
	user_item_df = pd.read_csv(i_file)

	user_item_dict={}
	for index, row in user_item_df.iterrows():
		uid = str(row['uid']).strip()
		rid = str(row['rid']).strip()
		if uid in user_item_dict:
			user_item_dict[uid].append(rid)
		else:
			user_item_dict[uid] = [rid]

	s_user_item_dict={}
	for key, value in user_item_dict.items():
		s_user_item_dict[key] = set(value)

	print('Initializing graph... \n')
	
	outer_list = list(user_item_dict.keys())
	inner_list = list(user_item_dict.keys())
	with open(o_file, 'a+', encoding='utf-8') as fout:
		for s in outer_list:
			print('Processing '+ s)
			for t in inner_list:
				if s != t:
					count = len(s_user_item_dict[s].intersection(s_user_item_dict[t]))
					if  count > 0:
						fout.writelines(str(s) +','+str(t) +','+str(count)+'\n')

def generateUserCoParticpationGraph (i_file,o_file):
	print('Loading user-item data... \n')
	user_item_df = pd.read_csv(i_file)
	
	item_user_dict = {}
	unique_users = []

	for index, row in user_item_df.iterrows():
		uid = str(row['uid']).strip()
		rid = str(row['rid']).strip()

		#update the repository-user dict
		if rid in item_user_dict:
			item_user_dict[rid] = item_user_dict[rid] + ';' + uid
		else:
			item_user_dict[rid] = uid

		#update unique_users list
		unique_users.append(uid)

	unique_users = set(unique_users)
	
	print('User-item data loaded')
	print('#user: ' +str(len(unique_users)))
	print('#items: ' +str(len(item_user_dict)))

	print('Initialize graph...')
	graph_dict = []
	for key, value in item_user_dict.items():
		users = value.split(';')
		for s in users:
			for t in users:	
				if s != t:
					if G.has_edge(str(s),str(t)):
						G[str(s)][str(t)]['count'] += 1
					else:
						G.add_edge(str(s),str(t), count=1)


	#G = nx.Graph()
	#for key, value in item_user_dict.items():
	#	users = value.split(';')
	#	for s in users:
	#		for t in users:	
	#			if s != t:
	#				G.add_node(str(s))
	#				G.add_node(str(t))
	#				if G.has_edge(str(s),str(t)):
	#					G[str(s)][str(t)]['count'] += 1
	#				else:
	#					G.add_edge(str(s),str(t), count=1)
	print('Initilize graph completed')

	print('Convert graph to undirected...')
	H = G.to_undirected()
	print('Conversion completed.')

	print('Saving graph...')
	with open(o_file, 'a+', encoding='utf-8') as fout:
		for u,v,a in H.edges(data=True):
			count = int(int(a['count'])/2)
			fout.writelines(str(u) +','+str(v) +','+str(count)+'\n')
	print(str(H.number_of_edges()) + ' user pairs saved.')




	#print('Initialize coords and values...')
	#coords = []
	#values = []
	#coords_dict = {}

	#print('Update user matrix...')
	#for key, value in item_user_dict.items():
	#	users = value.split(';')
	#	for s in users:
	#		for t in users:
	#			if s != t:
	#				key = str(s) + ',' + str(t)

	#				if key in coords_dict:
	#					coords_dict[key] += 1
	#				else:
	#					coords_dict[key] = 1
	#				#user_matrix[user_index_dict[s],user_index_dict[t]] += 1
	#print('Update user matrix Completed.')
	
	#print('Saving user matrix...')
	#already_printed = []
	#with open(o_file, 'a+', encoding='utf-8') as fout:
	#	for key, value in coords_dict.items():
	#		key_pairs = key.split(',')
	#		reverse_key = str(key_pairs[1]) +','+str(key_pairs[0])
	#		if key in already_printed or reverse_key in already_printed:
	#			continue
	#		else:
	#			fout.writelines(str(key) +','+str(value)+'\n')
	#			already_printed.append(key)

	#print(str(len(already_printed)) + ' user pairs saved.')

def generateItemCoParticpationGraph (i_file,o_file):
	print('Loading user-item data... \n')
	user_item_df = pd.read_csv(i_file, header=None)
	user_item_df.columns = ['uid','rid','activity']

	user_item_dict = {}
	unique_items = []

	for index, row in user_item_df.iterrows():
		uid = str(row['uid']).strip()
		rid = str(row['rid']).strip()

		#update the repository-user dict
		if uid in user_item_dict:
			user_item_dict[uid] = user_item_dict[uid] + ';' + rid
		else:
			user_item_dict[uid] = rid

		#update unique_users list
		unique_items.append(rid)

	unique_items = set(unique_items)
	unique_items = sorted(unique_items,key=int)

	print('User-item data loaded')
	print('#user: ' +str(len(user_item_dict)))
	print('#items: ' +str(len(unique_items)))

	print('Initialize graph...')
	G = nx.Graph()
	for key, value in user_item_dict.items():
		users = value.split(';')
		for s in users:
			for t in users:	
				if s != t:
					G.add_node(str(s))
					G.add_node(str(t))
					if G.has_edge(str(s),str(t)):
						G[str(s)][str(t)]['count'] += 1
					else:
						G.add_edge(str(s),str(t), count=1)
	print('Initilize graph completed')

	print('Convert graph to undirected...')
	H = G.to_undirected()
	print('Conversion completed.')

	print('Saving graph...')
	with open(o_file, 'a+', encoding='utf-8') as fout:
		for u,v,a in H.edges(data=True):
			count = int(int(a['count'])/2)
			fout.writelines(str(u) +','+str(v) +','+str(count)+'\n')
	print(str(H.number_of_edges()) + ' item pairs saved.')

def generateMFTrainTestNonNegGithub(user_item_file, train_ratio, n_Neg, n_users):
	all_pos_df = pd.read_csv(user_item_file)

	user_all_count = {}
	user_fork_count = {}
	user_watch_count = {}

	l_items = []
	user_pos_all = {}
	user_pos_fork = {}
	user_pos_watch = {}

	for i in range(n_users):
		user_all_count[i] = 0
		user_fork_count[i] = 0
		user_watch_count[i] = 0
		user_pos_fork[i] = []
		user_pos_watch[i] = []
		user_pos_all[i] = []

	for index, row in all_pos_df.iterrows():
		uid = int(row['uid'])
		rid = str(row['rid']).strip()
		fork = int(row['fork'])
		watch = int(row['watch'])
		l_items.append(rid)
		if fork == 1:
			user_pos_fork[uid].append(rid)
			user_fork_count[uid] += 1
		if watch == 1:
			user_pos_watch[uid].append(rid)
			user_watch_count[uid] +=1
		if fork ==1 or watch ==1:
			user_pos_all[uid].append(rid)
			user_all_count[uid] += 1


	s_items = set(l_items)
	
	#The training-test split ratio is at user level.
	user_train_all_pos_count = {}
	user_train_fork_pos_count = {}
	user_train_watch_pos_count = {}
	
	user_test_all_pos_count = {}
	user_test_fork_pos_count = {}
	user_test_watch_pos_count = {}

	for key,value in user_all_count.items():
		#print(str(user_fork_count[key])+' '+str(math.floor(user_fork_count[key] * train_ratio)))
		user_train_all_pos_count[key] = math.ceil(user_all_count[key] * train_ratio)
		user_test_all_pos_count[key] = user_all_count[key] - user_train_all_pos_count[key]

		user_train_fork_pos_count[key] = math.ceil(user_fork_count[key] * train_ratio)
		user_test_fork_pos_count[key] = user_fork_count[key] - user_train_fork_pos_count[key]
		
		user_train_watch_pos_count[key] = math.ceil(user_watch_count[key] * train_ratio)
		user_test_watch_pos_count[key] = user_watch_count[key] - user_train_watch_pos_count[key]

	train_all_dict={}
	test_all_dict={}
	train_fork_dict={}
	test_fork_dict={}
	train_watch_dict={}
	test_watch_dict={}
	train_all_index = 0
	test_all_index=0
	train_fork_index = 0
	test_fork_index=0
	train_watch_index = 0
	test_watch_index=0
	#Add positive records to train and test df
	print('Working on positive now...')
	for index, row in all_pos_df.iterrows():
		uid = int(row['uid'])
		rid = str(row['rid']).strip()
		fork = int(row['fork'])
		watch = int(row['watch'])

		if user_train_all_pos_count[uid] != 0:
			train_all_dict[train_all_index] = {'uid':int(uid),'rid':rid,'rate':1}
			user_train_all_pos_count[uid] -= 1
			train_all_index += 1
		else:
			test_all_dict[test_all_index] = {'uid':int(uid),'rid':rid,'rate':1}
			test_all_index += 1

		if fork == 1:
			if user_train_fork_pos_count[uid] != 0:
				train_fork_dict[train_fork_index] = {'uid':int(uid),'rid':rid,'rate':1}
				user_train_fork_pos_count[uid] -= 1
				train_fork_index +=1
			else:
				test_fork_dict[test_fork_index] = {'uid':int(uid),'rid':rid,'rate':1}
				test_fork_index +=1

		if watch == 1:
			if user_train_watch_pos_count[uid] != 0:
				train_watch_dict[train_watch_index]  = {'uid':int(uid),'rid':rid,'rate':1}
				user_train_watch_pos_count[uid] -= 1
				train_watch_index +=1
			else:
				test_watch_dict[test_watch_index] = {'uid':int(uid),'rid':rid,'rate':1}
				test_watch_index +=1
			
	train_all_df = pd.DataFrame.from_dict(train_all_dict,'index')
	test_all_df = pd.DataFrame.from_dict(test_all_dict,'index')
	
	train_fork_df = pd.DataFrame.from_dict(train_fork_dict,'index')
	test_fork_df = pd.DataFrame.from_dict(test_fork_dict,'index')
	
	train_watch_df = pd.DataFrame.from_dict(train_watch_dict,'index')
	test_watch_df = pd.DataFrame.from_dict(test_watch_dict,'index')

	train_all_df.to_csv('full_data_v2/GH_no_negative_train_test/train_all.csv')
	test_all_df.to_csv('full_data_v2/GH_no_negative_train_test/test_all.csv')

	train_fork_df.to_csv('full_data_v2/GH_no_negative_train_test/train_fork.csv')
	test_fork_df.to_csv('full_data_v2/GH_no_negative_train_test/test_fork.csv')

	train_watch_df.to_csv('full_data_v2/GH_no_negative_train_test/train_watch.csv')
	test_watch_df.to_csv('full_data_v2/GH_no_negative_train_test/test_watch.csv')

def generateMFTrainTestNonNegStackoverflow(user_item_file, train_ratio, n_Neg, n_users):
	all_pos_df = pd.read_csv(user_item_file)

	user_all_count = {}
	user_answer_count = {}
	user_favorite_count = {}

	l_items = []
	user_pos_all = {}
	user_pos_answer = {}
	user_pos_favorite = {}

	for i in range(n_users):
		user_all_count[i] = 0
		user_answer_count[i] = 0
		user_favorite_count[i] = 0
		user_pos_answer[i] = []
		user_pos_favorite[i] = []
		user_pos_all[i] = []

	for index, row in all_pos_df.iterrows():
		uid = int(row['uid'])
		rid = str(row['rid']).strip()
		answer = int(row['answer'])
		favorite = int(row['favorite'])
		l_items.append(rid)
		if answer == 1:
			user_pos_answer[uid].append(rid)
			user_answer_count[uid] += 1
		if favorite == 1:
			user_pos_favorite[uid].append(rid)
			user_favorite_count[uid] +=1
		if answer ==1 or favorite ==1:
			user_pos_all[uid].append(rid)
			user_all_count[uid] += 1


	s_items = set(l_items)
	
	#The training-test split ratio is at user level.
	user_train_all_pos_count = {}
	user_train_answer_pos_count = {}
	user_train_favorite_pos_count = {}
	
	user_test_all_pos_count = {}
	user_test_answer_pos_count = {}
	user_test_favorite_pos_count = {}

	for key,value in user_all_count.items():
		#print(str(user_answer_count[key])+' '+str(math.floor(user_answer_count[key] * train_ratio)))
		user_train_all_pos_count[key] = math.ceil(user_all_count[key] * train_ratio)
		user_test_all_pos_count[key] = user_all_count[key] - user_train_all_pos_count[key]

		user_train_answer_pos_count[key] = math.ceil(user_answer_count[key] * train_ratio)
		user_test_answer_pos_count[key] = user_answer_count[key] - user_train_answer_pos_count[key]
		
		user_train_favorite_pos_count[key] = math.ceil(user_favorite_count[key] * train_ratio)
		user_test_favorite_pos_count[key] = user_favorite_count[key] - user_train_favorite_pos_count[key]

	train_all_dict={}
	test_all_dict={}
	train_answer_dict={}
	test_answer_dict={}
	train_favorite_dict={}
	test_favorite_dict={}
	train_all_index = 0
	test_all_index=0
	train_answer_index = 0
	test_answer_index=0
	train_favorite_index = 0
	test_favorite_index=0
	#Add positive records to train and test df
	print('Working on positive now...')
	for index, row in all_pos_df.iterrows():
		uid = int(row['uid'])
		rid = str(row['rid']).strip()
		answer = int(row['answer'])
		favorite = int(row['favorite'])
		
		if user_train_all_pos_count[uid] != 0:
			train_all_dict[train_all_index] = {'uid':int(uid),'rid':rid,'rate':1}
			user_train_all_pos_count[uid] -= 1
			train_all_index += 1
		else:
			test_all_dict[test_all_index] = {'uid':int(uid),'rid':rid,'rate':1}
			test_all_index += 1

		if answer == 1:
			if user_train_answer_pos_count[uid] != 0:
				train_answer_dict[train_answer_index] = {'uid':int(uid),'rid':rid,'rate':1}
				user_train_answer_pos_count[uid] -= 1
				train_answer_index +=1
			else:
				test_answer_dict[test_answer_index] = {'uid':int(uid),'rid':rid,'rate':1}
				test_answer_index +=1

		if favorite == 1:
			if user_train_favorite_pos_count[uid] != 0:
				train_favorite_dict[train_favorite_index]  = {'uid':int(uid),'rid':rid,'rate':1}
				user_train_favorite_pos_count[uid] -= 1
				train_favorite_index +=1
			else:
				test_favorite_dict[test_favorite_index] = {'uid':int(uid),'rid':rid,'rate':1}
				test_favorite_index +=1
			
	train_all_df = pd.DataFrame.from_dict(train_all_dict,'index')
	test_all_df = pd.DataFrame.from_dict(test_all_dict,'index')
	
	train_answer_df = pd.DataFrame.from_dict(train_answer_dict,'index')
	test_answer_df = pd.DataFrame.from_dict(test_answer_dict,'index')
	
	train_favorite_df = pd.DataFrame.from_dict(train_favorite_dict,'index')
	test_favorite_df = pd.DataFrame.from_dict(test_favorite_dict,'index')

	train_all_df.to_csv('full_data_v2/SO_no_negative_train_test/train_all.csv')
	test_all_df.to_csv('full_data_v2/SO_no_negative_train_test/test_all.csv')

	train_answer_df.to_csv('full_data_v2/SO_no_negative_train_test/train_answer.csv')
	test_answer_df.to_csv('full_data_v2/SO_no_negative_train_test/test_answer.csv')

	train_favorite_df.to_csv('full_data_v2/SO_no_negative_train_test/train_favorite.csv')
	test_favorite_df.to_csv('full_data_v2/SO_no_negative_train_test/test_favorite.csv')

def generateFMTrainingTestingWithNegSampleGithub(user_item_file, train_ratio, n_Neg, n_users):
	all_pos_df = pd.read_csv(user_item_file)

	user_all_count = {}
	user_fork_count = {}
	user_watch_count = {}

	l_items = []
	user_pos_all = {}
	user_pos_fork = {}
	user_pos_watch = {}

	for i in range(n_users):
		user_all_count[i] = 0
		user_fork_count[i] = 0
		user_watch_count[i] = 0
		user_pos_fork[i] = []
		user_pos_watch[i] = []
		user_pos_all[i] = []

	for index, row in all_pos_df.iterrows():
		uid = int(row['uid'])
		rid = str(row['rid']).strip()
		fork = int(row['fork'])
		watch = int(row['watch'])
		l_items.append(rid)
		if fork == 1:
			user_pos_fork[uid].append(rid)
			user_fork_count[uid] += 1
		if watch == 1:
			user_pos_watch[uid].append(rid)
			user_watch_count[uid] +=1
		if fork ==1 or watch ==1:
			user_pos_all[uid].append(rid)
			user_all_count[uid] += 1


	s_items = set(l_items)
	
	#The training-test split ratio is at user level.
	user_train_all_pos_count = {}
	user_train_fork_pos_count = {}
	user_train_watch_pos_count = {}
	user_train_all_neg_count = {}
	user_train_fork_neg_count = {}
	user_train_watch_neg_count = {}
	
	user_test_all_pos_count = {}
	user_test_fork_pos_count = {}
	user_test_watch_pos_count = {}
	user_test_all_neg_count = {}
	user_test_fork_neg_count = {}
	user_test_watch_neg_count = {}

	for key,value in user_all_count.items():
		#print(str(user_fork_count[key])+' '+str(math.floor(user_fork_count[key] * train_ratio)))
		user_train_all_pos_count[key] = math.ceil(user_all_count[key] * train_ratio)
		user_test_all_pos_count[key] = user_all_count[key] - user_train_all_pos_count[key]
		user_train_all_neg_count[key] = user_train_all_pos_count[key] * n_Neg
		user_test_all_neg_count[key] = user_test_all_pos_count[key] * n_Neg

		user_train_fork_pos_count[key] = math.ceil(user_fork_count[key] * train_ratio)
		user_test_fork_pos_count[key] = user_fork_count[key] - user_train_fork_pos_count[key]
		user_train_fork_neg_count[key] = user_train_fork_pos_count[key] * n_Neg
		user_test_fork_neg_count[key] = user_test_fork_pos_count[key] * n_Neg
		
		user_train_watch_pos_count[key] = math.ceil(user_watch_count[key] * train_ratio)
		user_test_watch_pos_count[key] = user_watch_count[key] - user_train_watch_pos_count[key]
		user_train_watch_neg_count[key] = user_train_watch_pos_count[key] * n_Neg
		user_test_watch_neg_count[key] = user_test_watch_pos_count[key] * n_Neg

	train_all_dict={}
	test_all_dict={}
	train_fork_dict={}
	test_fork_dict={}
	train_watch_dict={}
	test_watch_dict={}
	train_all_index = 0
	test_all_index=0
	train_fork_index = 0
	test_fork_index=0
	train_watch_index = 0
	test_watch_index=0
	#Add positive records to train and test df
	print('Working on positive now...')
	for index, row in all_pos_df.iterrows():
		uid = int(row['uid'])
		rid = str(row['rid']).strip()
		fork = int(row['fork'])
		watch = int(row['watch'])

		if user_train_all_pos_count[uid] != 0:
			train_all_dict[train_all_index] = {'uid':int(uid),'rid':rid,'rate':1}
			user_train_all_pos_count[uid] -= 1
			train_all_index += 1
		else:
			test_all_dict[test_all_index] = {'uid':int(uid),'rid':rid,'rate':1}
			test_all_index += 1

		if fork == 1:
			if user_train_fork_pos_count[uid] != 0:
				train_fork_dict[train_fork_index] = {'uid':int(uid),'rid':rid,'rate':1}
				user_train_fork_pos_count[uid] -= 1
				train_fork_index +=1
			else:
				test_fork_dict[test_fork_index] = {'uid':int(uid),'rid':rid,'rate':1}
				test_fork_index +=1

		if watch == 1:
			if user_train_watch_pos_count[uid] != 0:
				train_watch_dict[train_watch_index]  = {'uid':int(uid),'rid':rid,'rate':1}
				user_train_watch_pos_count[uid] -= 1
				train_watch_index +=1
			else:
				test_watch_dict[test_watch_index] = {'uid':int(uid),'rid':rid,'rate':1}
				test_watch_index +=1

	#Add negative samples to train and test df
	print('Working on negative now...')
	for key, value in user_train_all_neg_count.items():
		print('train negative: '+ str(key))
		#Negative sample for all items
		if value > 0:
			all_neg_samples = (s_items - set(user_pos_all[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				train_all_dict[train_all_index] = {'uid':int(uid),'rid':item,'rate':0}
				train_all_index+=1
				user_pos_all[uid].append(item)
		#Negative sample for fork items
		if user_train_fork_pos_count[key] > 0:
			all_neg_samples = (s_items - set(user_pos_fork[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				train_fork_dict[train_fork_index] = {'uid':int(uid),'rid':item,'rate':0}
				train_fork_index+=1
				user_pos_fork[uid].append(item)
		#Negative sample for watch items
		if user_train_watch_pos_count[key] > 0:
			all_neg_samples = (s_items - set(user_pos_watch[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				train_watch_dict[train_watch_index] = {'uid':int(uid),'rid':item,'rate':0}
				train_watch_index+=1
				user_pos_watch[uid].append(item)

	for key, value in user_test_all_neg_count.items():
		print('test negative: '+ str(key))
		#Negative sample for all items
		if value > 0:
			all_neg_samples = (s_items - set(user_pos_all[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				test_all_dict[test_all_index] = {'uid':int(uid),'rid':item,'rate':0}
				test_all_index+=1
		#Negative sample for fork items
		if user_test_fork_pos_count[key] > 0:
			all_neg_samples = (s_items - set(user_pos_fork[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				test_fork_dict[test_fork_index] = {'uid':int(uid),'rid':item,'rate':0}
				test_fork_index+=1
		#Negative sample for watch items
		if user_test_watch_pos_count[key] > 0:
			all_neg_samples = (s_items - set(user_pos_watch[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				test_watch_dict[test_watch_index] = {'uid':int(uid),'rid':item,'rate':0}
				test_watch_index+=1
			
	train_all_df = pd.DataFrame.from_dict(train_all_dict,'index')
	test_all_df = pd.DataFrame.from_dict(test_all_dict,'index')
	
	train_fork_df = pd.DataFrame.from_dict(train_fork_dict,'index')
	test_fork_df = pd.DataFrame.from_dict(test_fork_dict,'index')
	
	train_watch_df = pd.DataFrame.from_dict(train_watch_dict,'index')
	test_watch_df = pd.DataFrame.from_dict(test_watch_dict,'index')

	train_all_df.to_csv('full_data_v2/GH_with_negative_train_test/train_all.csv')
	test_all_df.to_csv('full_data_v2/GH_with_negative_train_test/test_all.csv')

	train_fork_df.to_csv('full_data_v2/GH_with_negative_train_test/train_fork.csv')
	test_fork_df.to_csv('full_data_v2/GH_with_negative_train_test/test_fork.csv')

	train_watch_df.to_csv('full_data_v2/GH_with_negative_train_test/train_watch.csv')
	test_watch_df.to_csv('full_data_v2/GH_with_negative_train_test/test_watch.csv')

def generateFMTrainingTestingWithNegSampleStackOverflow(user_item_file, train_ratio, n_Neg, n_users):
	all_pos_df = pd.read_csv(user_item_file)

	user_all_count = {}
	user_answer_count = {}
	user_favorite_count = {}

	l_items = []
	user_pos_all = {}
	user_pos_answer = {}
	user_pos_favorite = {}

	for i in range(n_users):
		user_all_count[i] = 0
		user_answer_count[i] = 0
		user_favorite_count[i] = 0
		user_pos_answer[i] = []
		user_pos_favorite[i] = []
		user_pos_all[i] = []

	for index, row in all_pos_df.iterrows():
		uid = int(row['uid'])
		rid = str(row['rid']).strip()
		answer = int(row['answer'])
		favorite = int(row['favorite'])
		l_items.append(rid)
		if answer == 1:
			user_pos_answer[uid].append(rid)
			user_answer_count[uid] += 1
		if favorite == 1:
			user_pos_favorite[uid].append(rid)
			user_favorite_count[uid] +=1
		if answer ==1 or favorite ==1:
			user_pos_all[uid].append(rid)
			user_all_count[uid] += 1


	s_items = set(l_items)
	
	#The training-test split ratio is at user level.
	user_train_all_pos_count = {}
	user_train_answer_pos_count = {}
	user_train_favorite_pos_count = {}
	user_train_all_neg_count = {}
	user_train_answer_neg_count = {}
	user_train_favorite_neg_count = {}
	
	user_test_all_pos_count = {}
	user_test_answer_pos_count = {}
	user_test_favorite_pos_count = {}
	user_test_all_neg_count = {}
	user_test_answer_neg_count = {}
	user_test_favorite_neg_count = {}

	for key,value in user_all_count.items():
		#print(str(user_answer_count[key])+' '+str(math.floor(user_answer_count[key] * train_ratio)))
		user_train_all_pos_count[key] = math.ceil(user_all_count[key] * train_ratio)
		user_test_all_pos_count[key] = user_all_count[key] - user_train_all_pos_count[key]
		user_train_all_neg_count[key] = user_train_all_pos_count[key] * n_Neg
		user_test_all_neg_count[key] = user_test_all_pos_count[key] * n_Neg

		user_train_answer_pos_count[key] = math.ceil(user_answer_count[key] * train_ratio)
		user_test_answer_pos_count[key] = user_answer_count[key] - user_train_answer_pos_count[key]
		user_train_answer_neg_count[key] = user_train_answer_pos_count[key] * n_Neg
		user_test_answer_neg_count[key] = user_test_answer_pos_count[key] * n_Neg
		
		user_train_favorite_pos_count[key] = math.ceil(user_favorite_count[key] * train_ratio)
		user_test_favorite_pos_count[key] = user_favorite_count[key] - user_train_favorite_pos_count[key]
		user_train_favorite_neg_count[key] = user_train_favorite_pos_count[key] * n_Neg
		user_test_favorite_neg_count[key] = user_test_favorite_pos_count[key] * n_Neg

	train_all_dict={}
	test_all_dict={}
	train_answer_dict={}
	test_answer_dict={}
	train_favorite_dict={}
	test_favorite_dict={}
	train_all_index = 0
	test_all_index=0
	train_answer_index = 0
	test_answer_index=0
	train_favorite_index = 0
	test_favorite_index=0
	#Add positive records to train and test df
	print('Working on positive now...')
	for index, row in all_pos_df.iterrows():
		uid = int(row['uid'])
		rid = str(row['rid']).strip()
		answer = int(row['answer'])
		favorite = int(row['favorite'])

		if user_train_all_pos_count[uid] != 0:
			train_all_dict[train_all_index] = {'uid':int(uid),'rid':rid,'rate':1}
			user_train_all_pos_count[uid] -= 1
			train_all_index += 1
		else:
			test_all_dict[test_all_index] = {'uid':int(uid),'rid':rid,'rate':1}
			test_all_index += 1

		if answer == 1:
			if user_train_answer_pos_count[uid] != 0:
				train_answer_dict[train_answer_index] = {'uid':int(uid),'rid':rid,'rate':1}
				user_train_answer_pos_count[uid] -= 1
				train_answer_index +=1
			else:
				test_answer_dict[test_answer_index] = {'uid':int(uid),'rid':rid,'rate':1}
				test_answer_index +=1

		if favorite == 1:
			if user_train_favorite_pos_count[uid] != 0:
				train_favorite_dict[train_favorite_index]  = {'uid':int(uid),'rid':rid,'rate':1}
				user_train_favorite_pos_count[uid] -= 1
				train_favorite_index +=1
			else:
				test_favorite_dict[test_favorite_index] = {'uid':int(uid),'rid':rid,'rate':1}
				test_favorite_index +=1

	#Add negative samples to train and test df
	print('Working on negative now...')
	for key, value in user_train_all_neg_count.items():
		print('train negative: '+ str(key))
		#Negative sample for all items
		if value > 0:
			all_neg_samples = (s_items - set(user_pos_all[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				train_all_dict[train_all_index] = {'uid':int(uid),'rid':item,'rate':0}
				train_all_index+=1
				user_pos_all[uid].append(item)
		#Negative sample for answer items
		if user_train_answer_neg_count[key] > 0:
			all_neg_samples = (s_items - set(user_pos_answer[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				train_answer_dict[train_answer_index] = {'uid':int(uid),'rid':item,'rate':0}
				train_answer_index+=1
				user_pos_answer[uid].append(item)
		#Negative sample for favorite items
		if user_train_favorite_neg_count[key] > 0:
			all_neg_samples = (s_items - set(user_pos_favorite[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				train_favorite_dict[train_favorite_index] = {'uid':int(uid),'rid':item,'rate':0}
				train_favorite_index+=1
				user_pos_favorite[uid].append(item)

	for key, value in user_test_all_neg_count.items():
		print('test negative: '+ str(key))
		#Negative sample for all items
		if value > 0:
			all_neg_samples = (s_items - set(user_pos_all[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				test_all_dict[test_all_index] = {'uid':int(uid),'rid':item,'rate':0}
				test_all_index+=1
		#Negative sample for answer items
		if user_train_answer_neg_count[key] > 0:
			all_neg_samples = (s_items - set(user_pos_answer[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				test_answer_dict[test_answer_index] = {'uid':int(uid),'rid':item,'rate':0}
				test_answer_index+=1
		#Negative sample for favorite items
		if user_test_favorite_neg_count[key] > 0:
			all_neg_samples = (s_items - set(user_pos_favorite[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				test_favorite_dict[test_favorite_index] = {'uid':int(uid),'rid':item,'rate':0}
				test_favorite_index+=1
			
	train_all_df = pd.DataFrame.from_dict(train_all_dict,'index')
	test_all_df = pd.DataFrame.from_dict(test_all_dict,'index')
	
	train_answer_df = pd.DataFrame.from_dict(train_answer_dict,'index')
	test_answer_df = pd.DataFrame.from_dict(test_answer_dict,'index')
	
	train_favorite_df = pd.DataFrame.from_dict(train_favorite_dict,'index')
	test_favorite_df = pd.DataFrame.from_dict(test_favorite_dict,'index')

	train_all_df.to_csv('full_data_v2/SO_with_negative_train_test/train_all.csv')
	test_all_df.to_csv('full_data_v2/SO_with_negative_train_test/test_all.csv')

	train_answer_df.to_csv('full_data_v2/SO_with_negative_train_test/train_answer.csv')
	test_answer_df.to_csv('full_data_v2/SO_with_negative_train_test/test_answer.csv')

	train_favorite_df.to_csv('full_data_v2/SO_with_negative_train_test/train_favorite.csv')
	test_favorite_df.to_csv('full_data_v2/SO_with_negative_train_test/test_favorite.csv')

def generateFMTrainingTestingWithNegSampleStackOverflowWithPartition(user_item_file, train_ratio, n_Neg, n_users, n_start, n_end):
	all_pos_df = pd.read_csv(user_item_file)

	user_all_count = {}
	user_answer_count = {}
	user_favorite_count = {}

	l_items = []
	user_pos_all = {}
	user_pos_answer = {}
	user_pos_favorite = {}

	for i in range(n_users):
		user_all_count[i] = 0
		user_answer_count[i] = 0
		user_favorite_count[i] = 0
		user_pos_answer[i] = []
		user_pos_favorite[i] = []
		user_pos_all[i] = []

	for index, row in all_pos_df.iterrows():
		uid = int(row['uid'])
		rid = str(row['rid']).strip()
		answer = int(row['answer'])
		favorite = int(row['favorite'])
		l_items.append(rid)
		if answer == 1:
			user_pos_answer[uid].append(rid)
			user_answer_count[uid] += 1
		if favorite == 1:
			user_pos_favorite[uid].append(rid)
			user_favorite_count[uid] +=1
		if answer ==1 or favorite ==1:
			user_pos_all[uid].append(rid)
			user_all_count[uid] += 1

	s_items = set(l_items)

	#The training-test split ratio is at user level.
	user_train_all_pos_count = {}
	user_train_answer_pos_count = {}
	user_train_favorite_pos_count = {}
	user_train_all_neg_count = {}
	user_train_answer_neg_count = {}
	user_train_favorite_neg_count = {}
	
	user_test_all_pos_count = {}
	user_test_answer_pos_count = {}
	user_test_favorite_pos_count = {}
	user_test_all_neg_count = {}
	user_test_answer_neg_count = {}
	user_test_favorite_neg_count = {}

	#for key,value in user_all_count.items():
	for i in range(n_start, n_end):	
		#print(str(user_answer_count[key])+' '+str(math.floor(user_answer_count[key] * train_ratio)))
		user_train_all_pos_count[i] = math.ceil(user_all_count[i] * train_ratio)
		user_test_all_pos_count[i] = user_all_count[i] - user_train_all_pos_count[i]
		user_train_all_neg_count[i] = user_train_all_pos_count[i] * n_Neg
		user_test_all_neg_count[i] = user_test_all_pos_count[i] * n_Neg

		user_train_answer_pos_count[i] = math.ceil(user_answer_count[i] * train_ratio)
		user_test_answer_pos_count[i] = user_answer_count[i] - user_train_answer_pos_count[i]
		user_train_answer_neg_count[i] = user_train_answer_pos_count[i] * n_Neg
		user_test_answer_neg_count[i] = user_test_answer_pos_count[i] * n_Neg
		
		user_train_favorite_pos_count[i] = math.ceil(user_favorite_count[i] * train_ratio)
		user_test_favorite_pos_count[i] = user_favorite_count[i] - user_train_favorite_pos_count[i]
		user_train_favorite_neg_count[i] = user_train_favorite_pos_count[i] * n_Neg
		user_test_favorite_neg_count[i] = user_test_favorite_pos_count[i] * n_Neg

	train_all_dict={}
	test_all_dict={}
	train_answer_dict={}
	test_answer_dict={}
	train_favorite_dict={}
	test_favorite_dict={}
	train_all_index = 0
	test_all_index=0
	train_answer_index = 0
	test_answer_index=0
	train_favorite_index = 0
	test_favorite_index=0
	#Add positive records to train and test df
	print('Working on positive now...')
	for index, row in all_pos_df.iterrows():
		uid = int(row['uid'])
		rid = str(row['rid']).strip()
		answer = int(row['answer'])
		favorite = int(row['favorite'])
		if uid >=n_start and uid < n_end:
			if user_train_all_pos_count[uid] != 0:
				train_all_dict[train_all_index] = {'uid':int(uid),'rid':rid,'rate':1}
				user_train_all_pos_count[uid] -= 1
				train_all_index += 1
			else:
				test_all_dict[test_all_index] = {'uid':int(uid),'rid':rid,'rate':1}
				test_all_index += 1

			if answer == 1:
				if user_train_answer_pos_count[uid] != 0:
					train_answer_dict[train_answer_index] = {'uid':int(uid),'rid':rid,'rate':1}
					user_train_answer_pos_count[uid] -= 1
					train_answer_index +=1
				else:
					test_answer_dict[test_answer_index] = {'uid':int(uid),'rid':rid,'rate':1}
					test_answer_index +=1

			if favorite == 1:
				if user_train_favorite_pos_count[uid] != 0:
					train_favorite_dict[train_favorite_index]  = {'uid':int(uid),'rid':rid,'rate':1}
					user_train_favorite_pos_count[uid] -= 1
					train_favorite_index +=1
				else:
					test_favorite_dict[test_favorite_index] = {'uid':int(uid),'rid':rid,'rate':1}
					test_favorite_index +=1

	#Add negative samples to train and test df
	print('Working on negative now...')
	for key, value in user_train_all_neg_count.items():
		print('train negative: '+ str(key))
		#Negative sample for all items
		if value > 0:
			all_neg_samples = (s_items - set(user_pos_all[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				train_all_dict[train_all_index] = {'uid':int(key),'rid':item,'rate':0}
				train_all_index+=1
				user_pos_all[key].append(item)
		#Negative sample for answer items
		if user_train_answer_neg_count[key] > 0:
			all_neg_samples = (s_items - set(user_pos_answer[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				train_answer_dict[train_answer_index] = {'uid':int(key),'rid':item,'rate':0}
				train_answer_index+=1
				user_pos_answer[key].append(item)
		#Negative sample for favorite items
		if user_train_favorite_neg_count[key] > 0:
			all_neg_samples = (s_items - set(user_pos_favorite[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				train_favorite_dict[train_favorite_index] = {'uid':int(key),'rid':item,'rate':0}
				train_favorite_index+=1
				user_pos_favorite[key].append(item)

	for key, value in user_test_all_neg_count.items():
		print('test negative: '+ str(key))
		#Negative sample for all items
		if value > 0:
			all_neg_samples = (s_items - set(user_pos_all[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				test_all_dict[test_all_index] = {'uid':int(key),'rid':item,'rate':0}
				test_all_index+=1
		#Negative sample for answer items
		if user_train_answer_neg_count[key] > 0:
			all_neg_samples = (s_items - set(user_pos_answer[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				test_answer_dict[test_answer_index] = {'uid':int(key),'rid':item,'rate':0}
				test_answer_index+=1
		#Negative sample for favorite items
		if user_test_favorite_neg_count[key] > 0:
			all_neg_samples = (s_items - set(user_pos_favorite[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				test_favorite_dict[test_favorite_index] = {'uid':int(key),'rid':item,'rate':0}
				test_favorite_index+=1
			
	train_all_df = pd.DataFrame.from_dict(train_all_dict,'index')
	test_all_df = pd.DataFrame.from_dict(test_all_dict,'index')
	
	train_answer_df = pd.DataFrame.from_dict(train_answer_dict,'index')
	test_answer_df = pd.DataFrame.from_dict(test_answer_dict,'index')
	
	train_favorite_df = pd.DataFrame.from_dict(train_favorite_dict,'index')
	test_favorite_df = pd.DataFrame.from_dict(test_favorite_dict,'index')

	train_all_df.to_csv('full_data_v3/SO_with_negative_train_test/train_all_'+str(n_start)+'.csv')
	test_all_df.to_csv('full_data_v3/SO_with_negative_train_test/test_all_'+str(n_start)+'.csv')

	train_answer_df.to_csv('full_data_v3/SO_with_negative_train_test/train_answer_'+str(n_start)+'.csv')
	test_answer_df.to_csv('full_data_v3/SO_with_negative_train_test/test_answer_'+str(n_start)+'.csv')

	train_favorite_df.to_csv('full_data_v3/SO_with_negative_train_test/train_favorite_'+str(n_start)+'.csv')
	test_favorite_df.to_csv('full_data_v3/SO_with_negative_train_test/test_favorite_'+str(n_start)+'.csv')

def generateFMTrainingTestingWithNegSampleGithubWithPartition(user_item_file, train_ratio, n_Neg, n_users, n_start, n_end):
	all_pos_df = pd.read_csv(user_item_file)

	user_all_count = {}
	user_fork_count = {}
	user_watch_count = {}

	l_items = []
	user_pos_all = {}
	user_pos_fork = {}
	user_pos_watch = {}

	for i in range(n_users):
		user_all_count[i] = 0
		user_fork_count[i] = 0
		user_watch_count[i] = 0
		user_pos_fork[i] = []
		user_pos_watch[i] = []
		user_pos_all[i] = []

	for index, row in all_pos_df.iterrows():
		uid = int(row['uid'])
		rid = str(row['rid']).strip()
		fork = int(row['fork'])
		watch = int(row['watch'])
		l_items.append(rid)
		if fork == 1:
			user_pos_fork[uid].append(rid)
			user_fork_count[uid] += 1
		if watch == 1:
			user_pos_watch[uid].append(rid)
			user_watch_count[uid] +=1
		if fork ==1 or watch ==1:
			user_pos_all[uid].append(rid)
			user_all_count[uid] += 1

	s_items = set(l_items)

	#The training-test split ratio is at user level.
	user_train_all_pos_count = {}
	user_train_fork_pos_count = {}
	user_train_watch_pos_count = {}
	user_train_all_neg_count = {}
	user_train_fork_neg_count = {}
	user_train_watch_neg_count = {}
	
	user_test_all_pos_count = {}
	user_test_fork_pos_count = {}
	user_test_watch_pos_count = {}
	user_test_all_neg_count = {}
	user_test_fork_neg_count = {}
	user_test_watch_neg_count = {}

	#for key,value in user_all_count.items():
	for i in range(n_start, n_end):	
		#print(str(user_fork_count[key])+' '+str(math.floor(user_fork_count[key] * train_ratio)))
		user_train_all_pos_count[i] = math.ceil(user_all_count[i] * train_ratio)
		user_test_all_pos_count[i] = user_all_count[i] - user_train_all_pos_count[i]
		user_train_all_neg_count[i] = user_train_all_pos_count[i] * n_Neg
		user_test_all_neg_count[i] = user_test_all_pos_count[i] * n_Neg

		user_train_fork_pos_count[i] = math.ceil(user_fork_count[i] * train_ratio)
		user_test_fork_pos_count[i] = user_fork_count[i] - user_train_fork_pos_count[i]
		user_train_fork_neg_count[i] = user_train_fork_pos_count[i] * n_Neg
		user_test_fork_neg_count[i] = user_test_fork_pos_count[i] * n_Neg
		
		user_train_watch_pos_count[i] = math.ceil(user_watch_count[i] * train_ratio)
		user_test_watch_pos_count[i] = user_watch_count[i] - user_train_watch_pos_count[i]
		user_train_watch_neg_count[i] = user_train_watch_pos_count[i] * n_Neg
		user_test_watch_neg_count[i] = user_test_watch_pos_count[i] * n_Neg

	train_all_dict={}
	test_all_dict={}
	train_fork_dict={}
	test_fork_dict={}
	train_watch_dict={}
	test_watch_dict={}
	train_all_index = 0
	test_all_index=0
	train_fork_index = 0
	test_fork_index=0
	train_watch_index = 0
	test_watch_index=0
	#Add positive records to train and test df
	print('Working on positive now...')
	for index, row in all_pos_df.iterrows():
		uid = int(row['uid'])
		rid = str(row['rid']).strip()
		fork = int(row['fork'])
		watch = int(row['watch'])
		if uid >=n_start and uid < n_end:
			if user_train_all_pos_count[uid] != 0:
				train_all_dict[train_all_index] = {'uid':int(uid),'rid':rid,'rate':1}
				user_train_all_pos_count[uid] -= 1
				train_all_index += 1
			else:
				test_all_dict[test_all_index] = {'uid':int(uid),'rid':rid,'rate':1}
				test_all_index += 1

			if fork == 1:
				if user_train_fork_pos_count[uid] != 0:
					train_fork_dict[train_fork_index] = {'uid':int(uid),'rid':rid,'rate':1}
					user_train_fork_pos_count[uid] -= 1
					train_fork_index +=1
				else:
					test_fork_dict[test_fork_index] = {'uid':int(uid),'rid':rid,'rate':1}
					test_fork_index +=1

			if watch == 1:
				if user_train_watch_pos_count[uid] != 0:
					train_watch_dict[train_watch_index]  = {'uid':int(uid),'rid':rid,'rate':1}
					user_train_watch_pos_count[uid] -= 1
					train_watch_index +=1
				else:
					test_watch_dict[test_watch_index] = {'uid':int(uid),'rid':rid,'rate':1}
					test_watch_index +=1

	#Add negative samples to train and test df
	print('Working on negative now...')
	for key, value in user_train_all_neg_count.items():
		print('train negative: '+ str(key))
		#Negative sample for all items
		if value > 0:
			all_neg_samples = (s_items - set(user_pos_all[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				train_all_dict[train_all_index] = {'uid':int(key),'rid':item,'rate':0}
				train_all_index+=1
				user_pos_all[key].append(item)
		#Negative sample for fork items
		if user_train_fork_neg_count[key] > 0:
			all_neg_samples = (s_items - set(user_pos_fork[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				train_fork_dict[train_fork_index] = {'uid':int(key),'rid':item,'rate':0}
				train_fork_index+=1
				user_pos_fork[key].append(item)
		#Negative sample for watch items
		if user_train_watch_neg_count[key] > 0:
			all_neg_samples = (s_items - set(user_pos_watch[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				train_watch_dict[train_watch_index] = {'uid':int(key),'rid':item,'rate':0}
				train_watch_index+=1
				user_pos_watch[key].append(item)

	for key, value in user_test_all_neg_count.items():
		print('test negative: '+ str(key))
		#Negative sample for all items
		if value > 0:
			all_neg_samples = (s_items - set(user_pos_all[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				test_all_dict[test_all_index] = {'uid':int(key),'rid':item,'rate':0}
				test_all_index+=1
		#Negative sample for fork items
		if user_train_fork_neg_count[key] > 0:
			all_neg_samples = (s_items - set(user_pos_fork[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				test_fork_dict[test_fork_index] = {'uid':int(key),'rid':item,'rate':0}
				test_fork_index+=1
		#Negative sample for watch items
		if user_test_watch_neg_count[key] > 0:
			all_neg_samples = (s_items - set(user_pos_watch[key]))
			if len(all_neg_samples) > value:
				neg_samples = random.sample(all_neg_samples, value)
			else:
				neg_samples = all_neg_samples
			for item in neg_samples:
				test_watch_dict[test_watch_index] = {'uid':int(key),'rid':item,'rate':0}
				test_watch_index+=1
			
	train_all_df = pd.DataFrame.from_dict(train_all_dict,'index')
	test_all_df = pd.DataFrame.from_dict(test_all_dict,'index')
	
	train_fork_df = pd.DataFrame.from_dict(train_fork_dict,'index')
	test_fork_df = pd.DataFrame.from_dict(test_fork_dict,'index')
	
	train_watch_df = pd.DataFrame.from_dict(train_watch_dict,'index')
	test_watch_df = pd.DataFrame.from_dict(test_watch_dict,'index')

	train_all_df.to_csv('full_data_v2/GH_with_negative_train_test/train_all_'+str(n_start)+'.csv')
	test_all_df.to_csv('full_data_v2/GH_with_negative_train_test/test_all_'+str(n_start)+'.csv')

	train_fork_df.to_csv('full_data_v2/GH_with_negative_train_test/train_fork_'+str(n_start)+'.csv')
	test_fork_df.to_csv('full_data_v2/GH_with_negative_train_test/test_fork_'+str(n_start)+'.csv')

	train_watch_df.to_csv('full_data_v2/GH_with_negative_train_test/train_watch_'+str(n_start)+'.csv')
	test_watch_df.to_csv('full_data_v2/GH_with_negative_train_test/test_watch_'+str(n_start)+'.csv')

def generateFMDataset(i_train,i_test,i_user_graph, i_item_graph, n_users, n_items, n_tags, o_train, o_test):
	
	print('Loading user graph...')
	user_graph={}
	for i in range(n_users):
		user_graph[i] = []
	with open(i_user_graph, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			src = int(row[0])
			des = int(row[1])
			count = int(row[2])
			user_graph[src].append([des,count])
			user_graph[des].append([src,count])
	
	print('Loading item graph...')
	item_graph={}
	for i in range(n_items):
		item_graph[i]=[]
	with open(i_item_graph, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			rid = int(row[0])
			tag = int(row[1])
			item_graph[rid].append(tag)

	print('Coverting train file...')
	with open(o_train, 'a+', encoding='utf-8') as fout:
		with open(i_train, encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader,None)
			for row in reader:
				record = ""
				uid = int(row[0])
				rid = int(row[1])
				target = int(row[2])

				#write target
				record += str(target) +''
				#write user feature
				record += ' '+str(uid)+':1'
				#write item feature
				record += ' '+str(rid+n_users)+':1'
				#write user graph feature
				for item in user_graph[uid]:
					record += ' '+str(item[0]+n_users+n_items)+':'+str(item[1])
				#write item graph feature
				for item in item_graph[rid]:
					record += ' '+str(item+n_users+n_users+n_items)+':1'
				fout.writelines(record+'\n')

	print('Coverting test file...')
	with open(o_test, 'a+', encoding='utf-8') as fout:
		with open(i_test, encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader,None)
			for row in reader:
				record = ""
				uid = int(row[0])
				rid = int(row[1])
				target = int(row[2])

				#write target
				record += str(target) +''
				#write user feature
				record += ' '+str(uid)+':1'
				#write item feature
				record += ' '+str(rid+n_users)+':1'
				#write user graph feature
				for item in user_graph[uid]:
					record += ' '+str(item[0]+n_users+n_items)+':'+str(item[1])
				#write item graph feature
				for item in item_graph[rid]:
					record += ' '+str(item+n_users+n_users+n_items)+':1'
				fout.writelines(record+'\n')

def generateFMDatasetParition(i_file,i_user_graph, i_item_graph, n_users, n_items, n_tags, o_file,n_start,n_end):
	
	print('Loading user graph...')
	user_graph={}
	for i in range(n_users):
		user_graph[i] = []
	with open(i_user_graph, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			src = int(row[0])
			des = int(row[1])
			count = int(row[2])
			user_graph[src].append([des,count])
			user_graph[des].append([src,count])
	
	print('Loading item graph...')
	item_graph={}
	for i in range(n_items):
		item_graph[i]=[]
	with open(i_item_graph, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			rid = int(row[0])
			tag = int(row[1])
			item_graph[rid].append(tag)

	print('Coverting train file...')
	with open(o_file, 'a+', encoding='utf-8') as fout:
		with open(i_file, encoding='utf-8') as f:
			i=0
			reader = csv.reader(f, delimiter=',')
			next(reader,None)
			for row in reader:
				if i>=n_start and i <n_end:
					record = ""
					uid = int(row[0])
					rid = int(row[1])
					target = int(row[2])

					#write target
					record += str(target) +''
					#write user feature
					record += ' '+str(uid)+':1'
					#write item feature
					record += ' '+str(rid+n_users)+':1'
					#write user graph feature
					for item in user_graph[uid]:
						record += ' '+str(item[0]+n_users+n_items)+':'+str(item[1])
					#write item graph feature
					for item in item_graph[rid]:
						record += ' '+str(item+n_users+n_users+n_items)+':1'
					fout.writelines(record+'\n')

def generateFFMDataset(i_train,i_test,i_user_graph, i_item_graph, n_users, n_items, n_tags, o_train, o_test):
	
	print('Loading user graph...')
	user_graph={}
	for i in range(n_users):
		user_graph[i] = []
	with open(i_user_graph, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			src = int(row[0])
			des = int(row[1])
			count = int(row[2])
			user_graph[src].append([des,count])
			user_graph[des].append([src,count])
	
	print('Loading item graph...')
	item_graph={}
	for i in range(n_items):
		item_graph[i]=[]
	with open(i_item_graph, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			rid = int(row[0])
			tag = int(row[1])
			item_graph[rid].append(tag)

	print('Coverting train file...')
	with open(o_train, 'a+', encoding='utf-8') as fout:
		with open(i_train, encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader,None)
			for row in reader:
				record = ""
				uid = int(row[0])
				rid = int(row[1])
				target = int(row[2])

				#write target
				record += str(target) +''
				#write user feature
				record += ' 1:'+str(uid)+':1'
				#write item feature
				record += ' 2:'+str(rid+n_users)+':1'
				#write user graph feature
				for item in user_graph[uid]:
					record += ' 3:'+str(item[0]+n_users+n_items)+':'+str(item[1])
				#write item graph feature
				for item in item_graph[rid]:
					record += ' 4:'+str(item+n_users+n_users+n_items)+':1'
				fout.writelines(record+'\n')
				break

	print('Coverting test file...')
	with open(o_test, 'a+', encoding='utf-8') as fout:
		with open(i_test, encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader,None)
			for row in reader:
				record = ""
				uid = int(row[0])
				rid = int(row[1])
				target = int(row[2])

				#write target
				record += str(target) +''
				#write user feature
				record += ' 1:'+str(uid)+':1'
				#write item feature
				record += ' 2:'+str(rid+n_users)+':1'
				#write user graph feature
				for item in user_graph[uid]:
					record += ' 3:'+str(item[0]+n_users+n_items)+':'+str(item[1])
				#write item graph feature
				for item in item_graph[rid]:
					record += ' 4:'+str(item+n_users+n_users+n_items)+':1'
				fout.writelines(record+'\n')


def generateFFMDatasetParition(i_file,i_user_graph, i_item_graph, n_users, n_items, n_tags, o_file,n_start,n_end):
	
	print('Loading user graph...')
	user_graph={}
	for i in range(n_users):
		user_graph[i] = []
	with open(i_user_graph, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			src = int(row[0])
			des = int(row[1])
			count = int(row[2])
			user_graph[src].append([des,count])
			user_graph[des].append([src,count])
	
	print('Loading item graph...')
	item_graph={}
	for i in range(n_items):
		item_graph[i]=[]
	with open(i_item_graph, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			rid = int(row[0])
			tag = int(row[1])
			item_graph[rid].append(tag)

	print('Coverting train file...')
	with open(o_file, 'a+', encoding='utf-8') as fout:
		with open(i_file, encoding='utf-8') as f:
			i=0
			reader = csv.reader(f, delimiter=',')
			next(reader,None)
			for row in reader:
				if i>=n_start and i <n_end:
					record = ""
					uid = int(row[0])
					rid = int(row[1])
					target = int(row[2])

					#write target
					record += str(target) +''
					#write user feature
					record += ' 1:'+str(uid)+':1'
					#write item feature
					record += ' 2:'+str(rid+n_users)+':1'
					#write user graph feature
					for item in user_graph[uid]:
						record += ' 3:'+str(item[0]+n_users+n_items)+':'+str(item[1])
					#write item graph feature
					for item in item_graph[rid]:
						record += ' 4:'+str(item+n_users+n_users+n_items)+':1'
					fout.writelines(record+'\n')

def generateItemIndex(i_file,o_file):
	item_df = pd.read_csv(i_file)
	i=0
	with open(o_file, 'a+', encoding='utf-8') as fout:
		for index, row in item_df.iterrows():
			rid = int(row['rid'])
			tag = row['tag']
			fout.writelines(str(i) +','+str(rid) +','+str(tag)+'\n')
			i+=1

def convertItemtoIndex(i_index, i_useritem, o_file):
	item_index_df = pd.read_csv(i_index)
	item_index_df.set_index('rid', inplace=True)

	user_item_df = pd.read_csv(i_useritem)
	with open(o_file, 'a+', encoding='utf-8') as fout:
		fout.writelines('uid,rid,answer,favorite\n')
		for index, row in user_item_df.iterrows():
			uid = int(row['uid'])
			rid = row['rid']
			answer = row['answer']
			favorite = row['favorite']
			fout.writelines(str(uid) +','+str(item_index_df.loc[rid]['index']) +','+str(answer)+','+str(favorite)+'\n')

def retrieveAnswersFromSQL(i_user,i_password,o_file):
	connection = pymysql.connect(host='10.0.106.71',
		user=i_user,
		password=i_password,
		db='roylee',
		charset='utf8mb4',
		cursorclass=pymysql.cursors.DictCursor)
	try:
		with connection.cursor() as cursor:
			# Read a single record
			sql = "SELECT count(*) FROM roylee.sb_soanswer where create_at>='2014-03-01' and create_at<'2015-03-01';"
			cursor.execute(sql)
			result = cursor.fetchall()
			total_rec = int(result[0]['count(*)'])
			print(total_rec)

			with open(o_file, 'a+', encoding='utf-8') as fout:
				batch_size = 1000
				for start_at in range (int(math.ceil(total_rec/1000))):
					sql = "SELECT * FROM roylee.sb_soanswer where create_at>='2014-03-01' and create_at<'2015-03-01' LIMIT %i, %i"
					sql = sql % (start_at * batch_size, batch_size)
					cursor.execute(sql)
					result = cursor.fetchall()
					for item in result:
						fout.writelines(str(item['ownerid']) +','+ str(item['parentid']) + ',' + str(item['tags']) +'\n')
	finally:
		connection.close()

def retrieveFavoriteFromSQL(i_user,i_password,o_file):
	connection = pymysql.connect(host='10.0.106.71',
		user=i_user,
		password=i_password,
		db='roylee',
		charset='utf8mb4',
		cursorclass=pymysql.cursors.DictCursor)
	try:
		with connection.cursor() as cursor:
			# Read a single record
			sql = "SELECT count(*) FROM roylee.sb_sofavorite where create_at>='2014-03-01' and create_at<'2015-03-01';"
			cursor.execute(sql)
			result = cursor.fetchall()
			total_rec = int(result[0]['count(*)'])
			print(total_rec)

			with open(o_file, 'a+', encoding='utf-8') as fout:
				batch_size = 1000
				for start_at in range (int(math.ceil(total_rec/1000))):
					sql = "SELECT * FROM roylee.sb_sofavorite where create_at>='2014-03-01' and create_at<'2015-03-01' LIMIT %i, %i"
					sql = sql % (start_at * batch_size, batch_size)
					cursor.execute(sql)
					result = cursor.fetchall()
					for item in result:
						fout.writelines(str(item['ownerid']) +','+ str(item['parentid']) + ',' + str(item['tags']) +'\n')
	finally:
		connection.close()

def retrieveForkFromSQL(i_user,i_password,o_file):
	connection = pymysql.connect(host='10.0.106.71',
		user=i_user,
		password=i_password,
		db='roylee',
		charset='utf8mb4',
		cursorclass=pymysql.cursors.DictCursor)
	try:
		with connection.cursor() as cursor:
			# Read a single record
			sql = "SELECT count(*) FROM roylee.sb_ghfork where create_at>='2014-03-01' and create_at<'2015-03-01';"
			cursor.execute(sql)
			result = cursor.fetchall()
			total_rec = int(result[0]['count(*)'])
			print(total_rec)

			with open(o_file, 'a+', encoding='utf-8') as fout:
				batch_size = 5000
				for start_at in range (int(math.ceil(total_rec/5000))):
					sql = "SELECT * FROM roylee.sb_ghfork where create_at>='2014-03-01' and create_at<'2015-03-01' LIMIT %i, %i"
					sql = sql % (start_at * batch_size, batch_size)
					cursor.execute(sql)
					result = cursor.fetchall()
					for item in result:
						fout.writelines(str(item['ownerid']) +','+ str(item['fork_from']) + ',' + str(item['tags']) +'\n')
	finally:
		connection.close()

def retrieveWatchFromSQL(i_user,i_password,o_file):
	connection = pymysql.connect(host='10.0.106.71',
		user=i_user,
		password=i_password,
		db='roylee',
		charset='utf8mb4',
		cursorclass=pymysql.cursors.DictCursor)
	try:
		with connection.cursor() as cursor:
			# Read a single record
			sql = "SELECT count(*) FROM roylee.sb_ghwatcher where create_at>='2014-03-01' and create_at<'2015-03-01';"
			cursor.execute(sql)
			result = cursor.fetchall()
			total_rec = int(result[0]['count(*)'])
			print(total_rec)

			with open(o_file, 'a+', encoding='utf-8') as fout:
				batch_size = 1000
				for start_at in range (int(math.ceil(total_rec/1000))):
					sql = "SELECT * FROM roylee.sb_ghwatcher where create_at>='2014-03-01' and create_at<'2015-03-01' LIMIT %i, %i"
					sql = sql % (start_at * batch_size, batch_size)
					cursor.execute(sql)
					result = cursor.fetchall()
					for item in result:
						fout.writelines(str(item['watcherid']) +','+ str(item['repoid']) + ',' + str(item['tags']) +'\n')
	finally:
		connection.close()

def retrieveUserDistribution(i_file, o_file):
	user_item_counts = {}
	with open(i_file, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			uid = str(row[0])
			if uid in user_item_counts:
				user_item_counts[uid] +=1
			else:
				user_item_counts[uid]=1
	with open(o_file, 'a+', encoding='utf-8') as fout:
		for key, value in user_item_counts.items():
			fout.writelines(str(key) +','+ str(value) +'\n')

def retrieveItemDistribution(i_file, o_file):
	user_item_counts = {}
	with open(i_file, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			rid = str(row[1])
			if rid in user_item_counts:
				user_item_counts[rid] +=1
			else:
				user_item_counts[rid]=1
	with open(o_file, 'a+', encoding='utf-8') as fout:
		for key, value in user_item_counts.items():
			fout.writelines(str(key) +','+ str(value) +'\n')

def filterItemsByUsers(i_users,i_userItems, o_userItems):
	users ={}

	with open(i_users, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			users[str(row[1])] = str(row[0])
			
	with open(o_userItems, 'a+', encoding='utf-8') as fout:
		with open(i_userItems, encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			for row in reader:
				uid = str(row[0])
				rid = str(row[1])
				tags = str(row[2])
				if uid in users:
					fout.writelines(str(users[uid]) +','+ str(rid) + ',' + str(tags) +'\n')

def indexItems(i_act_1,i_act_2,o_item_index,o_act_1,o_act_2):
	items_indexes ={}
	item_tags={}
	index =0
	with open(i_act_1, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			rid = str(row[1])
			if rid not in items_indexes:
				items_indexes[rid] = index
				item_tags[rid] = str(row[2])
				index+=1
	
	with open(i_act_2, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			rid = str(row[1])
			if rid not in items_indexes:
				items_indexes[rid] = index
				item_tags[rid] = str(row[2])
				index+=1
	print(len(items_indexes))
	with open(o_act_1, 'a+', encoding='utf-8') as fout:
		with open(i_act_1, encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			for row in reader:
				uid = str(row[0])
				rid = str(row[1])
				tags = str(row[2])
				fout.writelines(str(uid) +','+ str(items_indexes.get(rid)) + ',' + str(tags) +'\n')

	with open(o_act_2, 'a+', encoding='utf-8') as fout:
		with open(i_act_2, encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			for row in reader:
				uid = str(row[0])
				rid = str(row[1])
				tags = str(row[2])
				fout.writelines(str(uid) +','+ str(items_indexes.get(rid)) + ',' + str(tags) +'\n')

	with open(o_item_index, 'a+', encoding='utf-8') as fout:
		for key,value in items_indexes.items():
			fout.writelines(str(value) +','+ str(key)+','+ str(item_tags.get(key))+'\n')

def cleanTrainTest(i_train,i_test,o_train,o_test):
	with open(o_train, 'a+', encoding='utf-8') as fout:
		#fout.writelines	('uid,rid,rate\n')
		with open(i_train, encoding='utf-8') as f:
			train_item = []
			reader = csv.reader(f,delimiter=',')
			for row in reader:
				empty=str(row[0])
				uid = str(row[1])
				rid = str(row[2])
				rate = str(row[3])
				train_item.append(rid)
				fout.writelines	(str(uid) +','+ str(rid) +','+ str(rate)+'\n')
	train_item = set(train_item)
	with open(o_test, 'a+', encoding='utf-8') as fout:
		#fout.writelines	('uid,rid,rate\n')
		with open(i_test, encoding='utf-8') as f:
			reader = csv.reader(f,delimiter=',')
			for row in reader:
				empty=str(row[0])
				uid = str(row[1])
				rid = str(row[2])
				rate = str(row[3])
				if rid in train_item:
					fout.writelines	(str(uid) +','+ str(rid) +','+ str(rate)+'\n')

def testSparseMatrix():
	M = scipy.sparse.lil_matrix((10,100), dtype=int)
	M[0,11] = 1
	#print(M.data.shape)
	#M = scipy.sparse.csc_matrix(M)
	print(M)

	#scipy.sparse.save_npz('user_sparse_matrix.npz', M)
	#scipy.sparse.save_npz('matrix.npz', M)

def testSparseMatrix2():
	sparse_matrix = scipy.sparse.dok_matrix(np.array([[0, 0, 3], [4, 0, 0]]))
	print(sparse_matrix)
	sparse_matrix.todense()
	print(sparse_matrix)
	scipy.sparse.save_npz('sparse_matrix.npz', sparse_matrix)

def testSparseMatrix3():
	coords = np.array([[0,0],[5,5]])
	values = np.array([1,1])
	A = scipy.sparse.coo_matrix((values, coords.T))

	print(A)
	print(A.todense())

def testSparseMatrix3():
	coords = np.array([[0,0],[5,5]])
	values = np.array([1,1])
	A = scipy.sparse.coo_matrix((values, coords.T))

	print(A)
	print(A.todense())

def testSQL():
	connection = pymysql.connect(host='10.0.106.71',
		user='roylee',
		password='LARCdata4848',
		db='roylee',
		charset='utf8mb4',
		cursorclass=pymysql.cursors.DictCursor)
	try:
		with connection.cursor() as cursor:
			# Read a single record
			sql = "SELECT count(*) FROM roylee.sb_sofavorite where create_at>='2013-10-01' and create_at<'2015-03-01' LIMIT 1000;"
			cursor.execute(sql)
			result = cursor.fetchall()
			total_rec = int(result[0]['count(*)'])
			print(total_rec)

			batch_size = 1000
			for start_at in range (int(math.ceil(total_rec/1000))):
				sql = "SELECT * FROM roylee.sb_sofavorite where create_at>='2013-10-01' and create_at<'2015-03-01' LIMIT %i, %i"
				sql = sql % (start_at * batch_size, batch_size)
				cursor.execute(sql)
				result = cursor.fetchall()
				for item in result:
					print(item)
	finally:
		connection.close()
	
def testDict():
	test_Dict = {}
	test_Dict['1,2'] = 1
	test_Dict['1,3'] = 1
	test_Dict['2,1'] = 1
	test_Dict['2,3'] = 1
	for key, value in test_Dict.items():
		pair = key.split(',')
		print(str(pair[0])+','+str(pair[1])+str(value))
		alter_key = str(pair[1])+','+str(pair[0])
		del test_Dict[alter_key]

def testNetworkx():
	d={0: {2: {'count':6}}}
	G=nx.Graph(d)
	print(G.edges())
	print(G.get_edge_data(0,2))

def sigmoid(x):
	return (1 / (1 + np.exp(-x)))

def softmax(x):
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()

def testSigmoid():
	x = softmax([-10,200,300])

	#for i in l:
	print(sigmoid(x[2]))

def mergefiles(type,o_file):
	with open(o_file, 'a+', encoding='utf-8') as fout:
		with open(type+'0.csv', encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader, None)
			for row in reader:
				fout.writelines(str(row[0]) +','+ str(row[1])+','+ str(row[2])+','+ str(row[3])+'\n')
		with open(type+'7000.csv', encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader, None)
			for row in reader:
				fout.writelines(str(row[0]) +','+ str(row[1])+','+ str(row[2])+','+ str(row[3])+'\n')
		with open(type+'14000.csv', encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader, None)
			for row in reader:
				fout.writelines(str(row[0]) +','+ str(row[1])+','+ str(row[2])+','+ str(row[3])+'\n')
		with open(type+'21000.csv', encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader, None)
			for row in reader:
				fout.writelines(str(row[0]) +','+ str(row[1])+','+ str(row[2])+','+ str(row[3])+'\n')
		with open(type+'28000.csv', encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader, None)
			for row in reader:
				fout.writelines(str(row[0]) +','+ str(row[1])+','+ str(row[2])+','+ str(row[3])+'\n')
		with open(type+'35000.csv', encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader, None)
			for row in reader:
				fout.writelines(str(row[0]) +','+ str(row[1])+','+ str(row[2])+','+ str(row[3])+'\n')
		with open(type+'42000.csv', encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader, None)
			for row in reader:
				fout.writelines(str(row[0]) +','+ str(row[1])+','+ str(row[2])+','+ str(row[3])+'\n')
		with open(type+'49000.csv', encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader, None)
			for row in reader:
				fout.writelines(str(row[0]) +','+ str(row[1])+','+ str(row[2])+','+ str(row[3])+'\n')
		with open(type+'56000.csv', encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader, None)
			for row in reader:
				fout.writelines(str(row[0]) +','+ str(row[1])+','+ str(row[2])+','+ str(row[3])+'\n')
		with open(type+'63000.csv', encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader, None)
			for row in reader:
				fout.writelines(str(row[0]) +','+ str(row[1])+','+ str(row[2])+','+ str(row[3])+'\n')
		with open(type+'65000.csv', encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader, None)
			for row in reader:
				fout.writelines(str(row[0]) +','+ str(row[1])+','+ str(row[2])+','+ str(row[3])+'\n')
		with open(type+'67000.csv', encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader, None)
			for row in reader:
				fout.writelines(str(row[0]) +','+ str(row[1])+','+ str(row[2])+','+ str(row[3])+'\n')
		with open(type+'69000.csv', encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader, None)
			for row in reader:
				fout.writelines(str(row[0]) +','+ str(row[1])+','+ str(row[2])+','+ str(row[3])+'\n')
		with open(type+'67000.csv', encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader, None)
			for row in reader:
				fout.writelines(str(row[0]) +','+ str(row[1])+','+ str(row[2])+','+ str(row[3])+'\n')

def removeNegative(i_file, o_file):
	with open(o_file, 'a+', encoding='utf-8') as fout:
		fout.writelines('uid,rid,rate\n')
		with open(i_file, encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader, None)
			for row in reader:
				uid = int(row[0])
				rid = int(row[1])
				rate = int(row[2])
				if rate == 1:
					fout.writelines(str(uid) +','+ str(rid)+','+ str(rate)+'\n')

def filterTrainingNeg (i_file, o_file):
	user_pos_count = {}
	with open(i_file, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		next(reader, None)
		for row in reader:
			uid = int(row[0])
			rid = int(row[1])
			rate = int(row[2])
			if rate == 1:
				if uid not in user_pos_count:
					user_pos_count[uid] = 1
				else:
					user_pos_count[uid] += 1

	with open(o_file, 'a+', encoding='utf-8') as fout:
		fout.writelines('uid,rid,rate\n')
		with open(i_file, encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader, None)
			for row in reader:
				uid = int(row[0])
				rid = int(row[1])
				rate = int(row[2])
				if rate == 1:
					fout.writelines(str(uid) +','+ str(rid)+','+ str(rate)+'\n')
				else:
					if user_pos_count[uid] != 0:
						fout.writelines(str(uid) +','+ str(rid)+','+ str(rate)+'\n')
						user_pos_count[uid] -= 1

def generateNewDataFormat(i_answer,i_favorite,o_file):
	with open(o_file, 'a+', encoding='utf-8') as fout:
		fout.writelines('uid,rid,rate\n')
		with open(i_answer, encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader, None)
			for row in reader:
				uid = int(row[0])
				rid = int(row[1])
				rate = int(row[2])
				fout.writelines(str(uid) +','+ str(rid)+','+ str(rate)+'\n')
		with open(i_favorite, encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader, None)
			for row in reader:
				uid = int(row[0])
				rid = int(row[1])
				rate = int(row[2]) + 1020809
				fout.writelines(str(uid) +','+ str(rid)+','+ str(rate)+'\n')

def generateItemTagGraphV2(i_tag,i_item,o_file):
	tag_index = {}
	with open(i_tag, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			tag_index[row[1]] = int(row[0])

	with open(o_file, 'a+', encoding='utf-8') as fout:
		with open(i_item, encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			for row in reader:
				rid = int(row[0])
				tags = row[2]
				for tag in tags.split(' ; '):
					fout.writelines(str(rid)+','+str(tag_index[tag])+'\n')
				#fout.writelines(str(rid)+','+str(len(tag_index))+'\n')
				for tag in tags.split(' ; '):
					fout.writelines(str(rid+1020809)+','+str(tag_index[tag])+'\n')
				#fout.writelines(str(rid)+','+str(len(tag_index)+1)+'\n')

def testLRAP():
	y_true = np.array([[1, 0, 0], [0, 1, 1]])
	y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
	score = label_ranking_average_precision_score(y_true, y_score)
	print(score)

def generateTrainTestByUser(user_item_file, train_ratio, n_users):
	n_test = math.ceil((1-train_ratio) * n_users)
	user_set = set(list(range(0,n_users)))
	test_set = set(random.sample(user_set, n_test)) 
	train_set = user_set.difference(test_set)

	with open('full_data_v5/gh/train.csv', 'a+', encoding='utf-8') as fTrain:
		fTrain.writelines('uid,rid,fork,watch\n')
		with open('full_data_v5/gh/test.csv', 'a+', encoding='utf-8') as fTest:
			fTest.writelines('uid,rid,fork,watch\n')
			with open(user_item_file, encoding='utf-8') as f:
				reader = csv.reader(f, delimiter=',')
				for row in reader:
					next(reader, None)
					for row in reader:
						uid = int(row[0])
						rid = int(row[1])
						ans = int(row[2])
						fav = int(row[3])
						if uid in train_set:
							fTrain.writelines(str(uid)+','+str(rid)+','+str(ans)+','+str(fav)+'\n')
						else:
							fTest.writelines(str(uid)+','+str(rid)+','+str(ans)+','+str(fav)+'\n')
				
def combinedActivityFiles (i_file1,i_file2, o_file, o_file_neg):
	with open(o_file_neg, 'a+', encoding='utf-8') as fout_neg:
		with open(o_file, 'a+', encoding='utf-8') as fout:
			fout.writelines('uid,rid,rate\n')
			with open(i_file1, encoding='utf-8') as f:
				reader = csv.reader(f, delimiter=',')
				next(reader, None)
				for row in reader:
					uid = int(row[0])
					rid = int(row[1])
					rate = int(row[2])
					if rate == 1:
						fout.writelines(str(uid) +','+ str(rid)+','+str(rate)+',0\n')
					fout_neg.writelines(str(uid) +','+ str(rid)+','+str(rate)+',0\n')
			with open(i_file2, encoding='utf-8') as f:
				reader = csv.reader(f, delimiter=',')
				next(reader, None)
				for row in reader:
					uid = int(row[0])
					rid = int(row[1])
					rate = int(row[2])
					if rate == 1:
						fout.writelines(str(uid) +','+ str(rid)+',0,'+str(rate)+'\n')
					fout_neg.writelines(str(uid) +','+ str(rid)+','+str(rate)+',0\n')

def generateTagsItems(item_graph,o_file):
	tag_items = {}
	tag_index = {}
	with open(item_graph, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			rid = int(row[0])
			tid = int(row[1])
			if tid not in tag_items:
				tag_items[tid] = str(rid)
			else:
				tag_items[tid] = tag_items[tid] + ';'+str(rid)
	with open(o_file, 'a+', encoding='utf-8') as fout:
		for key, value in tag_items.items():
			fout.writelines(str(key) +','+ str(value)+'\n')

def generateUserTags(item_graph,user_item,o_file):
	item_tags = {}
	with open(item_graph, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			rid = int(row[0])
			tid = int(row[1])
			if rid not in item_tags:
				item_tags[rid] = [tid]
			else:
				item_tags[rid].append(tid)

	user_tags = {}
	with open(user_item, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			uid = int(row[0])
			rid = int(row[1])
			if uid not in user_tags:
				user_tags[uid] = item_tags[rid]
			else:
				user_tags[uid].extend(item_tags[rid])

	with open(o_file, 'a+', encoding='utf-8') as fout:
		for uid, tags in tag_items.items():
			tags = set(tags)
			output = str(tags).replace(', ',';').strip('{}')
			fout.writelines(str(uid) +','+ str(output)+'\n')

def generateItemPairsWithSharedTag(i_tag_items, o_file):
	with open(o_file, 'a+', encoding='utf-8') as fout:

		item_pairs = []
		with open(i_tag_items, encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			for row in reader:
				tid = int(row[0])
				items = str(row[1]).split(';')
				for i in items:
					for j in items:
						if i != j:
							item_pairs.append(str(i)+','+str(j))

	s_item_pairs = set(item_pairs)
	for p in s_item_pairs:
		fout.print()

def cleanTrainTest2(i_train,i_test,o_test):
	with open(i_train, encoding='utf-8') as f:
		train_item = []
		reader = csv.reader(f,delimiter=',')
		next(reader, None)
		for row in reader:
			uid = int(row[0])
			rid = int(row[1])
			train_item.append(rid)
	train_item = set(train_item)
	with open(o_test, 'a+', encoding='utf-8') as fout:
		with open(i_test, encoding='utf-8') as f:
			reader = csv.reader(f,delimiter=',')
			next(reader, None)
			for row in reader:
				uid = int(row[0])
				rid = int(row[1])
				ans = int(row[2])
				fav = int(row[3])
				if rid in train_item:
					fout.writelines	(str(uid) +','+ str(rid) +','+ str(ans) +','+ str(fav)+'\n')

def loadItemFeatures(i_file,n_items,n_tags, n_activities):
	print('Loading item graph...')
	#Sparse Matrix of size (IA) x (N_i + A)
	row = n_items
	columns = n_tags
	A = scipy.sparse.lil_matrix((row,columns), dtype=int)
	with open(i_file, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			rid = int(row[0])
			tag = int(row[1])
			A[rid,tag] = 1
		
	manhattan_sim_A = pairwise_distances(np.array(A), np.array(A), metric='manhattan', n_jobs=-1)
	np.save('manhattan_sim_A', manhattan_sim_A)
	print(sim_A)
	#return A

def ComputeNearestNeighbours(i_file,n_items,n_tags, n_activities, k_neighbors, o_file):
	print('Loading item graph...')
	#Sparse Matrix of size (IA) x (N_i + A)
	row = n_items
	columns = n_tags
	A = scipy.sparse.lil_matrix((row,columns), dtype=int)
	with open(i_file, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			rid = int(row[0])
			tag = int(row[1])
			A[rid,tag] = 1

	neigh = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto', metric='cosine', n_jobs=-1)
	neigh.fit(A)
	
	start_index = 320000
	end_index = 325000
	roll_on = True
	with open(o_file, 'a+', encoding='utf-8') as fout:
		while roll_on:
			if end_index >= 400000:
				end_index = 400000
				roll_on = False
			print('Processing '+str(start_index))
			results = neigh.kneighbors(A[start_index:end_index,],return_distance=True)
			for i in range(5000):#n_items):
				for k in range(k_neighbors):
					if int(results[1][i][k]) != i:
						fout.writelines(str(i+start_index)+','+str(results[1][i][k])+','+str(1-results[0][i][k])+'\n')

			start_index += 5000
			end_index += 5000

def combinedNearestNeighbors(o_files):
	with open(o_file, 'a+', encoding='utf-8') as fout:
		for i in range(1,7):
			with open('full_data_v7/so_item_neighbors'+str(i)+'.csv', encoding='utf-8') as f:
				reader = csv.reader(f, delimiter=',')
				for row in reader:
					src = int(row[0])
					des = int(row[1])
					dist = math.round(float(row[3]),3)
					if src != des:
						fout.writelines(str(src)+','+str(des)+','+str(dist)+'\n')

def generatTrainTestSmall(i_train, i_test, o_train, o_test):
	train_users = []
	with open(i_train, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		next(reader, None)
		for row in reader:
			uid = int(row[0])
			train_users.append(uid)

	train_users = set(train_users)
	train_samples = random.choices(list(train_users),k=1000)
	
	with open(o_train, 'a+', encoding='utf-8') as fout:
		with open(i_train, encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader, None)
			for row in reader:
				uid = int(row[0])
				rid = int(row[1])
				act1 = int(row[2])
				act2 = int(row[3])	
				if uid in train_samples:
					fout.writelines	(str(uid) +','+ str(rid) +','+ str(act1) +','+ str(act2)+'\n')

	test_samples = random.choices(list(train_samples),k=200)
	
	with open(o_test, 'a+', encoding='utf-8') as fout:
		with open(i_test, encoding='utf-8') as f:
			reader = csv.reader(f, delimiter=',')
			next(reader, None)
			for row in reader:
				uid = int(row[0])
				rid = int(row[1])
				act1 = int(row[2])
				act2 = int(row[3])	
				if uid in train_samples:
					fout.writelines	(str(uid) +','+ str(rid) +','+ str(act1) +','+ str(act2)+'\n')

def countBothActivities(user_item_file):
	with open(user_item_file, encoding='utf-8') as f:
		reader = csv.reader(f, delimiter=',')
		next(reader, None)
		count = 0
		one_instance = 0
		for row in reader:
			uid = int(row[0])
			rid = int(row[1])
			act1 = int(row[2])
			act2 = int(row[3])
			if act1 == 1 and act2 == 1:
				count+=1
			else:
				one_instance+=1
	print(count)
	print(one_instance)


def setRange():
	#user_set = set(range(0,10))
	#test_set = set(random.sample(user_set, 3)) 
	#train_set = user_set.difference(test_set)
	#print(user_set)
	#print(test_set)
	#print(train_set)
	l = [0,0,2,3,1]
	s = set(l)
	output = str(s).replace(', ',';').strip('{}')
	print(output)
	#print([0]*(10))


if __name__ == "__main__":
	
	###================ Generate User and Item Graphs ==============================
	#generateUserCoParticpationGraph('toy_data_v3/so_user_items.csv', 'toy_data_v3/so_user_user_graph.csv')
	#generateUserCoParticpationGraph('toy_data_v3/gh_user_items.csv', 'toy_data_v3/gh_user_user_graph.csv')
	#generateItemCoParticpationGraph('toy_data_v3/so_user_items.csv', 'toy_data_v3/so_item_item_graph.csv')
	#generateItemCoParticpationGraph('toy_data_v3/gh_user_items.csv', 'toy_data_v3/gh_item_item_graph.csv')

	#generateUserItemGithub('Merlot_Full_Data_190301/gh_users.csv', 
	#						'Merlot_Full_Data_190301/user_fork_repository.csv', 
	#						'Merlot_Full_Data_190301/user_watch_repository.csv', 
	#						'full_data_v1/gh_user_items.csv', 
	#						'full_data_v1/gh_user_index.csv', 10)

	#generateUserItemStackoverflow('Merlot_Full_Data_190301/so_users.csv', 
	#							'Merlot_Full_Data_190301/user_answer_question.csv', 
	#							'Merlot_Full_Data_190301/user_favorite_question.csv', 
	#							'full_data_v1/so_user_items.csv', 
	#							'full_data_v1/so_user_index.csv', 10)

	#generateItemTags('Merlot_Full_Data_190301/gh_repositories.csv', 
	#					'full_data_v1/gh_user_items.csv', 
	#					'full_data_v1/gh_items.csv')

	#generateItemTags('Merlot_Full_Data_190301/so_questions.csv', 
	#					'full_data_v1/so_user_items.csv', 
	#					'full_data_v1/so_items.csv')

	#generateUserCoParticpationGraph('full_data_v1/gh_user_items.csv', 'full_data_v1/gh_user_user_graph.csv')
	#generateUserCoParticpationGraph('full_data_v1/so_user_items.csv', 'full_data_v1/so_user_user_graph.csv')

	#generateMFTrainTestNonNegGithub('full_data_v1/gh_user_items.csv', 0.8, 1,16186)
	#generateMFTrainTestNonNegStackoverflow('full_data_v1/so_user_items.csv', 0.8, 1,11607)

	#generateItemIndex('full_data_v1/gh_items.csv','full_data_v1/gh_item_index.csv')
	#generateItemIndex('full_data_v1/so_items.csv','full_data_v1/so_item_index.csv')
	#convertItemtoIndex('full_data_v1/so_item_index.csv', 'full_data_v1/so_user_items.csv', 'full_data_v1/so_user_items_new.csv')

	#retrieveAnswersFromSQL('roylee','LARCdata4848','full_data_v2/user_answer.csv')
	#retrieveFavoriteFromSQL('roylee','LARCdata4848','full_data_v2/user_favorite.csv')
	#retrieveForkFromSQL('roylee','LARCdata4848','full_data_v2/user_fork.csv')
	#retrieveWatchFromSQL('roylee','LARCdata4848','full_data_v2/user_watch.csv')

	#retrieveUserDistribution('full_data_v2/user_fork.csv', 'full_data_v2/user_fork_count.csv')
	#filterItemsByUsers('full_data_v2/gh_user_index.csv','full_data_v2/user_fork.csv', 'full_data_v2/user_fork_new.csv')
	#filterItemsByUsers('full_data_v2/gh_user_index.csv','full_data_v2/user_watch.csv', 'full_data_v2/user_watch_new.csv')
	#retrieveItemDistribution('full_data_v2/user_watch_new.csv', 'full_data_v2/item_watch_count.csv')
	#indexItems('full_data_v2/user_fork_new.csv','full_data_v2/user_watch_new.csv','full_data_v2/gh_item_index.csv','full_data_v2/user_fork_new2.csv','full_data_v2/user_watch_new2.csv')
	#generateUserItemGithub('full_data_v2/gh_user_index.csv', 'full_data_v2/user_fork_new2.csv', 'full_data_v2/user_watch_new2.csv', 'full_data_v2/gh_user_items.csv', '', 0)
	#generateUserItemStackoverflow('full_data_v2/so_user_index.csv', 'full_data_v2/user_answer_new2.csv', 'full_data_v2/user_favorite_new2.csv', 'full_data_v2/so_user_items.csv', '', 0)
	#generateUserCoParticpationGraph('full_data_v2/gh_user_items.csv', 'full_data_v2/gh_user_user_graph.csv')
	#generateMFTrainTestNonNegGithub('full_data_v2/gh_user_items.csv', 0.8, 1,79628)
	#cleanTrainTest('full_data_v2/GH_no_negative_train_test/train_fork.csv',
	#				'full_data_v2/GH_no_negative_train_test/test_fork.csv',
	#				'full_data_v2/GH_no_negative_train_test/train_fork_new.csv',
	#				'full_data_v2/GH_no_negative_train_test/test_fork_new.csv')

	#generateFMTrainingTestingWithNegSampleStackOverflow('full_data_v2/so_user_items.csv', 0.8, 1,73108)
	#cleanTrainTest('full_data_v2/SO_with_negative_train_test/train_all.csv',
	#				'full_data_v2/SO_with_negative_train_test/test_all.csv',
	#				'full_data_v2/SO_with_negative_train_test/train_all_new.csv',
	#				'full_data_v2/SO_with_negative_train_test/test_all_new.csv')

	#generateFMTrainingTestingWithNegSampleGithub('full_data_v2/gh_user_items.csv', 0.8, 1,79628)
	#cleanTrainTest('full_data_v2/GH_with_negative_train_test/train_all.csv',
	#				'full_data_v2/GH_with_negative_train_test/test_all.csv',
	#				'full_data_v2/GH_with_negative_train_test/train_all_new.csv',
	#				'full_data_v2/GH_with_negative_train_test/test_all_new.csv')

	#genenerateTagIndex('full_data_v2/so_item_index.csv', 'full_data_v2/so_item_index.csv')

	#generateFMDataset('full_data_v2/SO_with_negative_train_test/train_all.csv',
	#				'full_data_v2/SO_with_negative_train_test/test_all.csv',
	#				'full_data_v2/so_user_user_graph.csv', 
	#				'full_data_v2/so_item_graph.csv', 73108, 2024219, 34631, 
	#				'full_data_v2/SO_with_negative_train_test/FM_train_all_x.csv', 
	#				'full_data_v2/SO_with_negative_train_test/FM_train_all_y.csv', 
	#				'full_data_v2/SO_with_negative_train_test/FM_test_all_x.csv', 
	#				'full_data_v2/SO_with_negative_train_test/FM_test_all_y.csv')

	#generateFMDataset('full_data_v2/SO_with_negative_train_test/train_all.csv',
	#				'full_data_v2/SO_with_negative_train_test/test_all.csv',
	#				'full_data_v2/so_user_user_graph.csv', 
	#				'full_data_v2/so_item_graph.csv', 73108, 2024219, 34631, 
	#				'full_data_v2/SO_with_negative_train_test/FM_train_all.txt', 
	#				'full_data_v2/SO_with_negative_train_test/FM_test_all.txt')




	#generateFFMDataset('full_data_v2/SO_with_negative_train_test/train_all.csv',
	#				'full_data_v2/SO_with_negative_train_test/test_all.csv',
	#				'full_data_v2/so_user_user_graph.csv', 
	#				'full_data_v2/so_item_graph.csv', 73108, 2024219, 34631,  
	#				'full_data_v2/SO_with_negative_train_test/FFM_train_all.txt', 
	#				'full_data_v2/SO_with_negative_train_test/FFM_test_all.txt')

	#mergefiles('full_data_v2/SO_with_negative_train_test/train_all','full_data_v2/SO_with_negative_train_test/train_all.csv')
	#mergefiles('full_data_v2/SO_with_negative_train_test/test_all','full_data_v2/SO_with_negative_train_test/test_all.csv')
	#mergefiles('full_data_v2/SO_with_negative_train_test/train_answer','full_data_v2/SO_with_negative_train_test/train_answer.csv')
	#mergefiles('full_data_v2/SO_with_negative_train_test/test_answer','full_data_v2/SO_with_negative_train_test/test_answer.csv')
	#mergefiles('full_data_v2/SO_with_negative_train_test/train_favorite','full_data_v2/SO_with_negative_train_test/train_favorite.csv')
	#mergefiles('full_data_v2/SO_with_negative_train_test/test_favorite','full_data_v2/SO_with_negative_train_test/test_favorite.csv')

	#generateFMTrainingTestingWithNegSampleGithubWithPartition('full_data_v2/gh_user_items.csv', 0.8, 10,79628, 0, 8000)
	#generateFMTrainingTestingWithNegSampleGithubWithPartition('full_data_v2/gh_user_items.csv', 0.8, 10,79628, 8000, 16000)
	#generateFMTrainingTestingWithNegSampleGithubWithPartition('full_data_v2/gh_user_items.csv', 0.8, 10,79628, 16000, 24000)
	#generateFMTrainingTestingWithNegSampleGithubWithPartition('full_data_v2/gh_user_items.csv', 0.8, 10,79628, 24000, 32000)
	#generateFMTrainingTestingWithNegSampleGithubWithPartition('full_data_v2/gh_user_items.csv', 0.8, 10,79628, 32000, 40000)
	#generateFMTrainingTestingWithNegSampleGithubWithPartition('full_data_v2/gh_user_items.csv', 0.8, 10,79628, 40000, 48000)
	#generateFMTrainingTestingWithNegSampleGithubWithPartition('full_data_v2/gh_user_items.csv', 0.8, 10,79628, 48000, 56000)
	#generateFMTrainingTestingWithNegSampleGithubWithPartition('full_data_v2/gh_user_items.csv', 0.8, 10,79628, 56000, 64000)
	#generateFMTrainingTestingWithNegSampleGithubWithPartition('full_data_v2/gh_user_items.csv', 0.8, 10,79628, 64000, 72000)
	#generateFMTrainingTestingWithNegSampleGithubWithPartition('full_data_v2/gh_user_items.csv', 0.8, 10,79628, 72000, 79628)
	#testSparseMatrix()
	#testSigmoid()

	#generateTrainTestByUser('full_data_v5/gh/gh_user_items.csv', 0.8, 33453)
	#testnpzero()
	#setRange()
	countBothActivities('full_data_v5/gh/gh _user_items.csv')



