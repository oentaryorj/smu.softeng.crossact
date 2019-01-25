__author__ = 'Wei Xie'
__email__ = 'linegroup3@gmail.com'
__affiliation__ = 'Living Analytics Research Centre, Singapore Management University'
__website__ = 'http://mysmu.edu/phdis2012/wei.xie.2012'


import codecs

import json

import os

import sys

import time

import jellyfish

import numpy as np

import scipy.sparse
from scipy.sparse import csr_matrix

from collections import defaultdict

import exp_config

settings_f = './settings.cnf'

print ('settings:', settings_f)

exp_config.read(settings_f)

import name2sim
import image2sim
import factoid_embedding

import argparse
flag_skip_network = False

PATH = exp_config.get('data', 'path')

SOURCE_PREFIX = exp_config.get('data', 'source_prefix')
TARGET_PREFIX = exp_config.get('data', 'target_prefix')
SOURCE_COL = int(exp_config.get('data', 'source_col'))
TARGET_COL = int(exp_config.get('data', 'target_col'))

USER_DIM = exp_config.get('triplet_embedding', 'user_dim')
ITER_NUM = exp_config.get('triplet_embedding', 'n_iter')
WARM_UP_NUM = exp_config.get('triplet_embedding', 'warm_up_iter')
NCE_NUM = exp_config.get('triplet_embedding', 'nce_sampling')

SUPERVISED_FLAG = eval(exp_config.get('triplet_embedding', 'supervised'))
BIAS_FLAG = eval(exp_config.get('triplet_embedding', 'bias'))


screen_name_exist = eval(exp_config.get('predicate_name', 'screen_name_exist'))
flag_preprocess = eval(exp_config.get('predicate_name', 'preprocess'))
flag_concatenate = eval(exp_config.get('predicate_name', 'concatenate'))

image_exist = eval(exp_config.get('predicate_image', 'exist'))

pass_embedding = eval(exp_config.get('cosine_embedding', 'pass'))

OUTPUT = exp_config.get('data', 'output')
if OUTPUT is None:
    OUTPUT = 'results'
    if SUPERVISED_FLAG:
        OUTPUT += '_supervised'
    else:
        OUTPUT += '_unsupervised'
    if BIAS_FLAG:
        OUTPUT += '_biased'
    else:
        OUTPUT += '_unbiased'
    OUTPUT += '_d' + USER_DIM
    OUTPUT += '_i' + ITER_NUM
    OUTPUT += '_w' + WARM_UP_NUM
    OUTPUT += '_ns' + NCE_NUM

    OUTPUT += '_' + exp_config.get('predicate_name', 'method')

    if screen_name_exist:
        OUTPUT += '_s'

    if image_exist:
        OUTPUT += '_m'

OUTPUT = SOURCE_PREFIX + '2' + TARGET_PREFIX + '_' + OUTPUT
OUTPUT += '/'


def preprocess_fun(s):
    return s.encode('ascii', 'ignore').decode('utf-8').replace('.', '').replace('_', '').replace(' ', '')


source_users = list()
target_users = list()

source_user_names = dict()

f = codecs.open(PATH + SOURCE_PREFIX + '_user_names.txt', 'r', 'utf-8')
for line in f:
    terms = line[:-1].split('\t')
    if terms[1] == 'None':
        un = None
    else:
        un = terms[1].lower().strip()
        if flag_preprocess:
            un = preprocess_fun(un)
            if len(un) == 0:
                un = None
    source_user_names[terms[0]] = un
    source_users.append(terms[0])
f.close()

print ('source_user_names', len(source_user_names))

source_screen_names = None
if screen_name_exist:
    source_screen_names = dict()

    f = codecs.open(PATH + SOURCE_PREFIX + '_screen_names.txt', 'r', 'utf-8')
    for line in f:
        terms = line[:-1].split('\t')
        if terms[1] == 'None':
            sn = None
        else:
            sn = terms[1].lower().strip()
            if flag_preprocess:
                sn = preprocess_fun(sn)
                if len(sn) == 0:
                    sn = None
        source_screen_names[terms[0]] = sn
    f.close()

    print ('source_screen_names', len(source_screen_names))


target_user_names = dict()

f = codecs.open(PATH + TARGET_PREFIX + '_user_names.txt', 'r', 'utf-8')
for line in f:
    terms = line[:-1].split('\t')
    if terms[1] == 'None':
        un = None
    else:
        un = terms[1].lower().strip()
        if flag_preprocess:
            un = preprocess_fun(un)
            if len(un) == 0:
                un = None
    target_user_names[terms[0]] = un
    target_users.append(terms[0])
f.close()

print ('target_user_names', len(target_user_names))

target_screen_names = None
if screen_name_exist:
    target_screen_names = dict()

    f = codecs.open(PATH + TARGET_PREFIX + '_screen_names.txt', 'r', 'utf-8')
    for line in f:
        terms = line[:-1].split('\t')
        if terms[1] == 'None':
            sn = None
        else:
            sn = terms[1].lower().strip()
            if flag_preprocess:
                sn = preprocess_fun(sn)
                if len(sn) == 0:
                    sn = None
        target_screen_names[terms[0]] = sn
    f.close()

    print ('target_screen_names', len(target_screen_names))


print ('source_users', len(source_users), 'target_users', len(target_users))

# attribute step
attr_id = 0
sc2aid = {}
aid2sc = {}
tg2aid = {}
aid2tg = {}
for user in source_users:
    sc2aid[user] = attr_id
    aid2sc[attr_id] = user
    attr_id += 1
for user in target_users:
    tg2aid[user] = attr_id
    aid2tg[attr_id] = user
    attr_id += 1

num_attr = attr_id

print('num_attr', num_attr)
np.save('id_process.npy', [sc2aid, aid2sc, tg2aid, aid2tg])


if not flag_concatenate:
    
    assert screen_name_exist == False
    # for user names
    user_name2eid, user_name_sim = name2sim.name2sim(
                                        list(map(lambda x: source_user_names[aid2sc[x]] if x in aid2sc \
                                                 else target_user_names[aid2tg[x]],
                                                 range(num_attr))), 
                                        'user_name', pass_embedding)
    #np.save('temp_un', [user_name2eid, user_name_sim])  # debugging
    
    max_sim = user_name_sim

else:
    source_concat_names = dict()
    for uid in source_users:
        cn = ''
        if source_user_names[uid]:
            cn += source_user_names[uid]
        if screen_name_exist:
            if source_screen_names[uid]:
                cn += source_screen_names[uid]
        if len(cn) == 0:
            cn = None
        source_concat_names[uid] = cn

    target_concat_names = dict()
    for uid in target_users:
        cn = ''
        if target_user_names[uid]:
            cn += target_user_names[uid]
        if screen_name_exist:
            if target_screen_names[uid]:
                cn += target_screen_names[uid]
        if len(cn) == 0:
            cn = None
        target_concat_names[uid] = cn
    
    concat_name2eid, concat_name_sim = name2sim.name2sim(
                                        list(map(lambda x: source_concat_names[aid2sc[x]] if x in aid2sc \
                                                 else target_concat_names[aid2tg[x]],
                                                 range(num_attr))),
                                'concat_name', pass_embedding)
    #np.save('temp_concat', [concat_name2eid, concat_name_sim])  # debugging

    max_sim = concat_name_sim
    


if not pass_embedding:
    n_dim = eval(exp_config.get('cosine_embedding', 'n_dim'))
    n_iter = eval(exp_config.get('cosine_embedding', 'n_iter'))
    n_gpu = eval(exp_config.get('cosine_embedding', 'n_gpu'))
    
    import cosine_embedding_for_sparse_input_files
    attribute_embeddings = cosine_embedding_for_sparse_input_files.embed(max_sim, num_attr, n_dim, n_iter, n_gpu)
    np.save('name_embedding.npy', attribute_embeddings)  # for debugging
    #attribute_embeddings = np.load('attribute_embeddings.npy')  # for debugging
else:
    pass#scipy.sparse.save_npz('max_sim.npz', max_sim) # for debugging
    
    
def unit_vector(vector):
    s = np.linalg.norm(vector)
    if s > 1e-8:
        return vector / s
    else:
        return vector


def sim(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.dot(v1_u, v2_u)


if not os.path.isdir(PATH + OUTPUT):
    os.makedirs(PATH + OUTPUT)

# training and testing
TRAIN_TEST_PATH = eval(exp_config.get('data', 'train_test_paths'))
testing_sample_rate = 1.1

# users in ground truth
if not SUPERVISED_FLAG:
    source_users_in_gt = list()
    f = open(PATH + '/ground_truth.txt', 'r')
    for line in f:
        terms = line.split()
        sc_id = terms[SOURCE_COL]
        source_users_in_gt.append(sc_id)
    f.close()

all_res = list()
name_dis_list = list()
for tt_path in TRAIN_TEST_PATH:
    print (tt_path)

    source_users_in_training = list()
    target_users_in_training = list()

    training_map = dict()
    f = open(PATH + tt_path + '/training.txt', 'r')
    for line in f:
        terms = line.split()
        sc_id = terms[SOURCE_COL]
        tg_id = terms[TARGET_COL]
        training_map[sc_id] = tg_id
        source_users_in_training.append(sc_id)
        target_users_in_training.append(tg_id)
    f.close()

    source_users_in_testing = list()
    target_users_in_testing = list()
    testing_map = dict()
    f = open(PATH + tt_path + '/testing.txt', 'r')
    for line in f:
        terms = line.split()
        sc_id = terms[SOURCE_COL]
        tg_id = terms[TARGET_COL]
        testing_map[sc_id] = tg_id
        source_users_in_testing.append(sc_id)
        target_users_in_testing.append(tg_id)
    f.close()

    # constructing triples
    triplets = list()

    sc2uid = dict()
    tg2uid = dict()

    uid = 0
    # for matched users
    if SUPERVISED_FLAG:
        for source_id, target_id in training_map.items():
            sc2uid[source_id] = uid
            tg2uid[target_id] = uid
            uid += 1

    unmatched_uid = uid
    print ('unmatched_uid', unmatched_uid)

    for source_id in source_users:
        if SUPERVISED_FLAG:
            if source_id in source_users_in_training:
                continue

        sc2uid[source_id] = uid
        triplets.append((uid, 'a', sc2aid[source_id]))

        uid += 1

    for target_id in target_users:
        if SUPERVISED_FLAG:
            if target_id in target_users_in_training:
                continue

        tg2uid[target_id] = uid
        triplets.append((uid, 'a', tg2aid[target_id]))

        uid += 1

    # for source_links
    f = open(PATH + SOURCE_PREFIX + '_sub_network.txt', 'r')
    for line in f:
        source_id1, source_id2 = line.split()
        if source_id1 in sc2uid and source_id2 in sc2uid:
            uid1 = sc2uid[source_id1]
            uid2 = sc2uid[source_id2]
            triplets.append((uid1, 'f', uid2))
    f.close()

    # for target_links
    f = open(PATH + TARGET_PREFIX + '_sub_network.txt', 'r')
    for line in f:
        target_id1, target_id2 = line[:-1].split('\t')
        if target_id1 in tg2uid and target_id2 in tg2uid:
            uid1 = tg2uid[target_id1]
            uid2 = tg2uid[target_id2]
            triplets.append((uid1, 'f', uid2))
    f.close()

    print ('triplets', len(triplets))
    
    

    # triplet embedding
    if not SUPERVISED_FLAG:
        tt_path = 'unsupervised_'

    testing_ids = None
    if SUPERVISED_FLAG:
        testing_ids = list()
        for source_user in source_users_in_testing:
            testing_ids.append(sc2uid[source_user])
        testing_ids = np.array(testing_ids, dtype=np.int32)
    else:
        testing_ids = list()
        for source_user in source_users_in_gt:
            testing_ids.append(sc2uid[source_user])
        testing_ids = np.array(testing_ids, dtype=np.int32)

    if not pass_embedding:
        if flag_skip_network:
            ue_s, dist_s = factoid_embedding.attribute_embed(triplets, attribute_embeddings, testing_ids)#for debugging
        else:
            with open(PATH + OUTPUT + tt_path + 'sc2uid.txt', 'w') as f_sc2uid:
                json.dump(sc2uid, f_sc2uid)
            with open(PATH + OUTPUT + tt_path + 'tg2uid.txt', 'w') as f_tg2uid:
                json.dump(tg2uid, f_tg2uid)
            
            if os.path.isfile(PATH + OUTPUT + tt_path + 'user_embedding_result.npy'):
                ue_s, dist_s, (f_losses, usn_losses, P_norms) = np.load(PATH + OUTPUT + tt_path + 'user_embedding_result.npy')
                np.save('factoid_embedding.npy', ue_s[0])
                print (PATH + OUTPUT + tt_path + 'user_embedding_result.npy', 'loaded')
            else:
                fe_current_time = time.time()
                ue_s, dist_s, (f_losses, usn_losses, P_norms) = factoid_embedding.embed(triplets, attribute_embeddings,
                                                                                                     testing_ids, BIAS_FLAG)
                print ('Factoid Embedding used', time.time() - fe_current_time, 's')
                np.save('factoid_embedding.npy', ue_s[0])
                np.save(PATH + OUTPUT + tt_path + 'user_embedding_result.npy',
                        [ue_s, dist_s, (f_losses, usn_losses, P_norms)])
       
    target_users_in_training = set(target_users_in_training)
    start_time = time.time()
    _id = 0
    for source_user in source_users_in_testing:
        snapshot_res = list()
        if not SUPERVISED_FLAG:
            source_user_id_in_gt = source_users_in_gt.index(source_user)
        
        if pass_embedding:
            snapshot_len = 1
        else:
            snapshot_len = len(ue_s)
            
        for snapshot_i in range(snapshot_len):
            if np.random.rand() >= testing_sample_rate:
                break
            testing_data = list()
            target_candidates = list()
            
            if not pass_embedding:
                ue = ue_s[snapshot_i]
                dist = dist_s[snapshot_i]
            else:
                dist = max_sim
                
            mrr_index = 0
            for target_user in target_users:
                if target_user in target_users_in_training:
                    continue

                if pass_embedding:
                    dis = dist[sc2aid[source_user], tg2aid[target_user]]
                else:
                    dis = dist[_id, tg2uid[target_user]] if SUPERVISED_FLAG else dist[source_user_id_in_gt, tg2uid[target_user]]
                    #dis = sim(ue[sc2uid[source_user]], ue[tg2uid[target_user]])  #for debugging
                 
                mrr_index += 1
                if dis > 0.:
                    target_candidates.append(target_user)
                    testing_data.append(dis)

            testing_data = np.array(testing_data)

            predict = testing_data

            scores = list(zip(predict.tolist(), target_candidates))

            scores.sort(reverse=True)

            # for mrr
            try:
                _, target_candidates = zip(*scores)
                mrr_index = target_candidates.index(testing_map[source_user])
            except:
                pass
            mrr = 1./(mrr_index + 1)

            # for top_list

            top_k = 30
            res = np.zeros(top_k+1)
            res[mrr_index:] = 1
            res[-1] = mrr

            snapshot_res.append(res)

        _id += 1

        sys.stdout.write("\r%d%%" % (100*_id//len(source_users_in_testing)))
        sys.stdout.flush()

        if len(snapshot_res) > 0:
            all_res.append(snapshot_res)

    print (time.time() - start_time, 'seconds used.')

print ('all_res', len(all_res))
all_res = np.array(all_res)

np.save(PATH + OUTPUT + 'all_res.npy', all_res)

precision = np.mean(all_res, axis=0) if len(all_res) > 0 else 0.
print (precision)

f = open(PATH + OUTPUT + 'report.txt', 'w')
f.write(str(precision))
f.write('\n')
f.write('########################################')
f.write('\n')

f.write(exp_config.get_info())

f.close()
