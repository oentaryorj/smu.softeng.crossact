__author__ = 'Wei Xie'
__email__ = 'linegroup3@gmail.com'
__affiliation__ = 'Living Analytics Research Centre, Singapore Management University'
__website__ = 'http://mysmu.edu/phdis2012/wei.xie.2012'

import threading
import numpy as np
import tensorflow as tf
import time
import exp_config

'''
A simple implementation of factoid embedding which requires less parameter updates.
'''


def embed(triplets, attribute_embeddings, testing_ids=None, bias=False):
    # parameters
    BATCH_SIZE = eval(exp_config.get('triplet_embedding', 'batch_size'))
    USER_DIM = eval(exp_config.get('triplet_embedding', 'user_dim'))
    NCE_SAM_NUM = eval(exp_config.get('triplet_embedding', 'nce_sampling'))
    SNAPSHOT_FLAG = eval(exp_config.get('triplet_embedding', 'snapshot'))
    SNAPSHOT_GAP = eval(exp_config.get('triplet_embedding', 'snapshot_gap'))
    LEARNING_RATE_FOLLOW = eval(exp_config.get('triplet_embedding', 'learning_rate_f'))
    LEARNING_RATE_ATTRIBUTE = eval(exp_config.get('triplet_embedding', 'learning_rate_a'))
    DEBUG_FLAG = eval(exp_config.get('debug', 'flag'))

    # process triplets
    net_degrees = dict()
    max_user_id = 0
    triplets_attribute = list()
    triplets_follow = list()
    predicates = set()

    for trip in triplets:
        s_, p_, o_ = trip
        predicates.add(p_)
        max_user_id = max(max_user_id, s_)

        if p_ == 'a':
            triplets_attribute.append((s_, o_))

        if p_ == 'f':
            triplets_follow.append((s_, o_))
            max_user_id = max(max_user_id, o_)
            if s_ in net_degrees:
                net_degrees[s_] += 1
            else:
                net_degrees[s_] = 1

    num_users = max_user_id + 1
    print ('num_users', num_users)
    print ('predicates', predicates)

    triplets_attribute = np.array(triplets_attribute, dtype=np.int32)
    triplets_follow = np.array(triplets_follow, dtype=np.int32)

    n_triplets_follow = len(triplets_follow)
    n_triplets_attribute = len(triplets_attribute)

    def follow_data_generator():
        follow_batch_start_id = 0
        follow_permutation = np.random.permutation(n_triplets_follow)

        while True:
            if follow_batch_start_id + BATCH_SIZE > n_triplets_follow:
                follow_batch_start_id = 0
                follow_permutation = np.random.permutation(n_triplets_follow)

            choice = follow_permutation[follow_batch_start_id: follow_batch_start_id + BATCH_SIZE]
            selected_triplets_follow = triplets_follow[choice, :]
            _source, _target = np.expand_dims(selected_triplets_follow[:, 0], axis=1), selected_triplets_follow[:, 1]

            follow_batch_start_id += BATCH_SIZE

            yield _source, _target

    def attribute_data_generator():
        attribute_batch_start_id = 0
        attribute_permutation = np.random.permutation(n_triplets_attribute)

        while True:
            if attribute_batch_start_id + BATCH_SIZE > n_triplets_attribute:
                attribute_batch_start_id = 0
                attribute_permutation = np.random.permutation(n_triplets_attribute)
            choice = attribute_permutation[attribute_batch_start_id: attribute_batch_start_id + BATCH_SIZE]
            selected_triplets_attribute = triplets_attribute[choice, :]

            attribute_s = selected_triplets_attribute[:, 0:1]
            attribute_s = np.concatenate((attribute_s, np.arange(BATCH_SIZE).reshape(BATCH_SIZE, 1)), axis=1)
            attribute_o = selected_triplets_attribute[:, 1]

            attribute_batch_start_id += BATCH_SIZE

            yield attribute_s, attribute_o

    net_probs = map(lambda x: net_degrees[x] ** 0.75 if x in net_degrees else 0., range(num_users))
    net_probs = list(net_probs)
    net_probs /= np.sum(net_probs)

    P_limit = 2.

    # build model
    graph = tf.Graph()

    with graph.as_default():
        with tf.device('/gpu:0'):
            follow_ds = tf.data.Dataset.from_generator(follow_data_generator,
                                                       (tf.int32, tf.int32),
                                                       ([BATCH_SIZE, 1], [BATCH_SIZE])).prefetch(3).repeat()

            follow_itr = follow_ds.make_one_shot_iterator()
            trip_follow_source, trip_follow_target = follow_itr.get_next()

            attribute_ds = tf.data.Dataset.from_generator(attribute_data_generator,
                                                          (tf.int32, tf.int32),
                                                          ([BATCH_SIZE, 2], [BATCH_SIZE])).prefetch(3).repeat()

            attribute_itr = attribute_ds.make_one_shot_iterator()
            trip_attribute_subject, trip_attribute_object = attribute_itr.get_next()

            user_embeddings = tf.Variable(tf.truncated_normal([num_users, USER_DIM], stddev=1e-2))

            if testing_ids is not None:
                normalized_user_embeddings = tf.nn.l2_normalize(user_embeddings, dim=1)
                testing_ids_ph = tf.placeholder(tf.int32, shape=[len(testing_ids)])
                dist = tf.tensordot(tf.nn.embedding_lookup(normalized_user_embeddings, testing_ids_ph),
                                    normalized_user_embeddings, axes=[[1], [1]])

            P_follow = tf.Variable(tf.truncated_normal([USER_DIM, USER_DIM], stddev=1.0 / USER_DIM))
            P_follow_norm = tf.norm(P_follow)

            update_P = tf.assign(P_follow, P_limit * tf.nn.l2_normalize(P_follow, [0, 1])).op

            bias_var = tf.Variable(tf.zeros(USER_DIM))
            bias_norm = tf.norm(bias_var)

            f_loss = 0
            a_loss = 0

            for p_ in predicates:
                if p_ == 'f':
                    target = tf.tensordot(tf.nn.embedding_lookup(user_embeddings, trip_follow_target), P_follow,
                                          [[1], [1]])
                    if bias:
                        target += bias_var

                    f_loss += tf.reduce_mean(tf.nn.nce_loss(user_embeddings, tf.zeros(num_users), trip_follow_source,
                                                            target, NCE_SAM_NUM, num_users, num_true=1,
                                                            sampled_values=(
                                                                np.random.choice(num_users, NCE_SAM_NUM, False,
                                                                                 net_probs),
                                                                tf.ones(BATCH_SIZE, dtype=tf.float32),
                                                                tf.ones(NCE_SAM_NUM, dtype=tf.float32))
                                                            ))

                if p_ == 'a':
                    attr_embeddings = tf.nn.embedding_lookup(attribute_embeddings, trip_attribute_object)

                    dot = tf.tensordot(user_embeddings, attr_embeddings, [[1], [1]])
                    softm = tf.nn.softmax(dot, dim=0)
                    softm = tf.gather_nd(softm, trip_attribute_subject)
                    a_loss -= tf.reduce_mean(tf.log(softm))

            f_global_step = tf.Variable(0, trainable=False)
            a_global_step = tf.Variable(0, trainable=False)
            f_learning_rate = tf.train.exponential_decay(LEARNING_RATE_FOLLOW, f_global_step, 10000, 0.96,
                                                         staircase=True)
            a_learning_rate = tf.train.exponential_decay(LEARNING_RATE_ATTRIBUTE, a_global_step, 10000, 0.96,
                                                         staircase=True)
            if f_loss != 0:
                f_optimizer = tf.train.GradientDescentOptimizer(f_learning_rate).minimize(f_loss,
                                                                                          var_list=[user_embeddings])
                f_optimizer_P = tf.train.GradientDescentOptimizer(1e-4).minimize(f_loss, var_list=[P_follow, bias_var])
            if a_loss != 0:
                a_optimizer = tf.train.GradientDescentOptimizer(a_learning_rate).minimize(a_loss)

    with tf.Session(graph=graph,
                    config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as session:

        tf.global_variables_initializer().run()

        warm_up_iter = eval(exp_config.get('triplet_embedding', 'warm_up_iter')) * len(triplets) // BATCH_SIZE
        n_iter = eval(exp_config.get('triplet_embedding', 'n_iter')) * len(triplets) // BATCH_SIZE

        ue_s = list()
        dist_s = list()

        n_threads = 5

        def loop():
            for i in range(n_iter // n_threads):
                if a_loss != 0:
                    start_time = time.time()
                    _, a_loss_val = session.run([a_optimizer, a_loss])

                    if DEBUG_FLAG and np.random.rand() < 0.001:
                        print (i, 'a_loss_val', a_loss_val, time.time() - start_time)

                if i >= warm_up_iter and f_loss != 0:
                    # follow
                    start_time = time.time()

                    if i % 5 == 0:
                        _, f_loss_val, P_norm_, bias_norm_ = session.run(
                            [f_optimizer_P, f_loss, P_follow_norm, bias_norm])

                        if P_norm_ > P_limit:
                            session.run(update_P)
                    else:
                        _, f_loss_val = session.run([f_optimizer, f_loss])

                    if DEBUG_FLAG and np.random.rand() < 0.001:
                        print (i, 'f_loss_val', f_loss_val, time.time() - start_time)

        threads = [threading.Thread(target=loop) for _ in range(n_threads)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        ue = user_embeddings.eval(session)
        dist_values = None if testing_ids is None else session.run(dist, feed_dict={testing_ids_ph: testing_ids})
        ue_s.append(ue)
        dist_s.append(dist_values)

    return ue_s, dist_s, (0, 0, (0, 0))


# for testing
def attribute_embed(triplets, attribute_embeddings, testing_ids):
    # process triplets
    max_user_id = 0
    aem = dict()

    for trip in triplets:
        s_, p_, o_ = trip
        max_user_id = max(max_user_id, s_)

        if p_ == 'a':
            aem[s_] = attribute_embeddings[o_]

        if p_ == 'f':
            max_user_id = max(max_user_id, o_)

    num_users = max_user_id + 1
    print ('num_users', num_users)

    user_ebd = np.random.normal(0., 1., [num_users, attribute_embeddings.shape[1]])

    for uid, ebd in aem.items():
        user_ebd[uid, :] = ebd

    graph = tf.Graph()

    with graph.as_default():
        with tf.device('/gpu:0'):
            user_embeddings = tf.constant(user_ebd)
            normalized_user_embeddings = tf.nn.l2_normalize(user_embeddings, dim=1)
            testing_ids_ph = tf.placeholder(tf.int32, shape=[len(testing_ids)])
            dist = tf.tensordot(tf.nn.embedding_lookup(normalized_user_embeddings, testing_ids_ph),
                                normalized_user_embeddings, axes=[[1], [1]])

    with tf.Session(graph=graph,
                    config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as session:
        dist_values = session.run(dist, feed_dict={testing_ids_ph: testing_ids})

    return [user_ebd], [dist_values]
