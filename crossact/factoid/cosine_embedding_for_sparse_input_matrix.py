__author__ = 'Wei Xie'
__email__ = 'linegroup3@gmail.com'
__affiliation__ = 'Living Analytics Research Centre, Singapore Management University'
__website__ = 'http://mysmu.edu/phdis2012/wei.xie.2012'

import numpy as np
import tensorflow as tf
import time
import exp_config


def embed(X, n_dim, n_iter):
    '''
    X is a sparse matrix, (CSR)
    '''
    DEBUG_FLAG = eval(exp_config.get('debug', 'flag'))
    BATCH_SIZE = eval(exp_config.get('cosine_embedding', 'batch_size'))
    LEARNING_RATE = eval(exp_config.get('cosine_embedding', 'learning_rate'))

    # checking
    if DEBUG_FLAG:
        print ('X', X.shape)
        print ('X[0,0]', X[0, 0])
        print ('X[1,1]', X[1, 1])
        print ('min X', np.min(X))
        print ('max X', np.max(X))

    # focused on neighbour pairs
    start_time = time.time()

    pairs_i, pairs_j = X.nonzero()
    x_list = X.data

    print ('len(x_list)', len(x_list), 'len(pairs_i)', len(pairs_i))
    assert len(x_list) == len(pairs_i)

    pairs_i = np.array(pairs_i, dtype=np.int32)
    pairs_j = np.array(pairs_j, dtype=np.int32)
    x_list = np.array(x_list, dtype=np.float32)

    print ('take ', time.time() - start_time, ' seconds')
    print ('pairs_i', len(pairs_i), 'pairs_j', len(pairs_j), 'x_list', len(x_list))

    n_W = len(x_list)

    if DEBUG_FLAG:
        print ('n_pairs', n_W, (n_W + 0.) / (X.shape[0] ** 2))

    batch_start_id = 0
    permutation = np.random.permutation(n_W)

    # building model
    graph = tf.Graph()

    with graph.as_default():
        global_step = tf.Variable(0, trainable=False)

        input_pairs_i = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        input_pairs_j = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        input_xs = tf.placeholder(tf.float32, shape=[BATCH_SIZE])

        attribute_embeddings = tf.Variable(
            tf.random_uniform([X.shape[0], n_dim], -2.0 / np.sqrt(n_dim), 2.0 / np.sqrt(n_dim)),
            name='attribute_embeddings')

        mt = tf.nn.embedding_lookup(attribute_embeddings, input_pairs_i) * tf.nn.embedding_lookup(attribute_embeddings,
                                                                                                  input_pairs_j)
        mt = tf.reduce_sum(mt, axis=1)
        loss = tf.nn.l2_loss(mt - input_xs, name='loss') * 2 / BATCH_SIZE

        learning_rate = tf.train.exponential_decay(LEARNING_RATE * np.sqrt(n_dim) / 2, global_step,
                                                   25 * n_W / BATCH_SIZE, 0.96, staircase=True)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=False,
                                                       allow_soft_placement=True)) as session:

        tf.global_variables_initializer().run()

        real_iters = n_iter * n_W // BATCH_SIZE
        print ('real iters', real_iters)

        for i in range(real_iters):
            if batch_start_id + BATCH_SIZE > n_W:
                batch_start_id = 0
                permutation = np.random.permutation(n_W)

            choice = permutation[batch_start_id: batch_start_id + BATCH_SIZE]

            _input_pairs_i, _input_pairs_j, _input_xs = pairs_i[choice], pairs_j[choice], x_list[choice]
            batch_start_id += BATCH_SIZE

            start_time = time.time()
            _, loss_val = session.run([optimizer, loss], feed_dict={input_pairs_i: _input_pairs_i,
                                                                    input_pairs_j: _input_pairs_j, input_xs: _input_xs})
            if DEBUG_FLAG and np.random.rand() < 0.001 or i == real_iters - 1:
                print (i, i / (n_W / BATCH_SIZE), 'f_loss_val', loss_val, time.time() - start_time, 's')

        attribute_embeddings = attribute_embeddings.eval()
    return attribute_embeddings


def embed2(X, n_dim, n_iter):
    '''
    X is a sparse matrix, (CSR), ignore the diagonal elements.
    '''
    DEBUG_FLAG = eval(exp_config.get('debug', 'flag'))
    BATCH_SIZE = eval(exp_config.get('cosine_embedding', 'batch_size'))

    # checking
    if DEBUG_FLAG:
        print ('X', X.shape)
        print ('X[0,0]', X[0, 0])
        print ('X[1,1]', X[1, 1])
        print ('min X', np.min(X))
        print ('max X', np.max(X))

    # focused on neighbour pairs
    start_time = time.time()
    pairs_i, pairs_j = X.nonzero()
    x_list = X.data

    assert len(x_list) == len(pairs_i)

    elements = list()

    for i, j, x in zip(pairs_i, pairs_j, x_list):
        if i != j:
            elements.append((i, j, x))

    pairs_i, pairs_j, x_list = zip(*elements)
    pairs_i = np.array(pairs_i, dtype=np.int32)
    pairs_j = np.array(pairs_j, dtype=np.int32)
    x_list = np.array(x_list, dtype=np.float32)

    print ('take ', time.time() - start_time, ' seconds')
    print ('pairs_i', len(pairs_i), 'pairs_j', len(pairs_j), 'x_list', len(x_list))

    n_W = len(x_list)

    if DEBUG_FLAG:
        print ('n_pairs', n_W, (n_W + 0.) / (X.shape[0] ** 2))

    batch_start_id = 0
    permutation = np.random.permutation(n_W)

    # building model
    graph = tf.Graph()

    with graph.as_default():
        global_step = tf.Variable(0, trainable=False)

        input_pairs_i = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        input_pairs_j = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
        input_xs = tf.placeholder(tf.float32, shape=[BATCH_SIZE])

        attribute_embeddings = tf.Variable(tf.truncated_normal([X.shape[0], n_dim], stddev=1e-5))
        mt = tf.nn.embedding_lookup(attribute_embeddings, input_pairs_i) * tf.nn.embedding_lookup(attribute_embeddings,
                                                                                                  input_pairs_j)
        mt = tf.reduce_sum(mt, axis=1)
        loss = tf.nn.l2_loss(mt - input_xs, name='loss') * 2 / BATCH_SIZE

        learning_rate = tf.train.exponential_decay(10., global_step, 25 * n_W / BATCH_SIZE, 0.96, staircase=True)  # !!!
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=False,
                                                       allow_soft_placement=True)) as session:
        tf.global_variables_initializer().run()
        real_iters = max(n_iter * n_W / BATCH_SIZE, 10000)

        print ('real iters', real_iters)

        for i in range(real_iters):
            if batch_start_id + BATCH_SIZE > n_W:
                batch_start_id = 0
                permutation = np.random.permutation(n_W)

            choice = permutation[batch_start_id: batch_start_id + BATCH_SIZE]
            _input_pairs_i, _input_pairs_j, _input_xs = pairs_i[choice], pairs_j[choice], x_list[choice]
            batch_start_id += BATCH_SIZE

            start_time = time.time()
            _, loss_val = session.run([optimizer, loss], feed_dict={input_pairs_i: _input_pairs_i,
                                                                    input_pairs_j: _input_pairs_j, input_xs: _input_xs})

            if DEBUG_FLAG and np.random.rand() < 0.001 or i == real_iters - 1:
                print (i, i / (n_W / BATCH_SIZE), 'f_loss_val', loss_val, time.time() - start_time, 's')

        attribute_embeddings = attribute_embeddings.eval()

    return attribute_embeddings
