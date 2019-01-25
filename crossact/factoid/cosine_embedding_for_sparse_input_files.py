__author__ = 'Wei Xie'
__email__ = 'linegroup3@gmail.com'
__affiliation__ = 'Singapore Management University'
__website__ = 'http://mysmu.edu/phdis2012/wei.xie.2012'

import threading
from collections import deque

import tensorflow as tf
import numpy as np
from array import array
import time

from functools import partial

import exp_config

DEBUG_FLAG = eval(exp_config.get('debug', 'flag'))
BATCH_SIZE = eval(exp_config.get('cosine_embedding', 'batch_size'))
LEARNING_RATE = eval(exp_config.get('cosine_embedding', 'learning_rate'))
PARTITION_PATH = exp_config.get('cosine_embedding', 'partition_path')

output_info = open('cosine_embedding_for_sparse_input_files_output.txt', 'wt')


def _variable(name, shape, dtype, initializer):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _step(i_partition, j_partition, attribute_embeddings_list,
          input_i, input_j, input_s, lr, i_instance=0):
    X = tf.gather(attribute_embeddings_list[i_partition], input_i)
    Y = tf.gather(attribute_embeddings_list[j_partition], input_j)
    e = tf.reduce_sum(tf.multiply(X, Y), axis=-1) - input_s

    if i_partition == j_partition:
        grad_i = lr * tf.multiply(Y, e[:, None])
        grad_j = lr * tf.multiply(X, e[:, None])
        update_i = tf.scatter_sub(attribute_embeddings_list[i_partition], input_i, grad_i).op
        update_j = tf.scatter_sub(attribute_embeddings_list[j_partition], input_j, grad_j).op

        with tf.control_dependencies([update_i, update_j]):
            update_op = tf.no_op(name='update_' + str(i_partition) + '_' + str(j_partition) + '_' + str(i_instance))
    else:
        grad_i = (1.5 * lr) * tf.multiply(Y, e[:, None])
        update_i = tf.scatter_sub(attribute_embeddings_list[i_partition], input_i, grad_i).op

        with tf.control_dependencies([update_i]):
            update_op = tf.no_op(name='update_' + str(i_partition) + '_' + str(j_partition) + '_' + str(i_instance))

    loss = tf.reduce_mean(tf.square(e))
    return update_op, loss


def partition_file(path, n_partition, partition_size, sims):
    ret = dict()
    output_id_i = dict()
    output_id_j = dict()
    output_v = dict()

    for i in range(n_partition):
        for j in range(n_partition):
            output_id_i[(i, j)] = open(path + '/similarity_i_part_' + str(i) + '_' + str(j) + '.dat', 'wb')
            output_id_j[(i, j)] = open(path + '/similarity_j_part_' + str(i) + '_' + str(j) + '.dat', 'wb')
            output_v[(i, j)] = open(path + '/similarity_v_part_' + str(i) + '_' + str(j) + '.dat', 'wb')

            ret[(i, j)] = (path + '/similarity_i_part_' + str(i) + '_' + str(j) + '.dat',
                           path + '/similarity_j_part_' + str(i) + '_' + str(j) + '.dat',
                           path + '/similarity_v_part_' + str(i) + '_' + str(j) + '.dat')

    def write(array_id, array_v):
        array_id = np.array(array_id)
        array_v = np.array(array_v)
        array_i, array_j = array_id[::2], array_id[1::2]

        for i in range(n_partition):
            index_i = np.where(array_i // partition_size == i)[0]
            _array_i, _array_j, _array_v = array_i[index_i], array_j[index_i], array_v[index_i]

            for j in range(n_partition):
                index_j = np.where(_array_j // partition_size == j)[0]
                __array_i, __array_j, __array_v = _array_i[index_j] % partition_size, _array_j[
                    index_j] % partition_size, _array_v[index_j]

                op_id = array('I')
                op_id.fromlist(list(__array_i))
                op_id.tofile(output_id_i[(i, j)])

                op_id = array('I')
                op_id.fromlist(list(__array_j))
                op_id.tofile(output_id_j[(i, j)])

                op_rd = array('f')
                op_rd.fromlist(list(__array_v))
                op_rd.tofile(output_v[(i, j)])

    array_id = array('I')
    array_v = array('f')

    for _, similarity_id_f, similarity_v_f in sims:
        similarity_id = open(similarity_id_f, 'rb')
        similarity_v = open(similarity_v_f, 'rb')

        while True:
            try:
                array_id.fromfile(similarity_id, 2 * BATCH_SIZE - len(array_id))
            except EOFError:
                pass

            try:
                array_v.fromfile(similarity_v, BATCH_SIZE - len(array_v))
            except EOFError:
                pass

            if len(array_v) < BATCH_SIZE:
                break

            write(array_id, array_v)
            array_id = array('I')
            array_v = array('f')

        similarity_id.close()
        similarity_v.close()

    write(array_id, array_v)

    for i in range(n_partition):
        for j in range(n_partition):
            output_id_i[(i, j)].close()
            output_id_j[(i, j)].close()
            output_v[(i, j)].close()

    return ret


def get_from_sim(part_sim, partition_i, partition_j):
    similarity_i_f, similarity_j_f, similarity_v_f = part_sim[(partition_i, partition_j)]

    similarity_i = open(similarity_i_f, 'rb')
    similarity_j = open(similarity_j_f, 'rb')
    similarity_v = open(similarity_v_f, 'rb')

    array_i = array('I')
    array_j = array('I')
    array_v = array('f')

    while True:
        current_len = len(array_v)
        # print ('current_len', current_len) for debugging

        try:
            array_i.fromfile(similarity_i, BATCH_SIZE)
        except EOFError:
            pass

        try:
            array_j.fromfile(similarity_j, BATCH_SIZE)
        except EOFError:
            pass

        try:
            array_v.fromfile(similarity_v, BATCH_SIZE)
        except EOFError:
            pass

        if len(array_v) == current_len:
            break

    similarity_i.close()
    similarity_j.close()
    similarity_v.close()

    array_i = np.array(array_i)
    array_j = np.array(array_j)
    array_v = np.array(array_v)

    return array_i, array_j, array_v


def embed(sm, n_names, n_dim, n_iters, n_gpus=1):
    partition_size = int(np.ceil(n_names / n_gpus))
    n_instances = 1
    part_sim = partition_file(PARTITION_PATH, n_gpus, partition_size, sm)

    print ('partition_size', partition_size)

    graph = tf.Graph()

    with graph.as_default():
        def data_generator(partition_i, partition_j):
            array_i = array('I')
            array_j = array('I')
            array_v = array('f')
            _ct = 0

            while True:
                print('_ct', _ct)
                _ct += 1

                if True:
                    similarity_i_f, similarity_j_f, similarity_v_f = part_sim[(partition_i, partition_j)]
                    similarity_i = open(similarity_i_f, 'rb')
                    similarity_j = open(similarity_j_f, 'rb')
                    similarity_v = open(similarity_v_f, 'rb')

                    while True:
                        try:
                            array_i.fromfile(similarity_i, BATCH_SIZE - len(array_i))
                        except EOFError:
                            pass

                        try:
                            array_j.fromfile(similarity_j, BATCH_SIZE - len(array_j))
                        except EOFError:
                            pass

                        try:
                            array_v.fromfile(similarity_v, BATCH_SIZE - len(array_v))
                        except EOFError:
                            pass

                        if len(array_v) < BATCH_SIZE:
                            break

                        array_i = np.array(array_i)
                        array_j = np.array(array_j)
                        array_v = np.array(array_v)

                        yield array_i, array_j, array_v

                        array_i = array('I')
                        array_j = array('I')
                        array_v = array('f')

                    similarity_i.close()
                    similarity_j.close()
                    similarity_v.close()

        data_mem = dict()
        len_mem = dict()

        for partition_i in range(n_gpus):
            for partition_j in range(n_gpus):
                array_i, array_j, array_v = get_from_sim(part_sim, partition_i, partition_j)
                l = len(array_v)
                len_mem[(partition_i, partition_j)] = l
                data_mem[(partition_i, partition_j)] = (array_i, array_j, array_v, l)

        def data_generator2(partition_i, partition_j):
            array_i, array_j, array_v, l = data_mem[(partition_i, partition_j)]
            print(type(array_i), array_i.shape)
            print(type(array_j), array_j.shape)
            print(type(array_v), array_v.shape)
            print('l=', l)

            _ct = 0
            start = 0

            while True:
                yield array_i[start:start + BATCH_SIZE], array_j[start:start + BATCH_SIZE], array_v[
                                                                                            start:start + BATCH_SIZE]

                start += BATCH_SIZE

                if start >= l:
                    start = 0
                    _ct += 1
                    print('_ct', _ct)

        update_ops = []
        syn_ops = []
        embedding_matrix = dict()

        for i_partition in range(n_gpus):
            with tf.device('/gpu:' + str(i_partition)), tf.variable_scope("scope_gpu_" + str(i_partition)):
                attribute_embeddings_list = list()

                for j_partition in range(n_gpus):
                    attribute_embeddings = _variable('attribute_embeddings_' + str(j_partition),
                                                     [partition_size, n_dim],
                                                     tf.float32,
                                                     tf.random_uniform_initializer(-2.0 / np.sqrt(n_dim),
                                                                                   2.0 / np.sqrt(n_dim))
                                                     )
                    attribute_embeddings_list.append(attribute_embeddings)
                    embedding_matrix[(i_partition, j_partition)] = attribute_embeddings

                for j_partition in range(n_gpus):
                    for instance in range(n_instances):
                        ds = tf.data.Dataset.from_generator(partial(data_generator2, i_partition, j_partition),
                                                            (tf.int32, tf.int32, tf.float32),
                                                            ([None], [None], [None])).prefetch(3).repeat()

                        itr = ds.make_one_shot_iterator()
                        input_i, input_j, input_s = itr.get_next()

                        learning_rate = 5 * 1e-3

                        update_op = _step(i_partition, j_partition, attribute_embeddings_list,
                                          input_i, input_j, input_s, learning_rate, instance)

                        update_ops.append((i_partition, j_partition, instance, update_op))

        print ('update_ops', len(update_ops))

        if n_gpus < 8:
            for j_partition in range(n_gpus):
                for i_partition in range(n_gpus):
                    if i_partition != j_partition:
                        syn_op = tf.assign(embedding_matrix[(i_partition, j_partition)],
                                           embedding_matrix[(j_partition, j_partition)], use_locking=False).op
                        print('op', (j_partition, j_partition), '-->', (i_partition, j_partition))
                        syn_ops.append((i_partition, j_partition, syn_op))

        if n_gpus == 8:
            for i_partition in range(8):
                group_id = i_partition // 4
                _i_partition = i_partition % 4 if i_partition >= 4 else i_partition % 4 + 4

                syn_op = tf.assign(embedding_matrix[(_i_partition, i_partition)],
                                   embedding_matrix[(i_partition, i_partition)], use_locking=False).op
                print('op', (i_partition, i_partition), '-->', (_i_partition, i_partition))
                syn_ops.append(((i_partition, i_partition), (_i_partition, i_partition), syn_op))

                for j in range(4):
                    if i_partition % 4 == j:
                        continue

                    j_partition = j + 4 * group_id

                    syn_op = tf.assign(embedding_matrix[(j_partition, i_partition)],
                                       embedding_matrix[(i_partition, i_partition)], use_locking=False).op

                    print('op', (i_partition, i_partition), '-->', (j_partition, i_partition))

                    syn_ops.append(((i_partition, i_partition), (j_partition, i_partition), syn_op))

                    syn_op = tf.assign(embedding_matrix[(j_partition, _i_partition)],
                                       embedding_matrix[(i_partition, _i_partition)], use_locking=False).op

                    print('op', (i_partition, _i_partition), '-->', (j_partition, _i_partition))

                    syn_ops.append(((i_partition, _i_partition), (j_partition, _i_partition), syn_op))

        def loop(_session, op, part_len, _tid):
            op, loss = op
            i_partition, j_partition, instance = map(int, _tid.split('_'))
            iters = n_iters  # 300 if i_partition == j_partition else 400

            print(i_partition, j_partition, instance, iters)

            one_round = int(np.ceil(part_len / BATCH_SIZE))

            print (_tid, 'one_round', one_round)

            gap = 1
            que_len = one_round // gap

            print ('que_len', que_len)

            que = deque(maxlen=que_len)
            i = 0

            while True:
                if i >= iters * one_round and i % gap == 0:
                    ct = time.time()
                    _, _loss = _session.run([op, loss])
                    dt = time.time() - ct
                    que.append((dt, _loss))

                    if len(que) == que_len:
                        m_dt, m_loss = np.mean(que, axis=0)
                        print(_tid, 'for each mini-batch', m_dt, m_loss)
                        print(_tid, 'for each mini-batch', m_dt, m_loss, file=output_info)
                        break
                else:
                    _session.run(op)

                i += 1

        def syn_loop(_session, coord, op, op_name):
            while not coord.should_stop():
                time.sleep(0.1)

                if np.random.rand() < 0.0001:
                    print(op_name)
                _session.run(op)

        init = tf.global_variables_initializer()

    with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
        sess.run(init)

        threads = [threading.Thread(target=loop, args=(sess, update_op, len_mem[(i_partition, j_partition)],
                                                       str(i_partition) + '_' + str(j_partition) + '_' + str(instance)))
                   for i_partition, j_partition, instance, update_op in update_ops]

        for t in threads:
            t.start()

        coord = tf.train.Coordinator()

        def syn_op_name(i_partition, j_partition):
            return ''.join([str((j_partition, j_partition)), '-->', str((i_partition, j_partition))])

        syn_threads = [
            threading.Thread(target=syn_loop, args=(sess, coord, syn_op, syn_op_name(i_partition, j_partition)))
            for i_partition, j_partition, syn_op in syn_ops]

        for t in syn_threads:
            t.start()

        for t in threads:
            t.join()

        coord.request_stop()
        coord.join(syn_threads)
        attribute_embeddings_ret = list()

        for i in range(n_gpus):
            attribute_embeddings_ret.append(embedding_matrix[(i, i)].eval(sess))

    return np.concatenate(attribute_embeddings_ret, axis=0)[:n_names]
