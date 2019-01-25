__author__ = 'Wei Xie'
__email__ = 'linegroup3@gmail.com'
__affiliation__ = 'Living Analytics Research Centre, Singapore Management University'
__website__ = 'http://mysmu.edu/phdis2012/wei.xie.2012'

import os, sys, time
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import itertools
import jellyfish
import json
import functools

max_ngram_len = 5
indexed_ngram_len = 3

inverted_index = dict()
_X = None  # tf-idf vectors


def get_ngram(name, length):
    ret = list()

    for i in range(len(name) - length + 1):
        token = name[i: i + length]
        ret.append(token)

    return ret


def get_all_ngrams(name, _id):
    ret = list()

    for length in range(1, max_ngram_len + 1):
        ngrams = get_ngram(name, length)
        ret.append(ngrams)

        if length == indexed_ngram_len:
            for ngram in ngrams:
                if ngram in inverted_index:
                    inverted_index[ngram].add(_id)
                else:
                    inverted_index[ngram] = set([_id])

    ret = list(itertools.chain.from_iterable(ret))
    return ' '.join(ret)


def process_unit_tfidf(i, name):
    global _X

    if i % 100 == 0:
        sys.stdout.write("\r%d%%" % (100 * i / _X.shape[0]))
        sys.stdout.flush()

    ngrams = get_ngram(name, indexed_ngram_len)

    candidate_set = set()
    for ngram in ngrams:
        if ngram in inverted_index:
            candidate_set |= inverted_index[ngram]

    candidates = list(candidate_set)

    row = [i] * len(candidates)
    col = candidates

    if len(candidates) > 0:
        data = np.power(linear_kernel(_X[i:i + 1], _X[candidates]).flatten(), 0.25)
    else:
        data = []

    return row, col, data


def process_unit_jw(i, names):
    name = names[i]

    if i % 100 == 0:
        sys.stdout.write("\r%d%%" % (100 * i / len(names)))
        sys.stdout.flush()

    ngrams = get_ngram(name, indexed_ngram_len)
    candidate_set = set()

    for ngram in ngrams:
        if ngram in inverted_index:
            candidate_set |= inverted_index[ngram]

    candidates = list(candidate_set)
    col = list()
    data = list()

    for x in candidates:
        sim = 2 * jellyfish.jaro_winkler(name, names[x]) - 1

        if sim != 0.:
            col.append(x)
            data.append(sim)

    row = [i] * len(col)
    return row, col, data


def name2sim_tfidf(names):
    '''
    transform a list of names to a similarity matrix, based on their n-gram tf-idf vectors

    :param names:
    :return: a sparse matrix (csr_matrix)
    '''
    global _X, inverted_index

    inverted_index = dict()

    print ('Maximum N-gram length', max_ngram_len)

    ct = time.time()
    vectorizer = TfidfVectorizer(min_df=1, stop_words=None, analyzer='word', tokenizer=lambda x: x.split())
    corpus = list()

    for _id, name in enumerate(names):
        corpus.append(get_all_ngrams(name, _id))

    _X = vectorizer.fit_transform(corpus)

    print ('TF-IDF matrix builded with ', (time.time() - ct) / 60., 'mins.')
    print (type(_X), _X.shape)
    print (vectorizer.get_feature_names()[:50])

    ct = time.time()
    ret = [process_unit_tfidf(i, name) for i, name in enumerate(names)]

    data_list = list()
    row_list = list()
    col_list = list()

    for element in ret:
        row, col, data = element
        row_list.append(row)
        col_list.append(col)
        data_list.append(data)

    row_list = list(itertools.chain(*row_list))
    col_list = list(itertools.chain(*col_list))
    data_list = list(itertools.chain(*data_list))

    print ('Sparse matrix X constructed, use ', (time.time() - ct), 's')

    return csr_matrix((data_list, (row_list, col_list)), shape=(len(names), len(names)))


def save_inverted_index(inv_index, path):
    ret = dict()

    for k, v in inv_index.items():
        ret[k] = list(v)

    with open(path + '/inverted_index.json', 'w') as outfile:
        json.dump(ret, outfile)


def process_job(blk_id, path, n_blk, send_file=False):
    global names, tfidf, inverted_index

    from array import array
    import scipy.sparse
    import numpy as np
    import xxhash
    from sklearn.metrics.pairwise import linear_kernel
    import scipy.sparse
    from scipy.sparse import csr_matrix
    import heapq
    from collections import defaultdict

    seed = 15137
    print ('ok')

    if send_file:
        top_30_dict = defaultdict(list)

    output_id = open(path + '/similarity_id_block_' + str(blk_id) + '.dat', 'wb')
    output_v = open(path + '/similarity_v_block_' + str(blk_id) + '.dat', 'wb')

    for ngram, ids in inverted_index.items():
        if xxhash.xxh32(ngram, seed).intdigest() % n_blk != blk_id:
            continue

        rows = list()
        cols = list()
        data = list()
        ids = list(ids)

        for i in range(len(ids)):
            for j in range(i, len(ids)):
                rows.append(ids[i])
                cols.append(ids[j])

        if len(ids) == 1:
            data.append(1.)
        elif len(ids) > 0:
            vs = tfidf[ids]
            data += list(np.power(linear_kernel(vs, vs)[np.triu_indices(len(ids))], 0.25))

        reduced_data = list()
        reduced_id = list()

        for row, col, dt in zip(rows, cols, data):
            if send_file:
                hq = top_30_dict[row]

                if len(hq) < 30:
                    heapq.heappush(hq, (dt, col))
                else:
                    heapq.heappushpop(hq, (dt, col))

                if col != row:
                    hq = top_30_dict[col]
                    if len(hq) < 30:
                        heapq.heappush(hq, (dt, row))
                    else:
                        heapq.heappushpop(hq, (dt, row))
            else:
                reduced_id.append(row)
                reduced_id.append(col)
                reduced_data.append(dt)

        if not send_file:
            op_id = array('I')
            op_id.fromlist(reduced_id)
            op_id.tofile(output_id)
            op_rd = array('f')
            op_rd.fromlist(reduced_data)
            op_rd.tofile(output_v)

    output_id.close()
    output_v.close()

    if not send_file:
        import socket
        host = socket.gethostname()
        return (host, path + '/similarity_id_block_' + str(blk_id) + '.dat',
                path + '/similarity_v_block_' + str(blk_id) + '.dat')
    else:
        os.remove(path + '/similarity_id_block_' + str(blk_id) + '.dat')
        os.remove(path + '/similarity_v_block_' + str(blk_id) + '.dat')

        _ct = time.time()
        exist_sid = set()
        rows = list()
        cols = list()
        values = list()

        for row, cs in top_30_dict.items():
            for sv, col in cs:
                sid = min(row, col) * len(names) + max(row, col)

                if sid in exist_sid:
                    continue

                exist_sid.add(sid)
                rows.append(row)
                cols.append(col)
                values.append(sv)

        mat = csr_matrix((values, (rows, cols)), shape=(len(names), len(names)))
        scipy.sparse.save_npz(path + '/similarity_mat_block_' + str(blk_id) + '.npz', mat)

        print ('Load block ', blk_id, 'used ', time.time() - _ct, 's')

        dispy_send_file(path + '/similarity_mat_block_' + str(blk_id) + '.npz')
        os.remove(path + '/similarity_mat_block_' + str(blk_id) + '.npz')

        import socket
        host = socket.gethostname()
        return (host, path + '/similarity_mat_block_' + str(blk_id) + '.npz')


def distributed_name2sim_tfidf(names, path, node_path,
                               n_jobs=3, nodes=['127.0.0.1'],
                               n_rounds=1, return_sim=False):
    np.save(path + '/names.npy', names)

    global _X, inverted_index
    inverted_index = dict()

    print ('Maximum N-gram length', max_ngram_len)

    ct = time.time()
    vectorizer = TfidfVectorizer(min_df=1, stop_words=None, analyzer='word', tokenizer=lambda x: x.split())
    corpus = list()

    for _id, name in enumerate(names):
        corpus.append(get_all_ngrams(name, _id))

    _X = vectorizer.fit_transform(corpus)

    print ('TF-IDF matrix builded with ', (time.time() - ct) / 60., 'mins.')
    print (type(_X), _X.shape)
    print (vectorizer.get_feature_names()[:50])

    save_inverted_index(inverted_index, path)

    scipy.sparse.save_npz(path + '/tfidf.npz', _X)

    # setup
    def setup(f_names, f_tfidf, f_inverted_index):
        global names, tfidf, inverted_index

        import numpy as np
        import scipy.sparse

        def load_inverted_index(path):
            import json

            with open(path, 'r') as infile:
                inv_index0 = json.load(infile)

            ret = dict()

            for k, v in inv_index0.items():
                ret[k] = set(v)

            return ret

        names = np.load(f_names)
        tfidf = scipy.sparse.load_npz(f_tfidf)
        inverted_index = load_inverted_index(f_inverted_index)

        return 0

    def cleanup():
        global names, tfidf, inverted_index
        del names, tfidf, inverted_index

    ct = time.time()

    # start submit jobs
    import dispy

    cluster = dispy.JobCluster(process_job, nodes=nodes,
                               depends=[get_ngram, path + '/names.npy',
                                        path + '/tfidf.npz', path + '/inverted_index.json'],
                               setup=functools.partial(setup, path + '/names.npy',
                                                       path + '/tfidf.npz', path + '/inverted_index.json'),
                               cleanup=cleanup)

    blk_id = 0
    block_size = int(len(names) / (n_rounds * n_jobs) + 1)

    print (block_size)

    ret_files = list()

    while blk_id * block_size < len(names):
        jobs = []

        while len(jobs) < n_jobs:
            job = cluster.submit(blk_id, node_path, n_rounds * n_jobs, return_sim)
            blk_id += 1
            jobs.append(job)

        for job in jobs:
            ret_file = job()
            ret_files.append(ret_file)

            if job.stdout:
                print(job.stdout)
            if job.stderr:
                print(job.stderr)
            if job.exception:
                print(job.exception)

        cluster.print_status()

    print('final blks', blk_id)

    if not return_sim:
        return ret_files

    mat = csr_matrix(([], ([], [])), shape=(len(names), len(names)))

    for bid in range(blk_id):
        mat0 = scipy.sparse.load_npz(path + '/similarity_mat_block_' + str(bid) + '.npz')
        mat0 = mat0 + mat0.transpose() - scipy.sparse.identity(len(names))
        mat = mat.maximum(mat0)
        # assert mat.max() <= 1.
        os.remove(path + '/similarity_mat_block_' + str(bid) + '.npz')

    print ('Sparse matrix X constructed, use ', (time.time() - ct), 's')
    return mat


def name2sim_jw(names):
    '''
    transform a list of names to a similarity matrix, based on their jaro_winkler similarities
    :param names:
    :return: a sparse matrix (csr_matrix)
    '''

    corpus = list()

    for _id, name in enumerate(names):
        corpus.append(get_all_ngrams(name, _id))

    ct = time.time()
    ret = [process_unit_jw(i, names) for i, name in enumerate(names)]

    data_list = list()
    row_list = list()
    col_list = list()

    for element in ret:
        row, col, data = element
        row_list.append(row)
        col_list.append(col)
        data_list.append(data)

    row_list = list(itertools.chain(*row_list))
    col_list = list(itertools.chain(*col_list))
    data_list = list(itertools.chain(*data_list))

    print ('Sparse matrix X constructed, use ', (time.time() - ct), 's')

    return csr_matrix((data_list, (row_list, col_list)), shape=(len(names), len(names)))
