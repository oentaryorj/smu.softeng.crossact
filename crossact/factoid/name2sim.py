__author__ = 'Wei Xie'
__email__ = 'linegroup3@gmail.com'
__affiliation__ = 'Living Analytics Research Centre, Singapore Management University'
__website__ = 'http://mysmu.edu/phdis2012/wei.xie.2012'

import os
import codecs
import dis_ngram_sim
import exp_config


def name2sim(input_names, prefix='name', return_matrix=False):
    print('prefix in name2sim', prefix)  # debugging

    PATH = exp_config.get('data', 'path')
    method = exp_config.get('predicate_name', 'method')

    ip_ad = exp_config.get('dispy', 'ip')
    port = int(exp_config.get('dispy', 'port'))
    remote_path = exp_config.get('dispy', 'remote_path')

    print ('In name2sim() method ', method)
    assert method in ['jaro_winkler', 'tfidf']

    if os.path.isfile(PATH + prefix + '_list_' + method + '.txt'):
        names = list()
        fin = codecs.open(PATH + prefix + '_list_' + method + '.txt', 'r', 'utf-8')

        for line in fin:
            names.append(line[:-1])

        fin.close()
    else:
        names = input_names
        fout = codecs.open(PATH + prefix + '_list_' + method + '.txt', 'w', 'utf-8')

        for name in names:
            fout.write(name)
            fout.write('\n')

        fout.close()

    sim = None

    if method == 'jaro_winkler':
        pass

    if method == 'tfidf':
        sim = dis_ngram_sim.distributed_name2sim_tfidf(names,
                                                       '.',
                                                       remote_path, 32,
                                                       nodes=[(ip_ad, port)],
                                                       return_sim=return_matrix)

    name2eid = dict(zip(names, range(len(names))))
    name2eid[None] = len(names)
    return name2eid, sim
