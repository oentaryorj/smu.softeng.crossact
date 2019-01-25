__author__ = 'Wei Xie'
__email__ = 'linegroup3@gmail.com'
__affiliation__ = 'Living Analytics Research Centre, Singapore Management University'
__website__ = 'http://mysmu.edu/phdis2012/wei.xie.2012'

import numpy as np
import matplotlib.pyplot as plt


class LSH:
    def __init__(self, n_bins, n_hashtables):
        self.n_bins = n_bins
        self.n_hashtables = n_hashtables
        self.random_projections = None
        self.hash_indices = list()

        for j in range(n_hashtables):
            self.hash_indices.append(dict())

    def load(self, X):
        n, m = X.shape
        np.random.seed(3)
        self.random_projections = np.random.normal(0., 1., [self.n_bins * self.n_hashtables, m])
        point_indices = list()

        for i in range(n):
            point = X[i]
            bins = self._hashbins(point)
            indices = list()

            for j in range(self.n_hashtables):
                indices.append(self._bins2index(bins[j * self.n_bins: (j + 1) * self.n_bins]))

            point_indices.append(indices)
            self._add(indices, i)

        return point_indices

    def _add(self, indices, i):
        for j, index in enumerate(indices):
            if index in self.hash_indices[j]:
                self.hash_indices[j][index].add(i)
            else:
                self.hash_indices[j][index] = {i}

    def print_indices(self):
        for j in range(self.n_hashtables):
            print ('table ', j, ':')
            print ('keys', len(self.hash_indices[j]))
            print (self.hash_indices[j])
            print ('-' * 25)

    @staticmethod
    def _bins2index(bins):
        ret = bins[0]

        for i in bins[1:]:
            ret <<= 1
            ret += i

        return ret

    def _hashbins(self, point):
        bins = np.zeros(self.random_projections.shape[0], dtype=np.int32)

        for i in range(self.random_projections.shape[0]):
            if np.dot(point, self.random_projections[i]) > 0:
                bins[i] = 1

        return bins

    def query(self, point_indices):
        ret = set()

        for j, index in enumerate(point_indices):
            if index in self.hash_indices[j]:
                ret |= self.hash_indices[j][index]

        return ret


if __name__ == '__main__':
    lsh = LSH(20, 5)
    X = np.random.normal(0., 1., [100, 2])

    point_indices = lsh.load(X)

    print ('len', len(point_indices))
    print (point_indices[0])

    ret = lsh.query(point_indices[0])
    ret = list(ret)

    print (ret)
    print (len(ret), (len(ret) + 0.) / X.shape[0])

    plt.plot(X[:, 0], X[:, 1], '.')
    plt.plot(X[ret, 0], X[ret, 1], 'k.')
    plt.plot(X[0, 0], X[0, 1], 'r+')
    plt.show()
