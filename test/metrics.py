# -*- coding:utf-8 -*-
"""
author: byangg
datettime: 2020/5/15 15:14
"""

import unittest

import numpy as np

from rankingmetrics import ndcgs

y_true = np.array([[1, 2, 1, 2, 0, 2, 1, 1, 1, 1, 0, 2, 1, 0, 2],
                   [2, 0, 2, 2, 2, 1, 0, 2, 0, 0, 2, 0, 1, 0, 0]])
y_pred = np.array([[0.467, 0.864, 0.723, 0.887, 0.524, 0.553, 0.479,
                    0.491, 0.043, 0.99, 0.295, 0.758, 0.4, 0.042, 0.754],
                   [0.915, 0.512, 0.591, 0.866, 0.679, 0.68, 0.913,
                    0.209, 0.592, 0.279, 0.268, 0.182, 0.095, 0.179, 0.357]])
ndcgs_true = {'ndcg@1': 0.6666666666666666, 'ndcg@3': 0.6955328025094819, 'ndcg@5': 0.7312640159272952,
              'ndcg@10': 0.7567085003672513}


class TestStringMethods(unittest.TestCase):
    def test_ndcgs_array(self):
        res = ndcgs.ndcgs(y_true, y_pred)
        self.assertEqual(ndcgs_true, res)

    def test_ndcgs_groups(self):
        N, M = y_true.shape
        groups = [M] * N
        y_true_ = y_true.flatten()
        y_pred_ = y_pred.flatten()
        res = ndcgs.ndcgs(y_true_, y_pred_, groups)
        self.assertEqual(ndcgs_true, res)

    def test_ndcg_group_class(self):
        N, M = y_true.shape
        groups = [M] * N
        y_true_ = y_true.flatten()
        y_pred_ = y_pred.flatten()

        nwg = ndcgs.NdcgWithGroup(y_true_, groups)
        res = nwg.ndcg_score(y_pred_)
        self.assertEqual(res, ndcgs_true)



if __name__ == '__main__':
    unittest.main()
