# -*- coding:utf-8 -*-
"""
author: byangg
datettime: 2020/5/15 16:31
"""

import numpy as np
from rankingmetrics import ndcgs

y_true = np.random.randint(0, 3, (2,15))
y_pred = np.round(np.random.rand(2, 15),3)
res = ndcgs.ndcgs(y_true, y_pred)

print(y_true)
print(y_pred,sep=',')
print(res)