import numpy as np

data = np.load('data/predictions.npz')
ids = data['ids']
true = data['true']
pred_end = data['pred_end']

# 抽查前5个停车场的真实值和预测值（和前端展示对比）
for i in range(5):
    print(ids[i], "true[0:3] =", true[i][:3], "pred_end[0:3] =", pred_end[i][:3])