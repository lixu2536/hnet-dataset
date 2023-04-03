import torch
import numpy as np

a1 = np.array([[[[3.0, 4.0, 5.0, 0.0],
              [3.0, 4.0, 5.0, 0.0]]],
              [[[3.0, 4.0, 5.0, 0.0],
                [3.0, 4.0, 5.0, 0.0]]]]
              )

a2 = np.array([[[[4.0, 5.0, 6.0, 2.3],
              [4.0, 5.0, 6.0, 2.3]]],
              [[[4.0, 5.0, 6.0, 2.3],
                [4.0, 5.0, 6.0, 2.3]]]]
              )
print(a1.shape)
print(a2.shape)

b1 = torch.from_numpy(a1)
b2 = torch.from_numpy(a2)

mask = b1 == 0.0
print(mask)
print('检查维度：', b1[~mask])

# criterion1 = torch.nn.MSELoss(reduction='sum')
criterion1 = torch.nn.MSELoss(reduction='mean')
loss1 = criterion1(b1[~mask], b2[~mask])
print(loss1)