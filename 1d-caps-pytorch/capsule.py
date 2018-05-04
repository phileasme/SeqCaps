import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def squash(x, axis=-1):
    s_squared_norm = x.pow(2).sum(dim=-1) + 1e-8 #epsilon
    scale = s_squared_norm.pow(2)/ (0.5 + s_squared_norm)
    # print(scale.view(-1,scale.size()[1],1).size())
    # print(x.size())
    scale = scale.view(32,10,1)
    return scale * x
    # torch.from_numpy(scale.data.numpy().reshape(-1,scale.size()[1]) * x.data.numpy())

def softmax(x, axis=-1):
    ex = torch.exp(x - torch.max(x, axis, keepdim=True))
    sum = torch.sum(ex, axis, keepdim=True)
    return ex / sum
    # ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    # return ex/K.sum(ex, axis=axis, keepdims=True)

class Capsule(nn.Module):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.w = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(1, 784, num_capsule * dim_capsule), gain=nn.init.calculate_gain('relu')))
        # self.b = nn.Parameter(torch.zeros(32, 1, self.num_capsule))
        self.b = nn.Parameter(torch.zeros((1, self.num_capsule)))

    def forward(self, u_vecs):
        batch_size = u_vecs.size()[0]
        input_num_capsule = u_vecs.size()[1]

        u_hat_vecs = torch.nn.functional.conv1d(u_vecs, self.w.view(160,1,784))
        u_hat_vecs = u_hat_vecs.view(batch_size, input_num_capsule, self.num_capsule * self.dim_capsule)
        u_hat_vecs = u_hat_vecs.view(batch_size, input_num_capsule, self.num_capsule, self.dim_capsule)

                                        # 32, 10, 1, 32
        u_hat_vecs = u_hat_vecs.permute(0,2,1,3)
        #
        # return agre(u_hat_vecs)
        # b = torch.zeros(batch_size, input_num_capsule, self.num_capsule)
        b_batch = self.b.expand((batch_size, input_num_capsule, self.num_capsule))
        for i in range(self.routings):
            c = F.softmax(b_batch, 1)
            c.unsqueeze(-1)
            o = torch.matmul(c, u_hat_vecs)
            print(o.size())
            # o = o[0,:,:,:]

            if i < self.routings - 1:
                o = o / torch.norm(o)
                # 32 10 16
                uh = u_hat_vecs
                uh = uh.squeeze(1)
                print(o.view(-1,16,1).size())
                print(uh.size())

                b_batch = torch.bbm(o.view(-1,1,16), uh.view(-1,10,16)).view(32,10,16)

                print(b_batch)

        return F.sigmoid(o)
