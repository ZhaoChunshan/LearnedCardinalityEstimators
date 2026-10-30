import torch
import torch.nn as nn
import numpy as np


class CVAE_G(nn.Module):
    """
    Generating new Plan depending on condition of query plan
    """
    def __init__(self, plan_size, latent_size, layers_size, query_size):
        super(CVAE_G, self).__init__()
        self.plan_size = plan_size
        self.query_size = query_size
        self.latent_size = latent_size
        # self.layers_size = layers_size
        # self.attn = nn.MultiheadAttention(3, 3, batch_first=True)

        # encode
        self.fc1  = nn.Linear(plan_size + query_size + 1 + 1, layers_size[0])
        # nn.init.xavier_uniform(self.fc1.weight)
        # self.norm1 = torch.nn.LayerNorm(layers_size[0])
        self.fc21 = nn.Linear(layers_size[0], layers_size[1])
        # nn.init.xavier_uniform(self.fc21.weight)
        # self.norm2 = torch.nn.LayerNorm(layers_size[1])
        self.fc31 = nn.Linear(layers_size[1], latent_size)
        # nn.init.xavier_uniform(self.fc31.weight)
        self.fc32 = nn.Linear(layers_size[1], latent_size)
        # nn.init.xavier_uniform(self.fc32.weight)

        # decode
        self.fc4 = nn.Linear(latent_size + query_size + 1 + 1, layers_size[1])
        # nn.init.xavier_uniform(self.fc4.weight)
        self.fc51 = nn.Linear(layers_size[1], layers_size[0])
        # nn.init.xavier_uniform(self.fc51.weight)
        self.fc6 = nn.Linear(layers_size[0], plan_size)
        # nn.init.xavier_uniform(self.fc6.weight)

        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c, conf_dist, out_dist): # Q(z|x, c)
        '''
        x: (bs, plan_size)
        c: (bs, query_size)
        '''
        # inputs = torch.cat([x, pg_dist, out_dist], 1) # (bs, plan_size + query_size)
        # bs = x.size(0)
        # x = x.reshape(bs, -1, 3)
        # c = c.reshape(bs, -1, 3)
        # multihead_attn, _ = self.attn(c, x, x)
        # c += multihead_attn
        # c = c.reshape(bs, -1)
        # x = x.reshape(bs, -1)
        inputs = torch.hstack((x, c, conf_dist, out_dist))
        h1 = self.elu(self.fc1(inputs))
        h21 = self.elu(self.fc21(h1))
        z_mu = self.fc31(h21)
        z_var = self.fc32(h21)
        return z_mu, z_var

    def reparameterize(self, mu, logvar, eps=None):
        """
        Reparametrization required for the gradient flow
        """
        std = torch.exp(0.5*logvar)
        if eps is None:
            eps = torch.randn_like(std)
        
        return mu + eps*std

    def decode(self, x, z, c, conf_dist, out_dist): # P(x|z, c)
        '''
        x: (bs, plan_size)
        z: (bs, latent_size)
        c: (bs, query_size)
        '''
        # bs = x.size(0)
        inputs = torch.hstack((z, c, conf_dist, out_dist)) # (bs, latent_size + query_size)
        # inputs = z
        h4 = self.elu(self.fc4(inputs))
        h51 = self.elu(self.fc51(h4))
        # h52 = self.elu(self.fc52(c))
        pred = self.fc6(h51)
        # pred = self.softmax(pred.reshape(bs, -1, 3)).reshape(bs, -1)
        # pred = self.fc4(h4)  # * x.bool().float()
        # return self.log_softmax(pred)
        return pred
    
    # def predict_conf(self, x, out, c): # Q(z|x, c)
    #     '''
    #     x: (bs, plan_size)
    #     c: (bs, query_size)
    #     '''
    #     inputs = torch.hstack((x, out, c))
    #     h6 = self.elu(self.fc7(inputs))
    #     h7 = self.elu(self.fc8(h6))
    #     h8 = self.elu(self.fc9(h7))
    #     conf = self.fc10(h8)
    #     return conf

    def forward(self, x, c, conf_dist, out_dist, eps=None):
        mu, logvar = self.encode(x, c, conf_dist, out_dist)
        z = self.reparameterize(mu, logvar, eps)
        out = self.decode(x, z, c, conf_dist, out_dist)
        # conf = self.predict_conf(x, out, c)
        return out, mu, logvar, z