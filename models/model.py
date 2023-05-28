# -*- coding: utf-8 -*-
# Author : yuxiang Zeng

import torch as t
from torch.nn import *
import torch.nn.functional as F
import numpy as np
import pickle as pk

from models.attention import GraphAttentionLayer, SpGraphAttentionLayer
from utility.utils import *
from tqdm import *
import dgl as d
from dgl.nn.pytorch import SAGEConv
import scipy.sparse as sp



class SpGAT(t.nn.Module):
    def __init__(self, graph, nfeat, nhid, dropout, alpha, nheads, args):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.adj = self.get_adj_nrom_matrix(graph).cuda()
        self.numbers = len(self.adj)
        self.attentions = t.nn.ModuleList()
        self.nheads = nheads
        for i in range(self.nheads):
            temp = SpGraphAttentionLayer(nfeat, nhid, dropout=args.dropout, alpha=alpha, concat=True)
            self.attentions += [temp]
        self.dropout_layer = t.nn.Dropout(p=self.dropout, inplace=False)
        self.out_att = SpGraphAttentionLayer(nhid * nheads, nfeat, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, embeds):
        x = self.dropout_layer(embeds)
        x = torch.cat([att(x, self.adj) for att in self.attentions], dim=1)
        x = self.dropout_layer(x)
        x = F.elu(self.out_att(x, self.adj))
        return x

    @staticmethod
    def get_adj_nrom_matrix(graph):
        g = graph
        # 转换为邻接矩阵
        n = g.number_of_nodes()
        in_deg = g.in_degrees().numpy()
        rows = g.edges()[1].numpy()
        cols = g.edges()[0].numpy()
        adj = sp.csr_matrix(([1] * len(rows), (rows, cols)), shape=(n, n))

        def normalize_adj(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))  # 求每一行的和
            r_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^{-0.5}
            r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
            r_mat_inv_sqrt = sp.diags(r_inv_sqrt)  # D^{-0.5}
            return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))  # adj = D^{-0.5}SD^{-0.5}, S=A+I
        adj = torch.FloatTensor(np.array(adj.todense()))
        return adj



# 模型二
class GATMF(t.nn.Module):
    def __init__(self, args):
        super(GATMF, self).__init__()
        self.args = args
        user_lookup, serv_lookup, userg, servg = create_graph()
        self.usergraph, self.servgraph = userg, servg
        self.dim = args.dimension
        self.user_embeds = t.nn.Embedding(self.usergraph.number_of_nodes(), self.dim)
        t.nn.init.kaiming_normal_(self.user_embeds.weight)
        self.item_embeds = t.nn.Embedding(self.servgraph.number_of_nodes(), self.dim)
        t.nn.init.kaiming_normal_(self.item_embeds.weight)
        self.user_attention = SpGAT(self.usergraph, args.dimension, 32, args.dropout, args.alpha, args.heads, args)
        self.item_attention = SpGAT(self.servgraph, args.dimension, 32, args.dropout, args.alpha, args.heads, args)
        # interaction modules
        self.layers = t.nn.Sequential(
            # t.nn.Linear(4 * args.dimension, 32),  # FFN
            t.nn.Linear(2 * args.dimension, 32),  # FFN
            t.nn.LayerNorm(32),     # LayerNorm
            t.nn.ReLU(),            # ReLU
            t.nn.Linear(32, 32),    # FFN
            t.nn.LayerNorm(32),     # LayerNorm
            t.nn.ReLU(),            # ReLU
            t.nn.Linear(32, 1)      # y
        )

        self.cache = {}

    def forward(self, userIdx, itemIdx, train):
        if train:
            Index = t.arange(self.usergraph.number_of_nodes()).cuda()
            user_embeds = self.user_embeds(Index)
            Index = t.arange(self.servgraph.number_of_nodes()).cuda()
            serv_embeds = self.item_embeds(Index)
            att_embeds1 = self.user_attention(user_embeds)[userIdx]
            att_embeds2 = self.item_attention(serv_embeds)[itemIdx]
            estimated = self.layers(t.cat((att_embeds1, att_embeds2), dim=-1)).sigmoid().reshape(-1)
        else:
            att_embeds1 = self.cache['user'][userIdx]
            att_embeds2 = self.cache['item'][itemIdx]
            estimated = self.layers(t.cat((att_embeds1, att_embeds2), dim=-1)).sigmoid().reshape(-1)

        return estimated

    def prepare_test_model(self):
        Index = t.arange(self.usergraph.number_of_nodes()).cuda()
        user_embeds = self.user_embeds(Index)
        Index = t.arange(self.servgraph.number_of_nodes()).cuda()
        serv_embeds = self.item_embeds(Index)
        att_embeds1 = self.user_attention(user_embeds)[t.arange(100).cuda()]
        att_embeds2 = self.item_attention(serv_embeds)[t.arange(200).cuda()]
        self.cache['user'] = att_embeds1
        self.cache['item'] = att_embeds2

    def get_embeds_parameters(self):
        parameters = []
        for params in self.user_embeds.parameters():
            parameters += [params]
        for params in self.item_embeds.parameters():
            parameters += [params]
        return parameters

    def get_attention_parameters(self):
        parameters = []
        for params in self.user_attention.parameters():
            parameters += [params]
        for params in self.item_attention.parameters():
            parameters += [params]
        return parameters

    def get_mlp_parameters(self):
        parameters = []
        for params in self.layers.parameters():
            parameters += [params]
        return parameters


# 这里只是为了代码复用，用做好的图构建邻接矩阵，事实上也可以直接手写邻接矩阵
def create_graph():

    userg = d.graph([])
    servg = d.graph([])
    user_lookup = FeatureLookup()
    serv_lookup = FeatureLookup()
    ufile = pd.read_csv('./datasets/ClientWithCTX.csv')
    ufile = pd.DataFrame(ufile)
    ulines = ufile.to_numpy()
    ulines = ulines

    sfile = pd.read_csv('./datasets/PeerWithCTX.csv')
    sfile = pd.DataFrame(sfile)
    slines = sfile.to_numpy()
    slines = slines

    for i in range(100):
        user_lookup.register('User', i)
    for j in range(200):
        serv_lookup.register('Serv', j)


    region_col = 6

    for ure in ulines[:, region_col]:
        user_lookup.register('URE', ure)
    for uas in ulines[:, 4]:
        user_lookup.register('UAS', uas)
    for uas2 in ulines[:, 2]:
        user_lookup.register('UAS2', uas2)
    # for ucity in ulines[:, 3]:
    #     user_lookup.register('UCITY', ucity)


    for sre in slines[:, region_col]:
        serv_lookup.register('SRE', sre)
    for sas in slines[:, 4]:
        serv_lookup.register('SAS', sas)
    for sas2 in slines[:, 2]:
        serv_lookup.register('SAS2', sas2)
    # for scity in slines[:, 3]:
    #     serv_lookup.register('SCITY', scity)


    userg.add_nodes(len(user_lookup))
    servg.add_nodes(len(serv_lookup))


    for line in ulines:
        uid = line[0]
        ure = user_lookup.query_id(line[region_col])
        if not userg.has_edges_between(uid, ure):
            userg.add_edges(uid, ure)

        uas = user_lookup.query_id(line[4])
        if not userg.has_edges_between(uid, uas):
            userg.add_edges(uid, uas)

        uas2 = user_lookup.query_id(line[2])
        if not userg.has_edges_between(uid, uas2):
            userg.add_edges(uid, uas2)

        # ucity = user_lookup.query_id(line[3])
        # if not userg.has_edges_between(uid, ucity):
        #     userg.add_edges(uid, ucity)

    for line in slines:
        sid = line[0]
        sre = serv_lookup.query_id(line[region_col])
        if not servg.has_edges_between(sid, sre):
            servg.add_edges(sid, sre)

        sas = serv_lookup.query_id(line[4])
        if not servg.has_edges_between(sid, sas):
            servg.add_edges(sid, sas)

        sas2 = serv_lookup.query_id(line[2])
        if not servg.has_edges_between(sid, sas2):
            servg.add_edges(sid, sas2)

        # scity = serv_lookup.query_id(line[3])
        # if not servg.has_edges_between(sid, scity):
        #     servg.add_edges(sid, scity)

    userg = d.add_self_loop(userg)
    userg = d.to_bidirected(userg)
    servg = d.add_self_loop(servg)
    servg = d.to_bidirected(servg)
    return user_lookup, serv_lookup, userg, servg


class FeatureLookup:

    def __init__(self):
        self.__inner_id_counter = 0
        self.__inner_bag = {}
        self.__category = set()
        self.__category_bags = {}
        self.__inverse_map = {}

    def register(self, category, value):
        # 添加进入类别
        self.__category.add(category)
        # 如果类别不存在若无则，则新增一个类别子树
        if category not in self.__category_bags:
            self.__category_bags[category] = {}

        # 如果值不在全局索引中，则创建之，id += 1
        if value not in self.__inner_bag:
            self.__inner_bag[value] = self.__inner_id_counter
            self.__inverse_map[self.__inner_id_counter] = value
            # 如果值不存在与类别子树，则创建之
            if value not in self.__category_bags[category]:
                self.__category_bags[category][value] = self.__inner_id_counter
            self.__inner_id_counter += 1

    def query_id(self, value):
        # 返回索引id
        return self.__inner_bag[value]

    def query_value(self, id):
        # 返回值
        return self.__inverse_map[id]

    def __len__(self):
        return len(self.__inner_bag)