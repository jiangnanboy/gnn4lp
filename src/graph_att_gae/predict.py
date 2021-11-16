import os
from configparser import ConfigParser
import numpy as np
import scipy.sparse as sp

import sys
sys.path.append(r'/home/shiyan/project/gnn4lp/')

from src.util.load_data import load_data_with_features, load_data_without_features

class Predict():
    def __init__(self):
        self.hidden_emb = None
        self.adj_orig = None

    def load_model_adj(self, config_path):
        '''
        load hidden_emb and adj
        :param config_path:
        :return:
        '''
        if os.path.exists(config_path) and (os.path.split(config_path)[1].split('.')[0] == 'config') and (os.path.splitext(config_path)[1].split('.')[1] == 'cfg'):
            # load config file
            config = ConfigParser()
            config.read(config_path)
            section = config.sections()[0]

            # data catalog path
            data_catalog = config.get(section, "data_catalog")

            # node cites path
            node_cites_path = config.get(section, "node_cites_path")
            node_cites_path = os.path.join(data_catalog, node_cites_path)

            # node features path
            node_features_path = config.get(section, 'node_features_path')
            node_features_path = os.path.join(data_catalog, node_features_path)

            # 是否带节点特征
            with_feats = config.getboolean(section, 'with_feats')

            # model save/load path
            model_path = config.get(section, "model_path")

            if not os.path.exists(model_path):
                raise FileNotFoundError('Not found model file!')
            if not os.path.exists(node_cites_path):
                raise FileNotFoundError('Not found node_cites_file!')

            self.hidden_emb = np.load(model_path)
            if with_feats:
                if not os.path.exists(os.path.join(data_catalog, node_features_path)):
                    raise FileNotFoundError('Not found node_features_file!')
                adj, _ = load_data_with_features(node_cites_path, node_features_path)
            else:
                adj = load_data_without_features(node_cites_path)

            # 除去对角线元素
            self.adj_orig = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            self.adj_orig.eliminate_zeros()
        else:
            raise FileNotFoundError('File config.cfg not found : ' + config_path)

    def predict(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        # 内积
        adj_rec = np.dot(self.hidden_emb, self.hidden_emb.T)
        adj_rec = sigmoid(adj_rec)
        return self.adj_orig, adj_rec

if __name__ == '__main__':
    config_path = os.path.join(os.getcwd(), 'config.cfg')
    predict = Predict()
    predict.load_model_adj(config_path)
    adj_orig, adj_rec = predict.predict()
    adj_rec = (adj_rec > 0.5) + 0
    print('adj_orig: {}, \n adj_rec: {}'.format(adj_orig, adj_rec[0][:100]))
