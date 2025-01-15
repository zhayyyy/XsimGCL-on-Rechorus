import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F

from models.BaseModel import GeneralModel

class XsimGCLBase(object):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=3,
                            help='Number of XsimGCL layers.')
        # extra args for XsimGCL
        parser.add_argument('--eps', type=float, default=0.2,
                            help='eps')
        parser.add_argument('--cl_rate', type=float, default=0.2,
                            help='cl_rate')
        parser.add_argument('--layer_cl', type=int, default=1,
                            help='layer_cl')
        return parser
    
    @staticmethod
    def build_adjmat(user_count, item_count, train_mat, selfloop_flag=False):
        R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
        for user in train_mat:
            for item in train_mat[user]:
                R[user, item] = 1
        R = R.tolil()

        adj_mat = sp.dok_matrix(
            (user_count+item_count, user_count+item_count), 
            dtype=np.float32
        )
        adj_mat = adj_mat.tolil()
        adj_mat[:user_count, user_count:] = R
        adj_mat[user_count:, :user_count] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            # D^-1/2 * A * D^-1/2
            rowsum = np.array(adj.sum(1)) + 1e-10

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()
        
        if selfloop_flag:
            norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        else:
            norm_adj_mat = normalized_adj_single(adj_mat)

        return norm_adj_mat.tocsr()
    
    def _base_init(self, args, corpus):
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.norm_adj = self.build_adjmat(
            corpus.n_users, corpus.n_items, corpus.train_clicked_set
        )
        self._base_define_params()
        self.apply(self.init_weights)
        # extra init for XsimGCL
        self.eps = args.eps
        self.cl_rate = args.cl_rate
        self.layer_cl = args.layer_cl
        self.temperature = 0.15

    def _base_define_params(self):
        self.encoder = XsimGCLEncoder(
            self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers
        )
    
    def forward(self, feed_dict):
        self.check_list = []

        users, items = feed_dict['user_id'], feed_dict['item_id']
        u_e, i_e, u_t_e, i_t_e, u_t_e_c, i_t_e_c = self.encoder(users, items)

        prediction = (u_e[:, None, :] * i_e).sum(dim=-1)
        prediction_loss = (u_t_e[:, None, :] * i_t_e).sum(dim=-1)

        return {
            'prediction': prediction.view(feed_dict['batch_size'], -1),
            'prediction_loss': prediction_loss.view(feed_dict['batch_size'], -1),
            'u_t_e': u_t_e, 'i_t_e': i_t_e, 'u_t_e_c': u_t_e_c, 'i_t_e_c': i_t_e_c,
            'feed_dict': feed_dict,
        }

class XsimGCL(GeneralModel, XsimGCLBase):
    reader = 'BaseReader'
    runner = 'XsimGCLRunner'
    extra_log_args = [
        'emb_size', 'n_layers', 'batch_size',
        # extra args for XsimGCL
        'eps', 'cl_rate', 'layer_cl'
    ]

    @staticmethod
    def parse_model_args(parser):
        parser = XsimGCLBase.parse_model_args(parser)
        return GeneralModel.parse_model_args(parser)
    
    def __init__(self, args, corpus): # self.temerature not defined
        GeneralModel.__init__(self, args, corpus)
        self._base_init(args, corpus)
        # extra init for XsimGCL    
    
    def forward(self, feed_dict):
        out_dict = XsimGCLBase.forward(self, feed_dict)
        return out_dict

    def loss(self, out_dict):
        prediction_loss = out_dict['prediction_loss']
        pos_pred, neg_pred = prediction_loss[:, 0], prediction_loss[:, 1:]
        neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
        loss = -(((pos_pred[:, None]-neg_pred).sigmoid() * \
                  neg_softmax).sum(dim=1)).clamp(min=1e-8, max=1-1e-8).log().mean()
        loss += self.cl_rate * self.cl_loss(
            out_dict['u_t_e'], out_dict['u_t_e_c'],
            out_dict['i_t_e'], out_dict['i_t_e_c']
        )
        reg_loss = self.l2_reg_loss(1e-4,
            out_dict['u_t_e'], out_dict['u_t_e_c'],
            out_dict['i_t_e'], out_dict['i_t_e_c']
        )
        loss += reg_loss
        return loss

    def l2_reg_loss(self, reg, *embeddings):
        emb_loss = 0
        for emb in embeddings:
            emb_loss += torch.norm(emb, p=2) / emb.shape[0]
        return reg * emb_loss

    def cl_loss(self, user_view, user_view_cl, item_view, item_view_cl):
        size, batch = item_view.shape[0], item_view.shape[2]

        user_cl_loss = self.infoNCE(user_view, user_view_cl)
        item_cl_loss = self.infoNCE(item_view.view(size*2, batch), item_view_cl.view(size*2, batch))
        return user_cl_loss + item_cl_loss

    def infoNCE(self, view1, view2, b_cos=True):
        if b_cos:
            view1 = F.normalize(view1, dim=1)
            view2 = F.normalize(view2, dim=1)

        pos_score  = torch.matmul(view1, view2.T) / self.temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()
    
class XsimGCLEncoder(nn.Module):
    def __init__(self, user_count, item_count, emb_size, norm_adj, n_layers=3, 
        # extra args for XsimGCL
        eps=0.2, cl_rate=0.2, layer_cl=1
    ):
        super(XsimGCLEncoder, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.emb_size = emb_size
        self.layers = [emb_size] * n_layers
        self.n_layers = n_layers
        self.norm_adj = norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).cuda()
        # extra init for XsimGCL
        self.eps = eps
        self.cl_rate = cl_rate
        self.layer_cl = layer_cl
    
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.emb_size))),
        })
        return embedding_dict
    
    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor(np.array([coo.row, coo.col]))
        v = torch.from_numpy(coo.data).float()
        return torch.sparse_coo_tensor(i, v, coo.shape)

    def forward(self, users, items):
        all_embeddings = []
        train_embeddings = []
        train_embeddings_cl = []

        ego_embeddings = torch.cat([
            self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0
        )

        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            
            random_noise = torch.rand_like(ego_embeddings).cuda()
            ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps

            train_embeddings.append(ego_embeddings)
            if k == self.layer_cl-1:
                train_embeddings_cl.append(ego_embeddings)

        size_list = [self.user_count, self.item_count]

        def func(embeddings):
            embeddings = torch.stack(embeddings, dim=1)
            embeddings = torch.mean(embeddings, dim=1)
            u_embeddings, i_embeddings = torch.split(embeddings, size_list)
            u_embeddings = u_embeddings[users, :]
            i_embeddings = i_embeddings[items, :]
            return u_embeddings, i_embeddings     

        return *func(all_embeddings), *func(train_embeddings), *func(train_embeddings_cl)
                
# python main.py --num_workers 0 --pin_memory 1 --model_name XsimGCL --dataset Grocery_and_Gourmet_Food --batch_size 1024
# python main.py --num_workers 0 --pin_memory 1 --model_name XsimGCL --dataset ML_1MTOPK --batch_size 1024
# python main.py --num_workers 0 --pin_memory 1 --model_name XsimGCL --dataset MINDTOPK --batch_size 1024