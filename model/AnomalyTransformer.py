import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer, AttentionLayer_new, AnomalyAttention_new
from .embed import DataEmbedding, TokenEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, attention, attention_new, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.attention_c = attention_new
        self.conv1 = nn.Linear(in_features=d_model, out_features=d_ff) # del
        self.conv2 = nn.Linear(in_features=d_ff, out_features=d_model) # del
        self.linear_channel = nn.Linear(in_features=4, out_features=55) # del

        self.conv3 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.conv4 = nn.Linear(in_features=d_ff, out_features=d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, layer, attn_mask=None, cluster_temporal=None, cluster_channel=None):

        B, L, C, D = x.shape
        ori_x = x.clone()

        #- Temporal clustering
        if layer==0:
            feats_t = [None] * C
            cluster_t = [None] * C
            for ch in range(C):
                idx_cluster_t, cluster_num_t = cluster_dpc_knn(ori_x[:,1:,ch,:], cluster_num=cluster_temporal, k=20) # [B,L,D] -> [B,L]
                clustered_feats_t = merge_tokens(ori_x[:,1:,ch,:].unsqueeze(1), idx_cluster_t, cluster_num_t) # [B,1,cluster,D]
                feats_t[ch] = clustered_feats_t.permute(1,0,2,3) # [1,B,cluster,D]
                cluster_t[ch] = idx_cluster_t.unsqueeze(0)
            feats_t = torch.cat(feats_t, dim=0).permute(1,0,2,3) # [B,C,cluster,D]
            feats_t = feats_t.permute(0,2,1,3) # [B,cluster,C,D]
            cluster_t = torch.cat(cluster_t, dim=0).permute(1,0,2) # [B,C,L]

            idx_cluster_t = cluster_t.unsqueeze(-1).repeat(1,1,1,D).permute(0,2,1,3) # [B,L,C,D]
            x = torch.gather(feats_t, 1, idx_cluster_t) # [B,L,C,D]
            x = torch.cat([ori_x[:,0,:,:].unsqueeze(1), x], dim=1)
            
        #- Self-attention
        new_x1, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )

        #- Channel clustering (all layer)
        idx_cluster, cluster_num = cluster_dpc_knn(ori_x[:,0,:,:], cluster_num=cluster_channel, k=5) # [B,D,D] -> [B,C]
        clustered_feats = merge_tokens(ori_x[:,:,:,:], idx_cluster, cluster_num) # [B,L+1,cluster,D]
        idx_cluster = idx_cluster.unsqueeze(2).unsqueeze(1).repeat(1, x.shape[1], 1, x.shape[-1])
        clustered_feats = torch.gather(clustered_feats, 2, idx_cluster) # [B,L,C,D]

        #- Self-attention
        x = clustered_feats.permute(0,2,1,3) # [bs,4,101,128]
        new_x = self.attention_c(
            x, x, x,
            attn_mask=attn_mask
        )
        new_x2 = new_x.permute(0,2,1,3)

        # Element-wise multiplication (feature fusion)
        new_x = new_x1 * new_x2

        #- FFN
        x = ori_x + self.dropout(new_x)
        y = x = self.norm1(x) 
        y = self.dropout(self.activation(self.conv3(y)))
        y = self.dropout(self.conv4(y))
        y = self.norm2(x + y)

        return y, attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, cluster_temporal, cluster_channel, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

        self.cluster_temporal = cluster_temporal
        self.cluster_channel = cluster_channel

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for num, attn_layer in enumerate(self.attn_layers):
            x, series, prior, sigma = attn_layer(x, layer=num, attn_mask=attn_mask, cluster_temporal=self.cluster_temporal, cluster_channel=self.cluster_channel)
            series_list.append(series[:,:,:,1:,1:])
            prior_list.append(prior[:,:,:,1:,1:])
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list

class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, cluster_temporal, cluster_channel, d_model=128, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.enc_in = enc_in
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    AttentionLayer_new(
                        AnomalyAttention_new(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            cluster_temporal = cluster_temporal,
            cluster_channel = cluster_channel,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, 1, bias=True)
        self.cls_tokens = nn.Parameter(torch.randn(1,1,enc_in, d_model)) # [1,1,25,dim]

    def forward(self, x):
        #- embedding
        enc_out = self.embedding(x) # [B,L,C,D]
        
        #- encoder
        enc_out, series, prior, sigmas = self.encoder(enc_out) 

        #- inverse embedding
        enc_out = enc_out[:,1:,:,:]
        enc_out = self.projection(enc_out).squeeze(3) # [B,L,C]

        if self.output_attention:
            return enc_out, series, prior, None
        else:
            return enc_out  # [B,L,D]


# ------------------------------
# https://github.com/zengwang430521/TCFormer

def merge_tokens(token_dict, idx_cluster, cluster_num, token_weight=None):
    """Merge tokens in the same cluster to a single cluster.
    Implemented by torch.index_add(). Flops: B*N*(C+2)
    Return:
        out_dict (dict): dict for output token information
    Args:
        token_dict (dict): dict for input token information
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): cluster number
        token_weight (Tensor[B, N, 1]): weight for each token.
    """

    x = token_dict

    B, L, N, C = x.shape 
    if token_weight is None:
        token_weight = x.new_ones(B, N, 1) 

    idx_batch = torch.arange(B, device=x.device)[:, None] 
    idx = idx_cluster + idx_batch * cluster_num 

    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                          source=token_weight.reshape(B * N, 1)) 
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]
    norm_weight = norm_weight.unsqueeze(1).repeat(1,L,1,1)

    # average token features
    x_merged = x.new_zeros(B*cluster_num, L, C) 
    source = x * norm_weight
    x_merged.index_add_(dim=0, index=idx.reshape(B*N),
                        source=source.reshape(B*N, L, C).type(x.dtype))
    x_merged = x_merged.reshape(B, L, cluster_num, C)
    return x_merged 



def cluster_dpc_knn(token_dict, cluster_num, k=5, token_mask=None):
    """Cluster tokens with DPC-KNN algorithm.
    Return:
        idx_cluster (Tensor[B, N]): cluster index of each token.
        cluster_num (int): actual cluster number. The same with
            input cluster number
    Args:
        token_dict (dict): dict for token information
        cluster_num (int): cluster number
        k (int): number of the nearest neighbor used for local density.
        token_mask (Tensor[B, N]): mask indicate the whether the token is
            padded empty token. Non-zero value means the token is meaningful,
            zero value means the token is an empty token. If set to None, all
            tokens are regarded as meaningful.
    """
    with torch.no_grad():
        x = token_dict#['x']
        B, N, C = x.shape

        dist_matrix = torch.cdist(x, x) / (C ** 0.5)

        if token_mask is not None:
            token_mask = token_mask > 0
            # in order to not affect the local density, the distance between empty tokens
            # and any other tokens should be the maximal distance.
            dist_matrix = dist_matrix * token_mask[:, None, :] + \
                          (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        # get local density
        dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)

        density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
        # add a little noise to ensure no tokens have the same density.
        density = density + torch.rand(
            density.shape, device=density.device, dtype=density.dtype) * 1e-6

        if token_mask is not None:
            # the density of empty token should be 0
            density = density * token_mask

        # get distance indicator
        mask = density[:, None, :] > density[:, :, None]
        mask = mask.type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        # select clustering center according to score
        score = dist * density
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        # assign tokens to the nearest center
        dist_matrix = index_points(dist_matrix, index_down)

        idx_cluster = dist_matrix.argmin(dim=1)

        # make sure cluster center merge to itself
        idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return idx_cluster, cluster_num


def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]
    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
