import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Optional, List


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, output_dim: int, dropout: float, batch_first: bool = True) -> None:
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = head_dim

        self.in_proj_weight = nn.Parameter(torch.empty((3 * head_dim * num_heads, embed_dim), dtype=torch.float32))
        self.in_proj_bias = nn.Parameter(torch.empty((3 * head_dim * num_heads,), dtype=torch.float32))
        self.out_proj_weight = nn.Parameter(torch.empty((output_dim, num_heads * head_dim), dtype=torch.float32))

        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_normal_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.xavier_normal_(self.out_proj_weight)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        if self.batch_first:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
            
        attn_output = MultiHeadAttention.multi_head_attention_forward(
            query, key, value, self.head_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            dropout_p=self.dropout, 
            out_proj_weight=self.out_proj_weight,
            out_proj_bias=None,
            training=self.training, 
        )

        if self.batch_first:
            return attn_output.transpose(1, 0)
        else:
            return attn_output
    
    @staticmethod
    def multi_head_attention_forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        head_dim: int,
        num_heads: int,
        in_proj_weight: Optional[torch.Tensor],
        in_proj_bias: Optional[torch.Tensor],
        dropout_p: float,
        out_proj_weight: torch.Tensor,
        out_proj_bias: Optional[torch.Tensor],
        training: bool = True,
    ) -> torch.Tensor:
        tgt_len, bsz, _ = query.shape
        src_len, _, _ = key.shape

        q, k, v = MultiHeadAttention._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

        q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

        if not training:
            dropout_p = 0.0

        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)

        attn_output = F.scaled_dot_product_attention(q, k, v, None, dropout_p, False)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, num_heads * head_dim)

        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        return attn_output

    @staticmethod
    def _in_projection_packed(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        E = w.size(0) // 3
        if k is v:
            if q is k:
                # self-attention
                proj = F.linear(q, w, b)
                # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
                proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
                return proj[0], proj[1], proj[2]
            else:
                # encoder-decoder attention
                w_q, w_kv = w.split([E, E * 2])
                if b is None:
                    b_q = b_kv = None
                else:
                    b_q, b_kv = b.split([E, E * 2])
                q_proj = F.linear(q, w_q, b_q)
                kv_proj = F.linear(k, w_kv, b_kv)
                # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as chunk()
                kv_proj = kv_proj.unflatten(-1, (2, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
                return (q_proj, kv_proj[0], kv_proj[1])
        else:
            w_q, w_k, w_v = w.chunk(3)
            if b is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = b.chunk(3)
            return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_attn_heads: int, attn_head_dim: int, feed_forward_dim: int, dropout: float):
        super(AttentionBlock, self).__init__()
        self.mha = MultiHeadAttention(embed_dim, num_attn_heads, attn_head_dim, embed_dim, dropout, True)
        self.ffn = FeedForwardNetwork(embed_dim, feed_forward_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(embed_dim, 1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, 1e-6)
    
    def forward(self, x, q):
        attn_output = self.mha(q, x, x)
        out1 = self.layernorm1(q + attn_output)
        ffn_output = self.ffn(out1)
        if self.training:
            ffn_output = self.dropout(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class ALECE(nn.Module):
    def __init__(
            self,
            n_bins: int,
            query_feature_dim: int,
            num_attn_heads: int,
            attn_head_dim: int,
            feed_forward_dim: int,
            num_self_attn_layers: int,
            num_cross_attn_layers: int,
            mlp_num_layers: int,
            mlp_hidden_dim: int,
            dropout: float
        ):
        super(ALECE, self).__init__()
        self.self_attn_layers = nn.ModuleList([
            AttentionBlock(n_bins, num_attn_heads, attn_head_dim, feed_forward_dim, dropout)
            for _ in range(num_self_attn_layers)
        ])
        self.proj = nn.Linear(n_bins, query_feature_dim)
        self.cross_attn_layers = nn.ModuleList([
            AttentionBlock(query_feature_dim, num_attn_heads, attn_head_dim, feed_forward_dim, dropout)
            for _ in range(num_cross_attn_layers)
        ])

        mlp_layers = []
        mlp_num_layers = max(1, mlp_num_layers)
        dims = [query_feature_dim] + [mlp_hidden_dim] * (mlp_num_layers - 1)
        for i in range(mlp_num_layers - 1):
            mlp_layers.append(nn.Linear(dims[i], dims[i+1]))
            mlp_layers.append(nn.ELU())
        mlp_layers.append(nn.Linear(dims[-1], 1))

        self.reg = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor, q: torch.Tensor):
        """
        :param x: Shape `(batch_size, num_attrs, n_bins)
        :param q: Shape `(batch_size, query_feature_dim)
        :return: Logorithm of cardinality, shape `(batch_size, )`
        """
        q = q.unsqueeze(1)
        for self_attn in self.self_attn_layers:
            x = self_attn(x, x)

        x = self.proj(x)

        for cross_attn in self.cross_attn_layers:
            q = cross_attn(x, q)

        preds = self.reg(q).view(-1)

        return preds


if __name__ == '__main__':
    batch_size = 64
    n_bins = 100
    n_attrs = 32
    query_feature_dim = 512

    x = torch.rand((batch_size, n_attrs, n_bins))
    q = torch.rand((batch_size, 1, query_feature_dim))
    model = ALECE(n_bins=n_bins, query_feature_dim=query_feature_dim, num_attn_heads=8,attn_head_dim=128, feed_forward_dim=128, num_self_attn_layers=6,
                  num_cross_attn_layers=6, mlp_num_layers=8, mlp_hidden_dim=128, dropout=0.1)
    output = model(x, q)
    print(output.shape)
