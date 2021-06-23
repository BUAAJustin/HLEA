# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, copy, time

class CNNmodel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, dropout, gpu=True):
        super(CNNmodel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.gpu = gpu

        self.cnn_layer0 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=1, padding=0)
        self.cnn_layers = [nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1) for i in range(self.num_layer-1)]
        self.drop = nn.Dropout(dropout)

        if self.gpu:
            self.cnn_layer0 = self.cnn_layer0.cuda()
            for i in range(self.num_layer-1):
                self.cnn_layers[i] = self.cnn_layers[i].cuda()

    def forward(self, input_feature):

        batch_size = input_feature.size(0)
        seq_len = input_feature.size(1)

        input_feature = input_feature.transpose(2,1).contiguous()
        cnn_output = self.cnn_layer0(input_feature)  #(b,h,l)
        cnn_output = self.drop(cnn_output)
        cnn_output = torch.tanh(cnn_output)

        for layer in range(self.num_layer-1):
            cnn_output = self.cnn_layers[layer](cnn_output)
            cnn_output = self.drop(cnn_output)
            cnn_output = torch.tanh(cnn_output)

        cnn_output = cnn_output.transpose(2,1).contiguous()
        return cnn_output



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)  ## (b,h,l,d) * (b,h,d,l)
    if mask is not None:
        # scores = scores.masked_fill(mask == 0, -1e9)
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn   ##(b,h,l,l) * (b,h,l,d) = (b,h,l,d)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + autograd.Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class AttentionModel(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, d_input, d_model, d_ff, head, num_layer, dropout):
        super(AttentionModel, self).__init__()
        c = copy.deepcopy
        # attn0 = MultiHeadedAttention(head, d_input, d_model)
        attn = MultiHeadedAttention(head, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        # position = PositionalEncoding(d_model, dropout)
        # layer0 = EncoderLayer(d_model, c(attn0), c(ff), dropout)
        layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
        self.layers = clones(layer, num_layer)
        # layerlist = [copy.deepcopy(layer0),]
        # for _ in range(num_layer-1):
        #     layerlist.append(copy.deepcopy(layer))
        # self.layers = nn.ModuleList(layerlist)
        self.norm = LayerNorm(layer.size)
        self.posi = PositionalEncoding(d_model, dropout)
        self.input2model = nn.Linear(d_input, d_model)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        # x: embedding (b,l,we)
        x = self.posi(self.input2model(x))
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)




class NERmodel(nn.Module):

    def __init__(self, model_type, input_dim, hidden_dim, num_layer, dropout=0.5, gpu=True, biflag=True):
        super(NERmodel, self).__init__()
        self.model_type = model_type

        if self.model_type == 'lstm':
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layer, batch_first=True, bidirectional=biflag)
            self.drop = nn.Dropout(dropout)

        if self.model_type == 'lstm_cnn':
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layer, batch_first=True, bidirectional=biflag)
            self.drop = nn.Dropout(dropout)
            self.cnn = CNNmodel(200, 100, 1, dropout, gpu)

        ## attention model
        if self.model_type == 'transformer':
            self.attention_model = AttentionModel(d_input=input_dim, d_model=hidden_dim, d_ff=2*hidden_dim, head=4, num_layer=num_layer, dropout=dropout)
            for p in self.attention_model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)


    def forward(self, input, mask=None):

        if self.model_type == 'lstm':
            hidden = None
            feature_out, hidden = self.lstm(input, hidden)
            feature_out_f = self.drop(feature_out)

        if self.model_type == 'lstm_cnn':
            hidden = None
            feature_out, hidden = self.lstm(input, hidden)
            feature_out_d = self.drop(feature_out)
            feature_out_f = self.cnn(feature_out_d)

        if self.model_type == 'transformer':
            feature_out_f = self.attention_model(input, mask)

        return feature_out_f









class MyRNNBase(torch.nn.Module):
    """
    RNN with packed sequence and dropout, only one layer
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        dropout_p: dropout probability to input data, and also dropout along hidden layers
        enable_layer_norm: layer normalization

    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each elem
        ent in the batch.

    Outputs: output, last_state
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features `(h_t)` from the last layer of the RNN,
          for each t.
        - **last_state** (batch, hidden_size * num_directions): the final hidden state of rnn
    """

    def __init__(self, mode, input_size, hidden_size, bidirectional, dropout_p, enable_layer_norm=False):
        super(MyRNNBase, self).__init__()
        self.mode = mode
        self.enable_layer_norm = enable_layer_norm

        if mode == 'LSTM':
            self.hidden = torch.nn.LSTM(input_size=input_size,
                                        hidden_size=hidden_size,
                                        num_layers=1,
                                        bidirectional=bidirectional)
        elif mode == 'GRU':
            self.hidden = torch.nn.GRU(input_size=input_size,
                                       hidden_size=hidden_size,
                                       num_layers=1,
                                       bidirectional=bidirectional)
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.dropout = torch.nn.Dropout(p=dropout_p)

        if enable_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(input_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, v, mask):
        # layer normalization
        if self.enable_layer_norm:
            seq_len, batch, input_size = v.shape
            v = v.view(-1, input_size)
            v = self.layer_norm(v)
            v = v.view(seq_len, batch, input_size)

        # get sorted v
        #lengths = mask.eq(1).long().sum(1)
        #lengths_sort, idx_sort = torch.sort(lengths, dim=0, descending=True)
        #_, idx_unsort = torch.sort(idx_sort, dim=0)

        #v_sort = v.index_select(1, idx_sort)

        #v_pack = torch.nn.utils.rnn.pack_padded_sequence(v_sort, lengths_sort)
        #v_dropout = self.dropout.forward(v_pack.data)
        #v_pack_dropout = torch.nn.utils.rnn.PackedSequence(v_dropout, v_pack.batch_sizes)

        #o_pack_dropout, _ = self.hidden.forward(v_pack_dropout)
        #o, _ = torch.nn.utils.rnn.pad_packed_sequence(o_pack_dropout)

        # unsorted o
        #o_unsort = o.index_select(1, idx_unsort)  # Note that here first dim is seq_len

        # get the last time state
        #len_idx = (lengths - 1).view(-1, 1).expand(-1, o_unsort.size(2)).unsqueeze(0)
        #o_last = o_unsort.gather(0, len_idx)
        #o_last = o_last.squeeze(0)

        hidden = None
        feature_out, hidden = self.hidden(v, hidden)
        o_unsort = self.dropout(feature_out)
        return o_unsort, hidden


def compute_mask(v, padding_idx=0):
    """
    compute mask on given tensor v
    :param v:
    :param padding_idx:
    :return:
    """
    mask = torch.ne(v, padding_idx).float()
    return mask

def masked_softmax(x, m=None, dim=-1):
    """
    Softmax with mask
    :param x:
    :param m:
    :param dim:
    :return:
    """
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=dim, keepdim=True) + 1e-6)
    return softmax

def masked_flip(vin, mask, flip_dim=0):
    """
    flip a tensor
    :param vin: (..., batch, ...), batch should on dim=1, input batch with padding values
    :param mask: (batch, seq_len), show whether padding index
    :param flip_dim: dim to flip on
    :return:
    """
    length = mask.data.eq(1).long().sum(1)
    batch_size = vin.shape[1]

    flip_list = []
    for i in range(batch_size):
        cur_tensor = vin[:, i, :]
        cur_length = length[i]

        idx = list(range(cur_length - 1, -1, -1)) + list(range(cur_length, vin.shape[flip_dim]))
        idx = vin.new_tensor(idx, dtype=torch.long)

        cur_inv_tensor = cur_tensor.unsqueeze(1).index_select(flip_dim, idx).squeeze(1)
        flip_list.append(cur_inv_tensor)
    inv_tensor = torch.stack(flip_list, dim=1)

    return inv_tensor

class MatchRNNAttention(torch.nn.Module):
    r"""
    attention mechanism in match-rnn
    Args:
        - input_size: The number of expected features in the input Hp and Hq
        - hidden_size: The number of features in the hidden state Hr

    Inputs:
        Hpi(batch, input_size): a context word encoded
        Hq(question_len, batch, input_size): whole question encoded
        Hr_last(batch, hidden_size): last lstm hidden output

    Outputs:
        alpha(batch, question_len): attention vector
    """

    def __init__(self, hp_input_size, hq_input_size, hidden_size):
        super(MatchRNNAttention, self).__init__()

        self.linear_wq = torch.nn.Linear(hq_input_size, hidden_size)
        self.linear_wp = torch.nn.Linear(hp_input_size, hidden_size)
        self.linear_wr = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_wg = torch.nn.Linear(hidden_size, 1)

    def forward(self, Hpi, Hq, Hr_last, Hq_mask):
        wq_hq = self.linear_wq(Hq)  # (question_len, batch, hidden_size)
        wp_hp = self.linear_wp(Hpi).unsqueeze(0)  # (1, batch, hidden_size)
        wr_hr = self.linear_wr(Hr_last).unsqueeze(0)  # (1, batch, hidden_size)
        G = F.tanh(wq_hq + wp_hp + wr_hr)  # (question_len, batch, hidden_size), auto broadcast
        wg_g = self.linear_wg(G) \
            .squeeze(2) \
            .transpose(0, 1)  # (batch, question_len)
        alpha = masked_softmax(wg_g, m=Hq_mask, dim=1)  # (batch, question_len)
        return alpha

class UniMatchRNN(torch.nn.Module):
    r"""
    interaction context and question with attention mechanism, one direction, using LSTM cell
    Args:
        - input_size: The number of expected features in the input Hp and Hq
        - hidden_size: The number of features in the hidden state Hr

    Inputs:
        Hp(context_len, batch, input_size): context encoded
        Hq(question_len, batch, input_size): question encoded

    Outputs:
        Hr(context_len, batch, hidden_size): question-aware context representation
        alpha(batch, question_len, context_len): used for visual show
    """

    def __init__(self, mode, hp_input_size, hq_input_size, hidden_size, gated_attention, enable_layer_norm):
        super(UniMatchRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gated_attention = gated_attention
        self.enable_layer_norm = enable_layer_norm
        rnn_in_size = hp_input_size + hq_input_size

        self.attention = MatchRNNAttention(hp_input_size, hq_input_size, hidden_size)

        if self.gated_attention:
            self.gated_linear = torch.nn.Linear(rnn_in_size, rnn_in_size)

        if self.enable_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(rnn_in_size)

        self.mode = mode
        if mode == 'LSTM':
            self.hidden_cell = torch.nn.LSTMCell(input_size=rnn_in_size, hidden_size=hidden_size)
        elif mode == 'GRU':
            self.hidden_cell = torch.nn.GRUCell(input_size=rnn_in_size, hidden_size=hidden_size)
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, Hp, Hq, Hq_mask):
        batch_size = Hp.shape[1]
        context_len = Hp.shape[0]

        # init hidden with the same type of input data
        h_0 = Hq.new_zeros(batch_size, self.hidden_size)
        hidden = [(h_0, h_0)] if self.mode == 'LSTM' else [h_0]
        vis_para = {}
        vis_alpha = []
        vis_gated = []

        for t in range(context_len):
            cur_hp = Hp[t, ...]  # (batch, input_size)
            attention_input = hidden[t][0] if self.mode == 'LSTM' else hidden[t]

            alpha = self.attention.forward(cur_hp, Hq, attention_input, Hq_mask)  # (batch, question_len)
            vis_alpha.append(alpha)

            question_alpha = torch.bmm(alpha.unsqueeze(1), Hq.transpose(0, 1)) \
                .squeeze(1)  # (batch, input_size)
            cur_z = torch.cat([cur_hp, question_alpha], dim=1)  # (batch, rnn_in_size)

            # gated
            if self.gated_attention:
                gate = F.sigmoid(self.gated_linear.forward(cur_z))
                vis_gated.append(gate.squeeze(-1))
                cur_z = gate * cur_z

            # layer normalization
            if self.enable_layer_norm:
                cur_z = self.layer_norm(cur_z)  # (batch, rnn_in_size)

            cur_hidden = self.hidden_cell.forward(cur_z, hidden[t])  # (batch, hidden_size), when lstm output tuple
            hidden.append(cur_hidden)

        #vis_para['gated'] = torch.stack(vis_gated, dim=-1)  # (batch, context_len)
        vis_para['alpha'] = torch.stack(vis_alpha, dim=2)  # (batch, question_len, context_len)

        hidden_state = list(map(lambda x: x[0], hidden)) if self.mode == 'LSTM' else hidden
        result = torch.stack(hidden_state[1:], dim=0)  # (context_len, batch, hidden_size)
        return result, vis_para


class MatchRNN(torch.nn.Module):
    r"""
    interaction context and question with attention mechanism
    Args:
        - input_size: The number of expected features in the input Hp and Hq
        - hidden_size: The number of features in the hidden state Hr
        - bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        - gated_attention: If ``True``, gated attention used, see more on R-NET

    Inputs:
        Hp(context_len, batch, input_size): context encoded
        Hq(question_len, batch, input_size): question encoded
        Hp_mask(batch, context_len): each context valued length without padding values
        Hq_mask(batch, question_len): each question valued length without padding values

    Outputs:
        Hr(context_len, batch, hidden_size * num_directions): question-aware context representation
    """

    def __init__(self, mode, hp_input_size, hq_input_size, hidden_size, bidirectional, gated_attention,
                 dropout_p, enable_layer_norm):
        super(MatchRNN, self).__init__()
        self.bidirectional = bidirectional
        self.num_directions = 1 if bidirectional else 2

        self.left_match_rnn = UniMatchRNN(mode, hp_input_size, hq_input_size, hidden_size, gated_attention,
                                          enable_layer_norm)
        if bidirectional:
            self.right_match_rnn = UniMatchRNN(mode, hp_input_size, hq_input_size, hidden_size, gated_attention,
                                               enable_layer_norm)

        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, Hp, Hp_mask, Hq, Hq_mask):
        Hp = self.dropout(Hp)
        Hq = self.dropout(Hq)

        left_hidden, left_para = self.left_match_rnn.forward(Hp, Hq, Hq_mask)
        rtn_hidden = left_hidden
        rtn_para = {'left': left_para}

        if self.bidirectional:
            Hp_inv = masked_flip(Hp, Hp_mask, flip_dim=0)
            right_hidden_inv, right_para_inv = self.right_match_rnn.forward(Hp_inv, Hq, Hq_mask)

            # flip back to normal sequence
            right_alpha_inv = right_para_inv['alpha']
            right_alpha_inv = right_alpha_inv.transpose(0, 1)  # make sure right flip
            right_alpha = masked_flip(right_alpha_inv, Hp_mask, flip_dim=2)
            right_alpha = right_alpha.transpose(0, 1)

            #right_gated_inv = right_para_inv['gated']
            #right_gated_inv = right_gated_inv.transpose(0, 1)
            #right_gated = masked_flip(right_gated_inv, Hp_mask, flip_dim=2)
            #right_gated = right_gated.transpose(0, 1)

            right_hidden = masked_flip(right_hidden_inv, Hp_mask, flip_dim=0)

            #rtn_para['right'] = {'alpha': right_alpha, 'gated': right_gated}
            rtn_hidden = torch.cat((left_hidden, right_hidden), dim=2)

        real_rtn_hidden = Hp_mask.transpose(0, 1).unsqueeze(2) * rtn_hidden
        last_hidden = rtn_hidden[-1, :]

        return real_rtn_hidden, last_hidden, rtn_para