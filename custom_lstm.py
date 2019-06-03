import torch.nn as nn
from torch.nn import Parameter
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence

class BiLSTM_SOFT_Encoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, layers,dropout_lstm, dropout_input=0.2):
        super(BiLSTM_SOFT_Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dropout_input = dropout_input
        self.dropout_lstm = dropout_lstm
        self.rnn = nn.LSTM(input_size=embed_dim, #512
                    hidden_size=hidden_dim, #hyper
                    num_layers=layers, #1
                    dropout=dropout_lstm, 
                    bidirectional=True,
                    batch_first=True)
        self.input_dropout = nn.Dropout(dropout_input)
        self.LSTM_m = gatedLSTM(input_sz = embed_dim, hidden_sz = hidden_dim)
        self.LSTM_n = gatedLSTM(input_sz = embed_dim, hidden_sz = hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, inputs, lengths, two_tasks = True):
        embedded_input = self.input_dropout(inputs)
        (sorted_input, sorted_lengths, input_unsort_indices, _) = sort_batch_by_length(embedded_input, lengths)
        packed_input = pack_padded_sequence(sorted_input, sorted_lengths.data.tolist(), batch_first=True)
  
        output_bi, (hidden_bi, c_bi) = self.rnn(packed_input)
        output_bi, _ = pad_packed_sequence(output_bi, batch_first=True)
        output_bi = output_bi[input_unsort_indices]
        meta, _ = self.LSTM_m(x = inputs, h_ts = output_bi)
        if two_tasks:
          hyper, _ = self.LSTM_n(x = inputs, h_ts = output_bi)
        else:
          hyper = None
        return meta, hyper

class gatedLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super(gatedLSTM, self).__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        # input gate
        self.W_ii = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hi = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = Parameter(torch.Tensor(hidden_sz))
        # forget gate
        self.W_if = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hf = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = Parameter(torch.Tensor(hidden_sz))

        self.W_ig = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hg = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.W_hgs = Parameter(torch.Tensor(hidden_sz*2, hidden_sz))
        self.b_g = Parameter(torch.Tensor(hidden_sz))
        self.b_gs = Parameter(torch.Tensor(hidden_sz))

        self.W_igc = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hgc = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.W_hgcs = Parameter(torch.Tensor(hidden_sz*2, hidden_sz))
        self.b_gc = Parameter(torch.Tensor(hidden_sz))
        # output gate
        self.W_io = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_ho = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = Parameter(torch.Tensor(hidden_sz))
         
        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
         
    def forward(self, x: torch.Tensor, 
                h_ts: torch.Tensor,
                init_states: torch.Tensor=None
               ):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = torch.zeros([bs,self.hidden_size]).to(x.device), torch.zeros([bs,self.hidden_size]).to(x.device)
        else:
            h_t, c_t = init_states
        for t in range(seq_sz): # iterate over the time steps
            x_t = x[:, t, :]
            i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_f)

            g_t = torch.sigmoid(x_t @ self.W_ig + h_t @ self.W_hg + self.b_g)

            # Shared Layer hidden state gate:
            g_ts = torch.sigmoid(x_t @ self.W_ig + h_ts[:, t, :] @ self.W_hgs + self.b_gs)
            # Compute combined cell-state 
            cs_t = torch.tanh(x_t @ self.W_igc + g_t *(h_t @ self.W_hgc ) + g_ts *(h_ts[:, t, :] @ self.W_hgcs) + self.b_gc)
            c_t = f_t * c_t + i_t * cs_t

            o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_o)

            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

def sort_batch_by_length(tensor: torch.Tensor, sequence_lengths: torch.Tensor):
    """
    Sort a batch first tensor by some specified lengths.
    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A batch first Pytorch tensor.
    sequence_lengths : torch.LongTensor, required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.
    Returns
    -------
    sorted_tensor : torch.FloatTensor
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    sorted_sequence_lengths : torch.LongTensor
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : torch.LongTensor
        Indices into the sorted_tensor such that
        ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
    permutation_index : torch.LongTensor
        The indices used to sort the tensor. This is useful if you want to sort many
        tensors using the same ordering.
    """

    if not isinstance(tensor, torch.Tensor) or not isinstance(sequence_lengths, torch.Tensor):
        raise ConfigurationError("Both the tensor and sequence lengths must be torch.Tensors.")

    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)

    index_range = torch.arange(0, len(sequence_lengths), device=sequence_lengths.device)
    # This is the equivalent of zipping with index, sorting by the original
    # sequence lengths and returning the now sorted indices.
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index



#from enum import IntEnum
'''class BiLSTM_SOFT_Encoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, layers,dropout_lstm, dropout_input=0.2):
        super(BiLSTM_SOFT_Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dropout_input = dropout_input
        self.dropout_lstm = dropout_lstm
        self.rnn = nn.LSTM(input_size=embed_dim, #512
                    hidden_size=hidden_dim, #hyper
                    num_layers=layers, #1
                    dropout=dropout_lstm, 
                    bidirectional=True,
                    batch_first=True)
        self.input_dropout = nn.Dropout(dropout_input)
        self.LSTM_m = NaiveLSTM(input_sz = embed_dim, hidden_sz = hidden_dim)
        self.LSTM_n = NaiveLSTM(input_sz = embed_dim, hidden_sz = hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, inputs, lengths, is_doc):
        embedded_input = self.input_dropout(inputs)

        (sorted_input, sorted_lengths, input_unsort_indices, _) = sort_batch_by_length(embedded_input, lengths)
        packed_input = pack_padded_sequence(sorted_input, sorted_lengths.data.tolist(), batch_first=True)
        # if torch.cuda.is_available():
        #     packed_input = packed_input.to(device=torch.device('cuda'))
        embedding, _ = self.rnn(packed_input)
        
        output_bi, (hidden_bi, c_bi) = self.rnn(packed_input)
        meta, _ = self.LSTM_m(x = packed_input, h_ts = output_bi)
        if is_doc:
          hyper, _ = self.LSTM_n(x = packed_input, h_ts = output_bi)
          embedding_hyper, _ = pad_packed_sequence(hyper, batch_first=True)
          embedding_hyper = embedding_hyper[input_unsort_indices]
        else:
          embedding_hyper = None
        embedding_meta, _ = pad_packed_sequence(meta, batch_first=True)
        embedding_meta = embedding_meta[input_unsort_indices]
        return embedding_meta, embedding_hyper

class NaiveLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        # input gate
        self.W_ii = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hi = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = Parameter(torch.Tensor(hidden_sz))
        # forget gate
        self.W_if = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hf = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = Parameter(torch.Tensor(hidden_sz))
        # ???
        self.W_ig = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hg = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.W_hgs = Parameter(torch.Tensor(hidden_sz*2, hidden_sz))
        self.b_g = Parameter(torch.Tensor(hidden_sz))
        self.b_gs = Parameter(torch.Tensor(hidden_sz))

        self.W_igc = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hgc = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.W_hgcs = Parameter(torch.Tensor(hidden_sz*2, hidden_sz))
        self.b_gc = Parameter(torch.Tensor(hidden_sz))

        # output gate
        self.W_io = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_ho = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = Parameter(torch.Tensor(hidden_sz))
         
        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
         
    def forward(self, x: torch.Tensor, 
                h_ts: torch.Tensor,
                init_states: [torch.Tensor,torch.Tensor]=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        seq_sz, _ = x.data.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = torch.zeros(self.hidden_size).to(x.data.device), torch.zeros(self.hidden_size).to(x.data.device)
        else:
            h_t, c_t = init_states
        batches = x.batch_sizes
        #print(batches)
        prev_batch = 0
        print(h_ts.data.shape,'bilstm')
        for batch in batches:
            #print('end_batch')
            #print(batch)
            h_t, c_t = torch.zeros(self.hidden_size).to(x.data.device), torch.zeros(self.hidden_size).to(x.data.device)
            for t in range(batch): # iterate over the time steps
                t_batch = t + prev_batch
                #print(t_batch)
                x_t = x.data[t_batch, :]
                i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_i)
                f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_f)

                g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_g)
                g_ts = torch.tanh(x_t @ self.W_ig + h_ts.data[t_batch] @ self.W_hgs + self.b_gs)

                cs_t = torch.tanh(x_t @ self.W_igc + h_t @ self.W_hgc @ g_t + h_ts.data[t_batch] @ self.W_hgcs @ g_ts + self.b_gc)

                o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_o)

                c_t = f_t * c_t + i_t * cs_t
                h_t = o_t * torch.tanh(c_t)
                hidden_seq.append(h_t.unsqueeze(Dim.batch))
            prev_batch = t_batch+1
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        #hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        output = PackedSequence(hidden_seq, x.batch_sizes)
        return output, (h_t, c_t)


class NaiveBatchLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        # input gate
        self.W_ii = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hi = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = Parameter(torch.Tensor(hidden_sz))
        # forget gate
        self.W_if = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hf = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = Parameter(torch.Tensor(hidden_sz))
        # ???
        self.W_ig = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hg = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.W_hgs = Parameter(torch.Tensor(hidden_sz*2, hidden_sz))
        self.b_g = Parameter(torch.Tensor(hidden_sz))
        self.b_gs = Parameter(torch.Tensor(hidden_sz))

        self.W_igc = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hgc = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.W_hgcs = Parameter(torch.Tensor(hidden_sz*2, hidden_sz))
        self.b_gc = Parameter(torch.Tensor(hidden_sz))

        # output gate
        self.W_io = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_ho = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = Parameter(torch.Tensor(hidden_sz))
         
        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
         
    def forward(self, x: torch.Tensor, 
                h_ts: torch.Tensor,
                init_states: [torch.Tensor,torch.Tensor]=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        seq_sz, _ = x.data.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = torch.zeros(self.hidden_size).to(x.data.device), torch.zeros(self.hidden_size).to(x.data.device)
        else:
            h_t, c_t = init_states
        batches = x.batch_sizes
        #print(batches)
        prev_batch = 0
        print(h_ts.data.shape,'bilstm')
        for batch in batches:
            #print('end_batch')
            #print(batch)
            h_t, c_t = torch.zeros(self.hidden_size).to(x.data.device), torch.zeros(self.hidden_size).to(x.data.device)
            for t in range(batch): # iterate over the time steps
                t_batch = t + prev_batch
                #print(t_batch)
                x_t = x.data[t_batch, :]
                i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_i)
                f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_f)

                g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_g)
                g_ts = torch.tanh(x_t @ self.W_ig + h_ts.data[t_batch] @ self.W_hgs + self.b_gs)

                cs_t = torch.tanh(x_t @ self.W_igc + h_t @ self.W_hgc @ g_t + h_ts.data[t_batch] @ self.W_hgcs @ g_ts + self.b_gc)

                o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_o)

                c_t = f_t * c_t + i_t * cs_t
                h_t = o_t * torch.tanh(c_t)
                hidden_seq.append(h_t.unsqueeze(Dim.batch))
            prev_batch = t_batch+1
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        #hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        output = PackedSequence(hidden_seq, x.batch_sizes)
        return output, (h_t, c_t)

class NaiveLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        # input gate
        self.W_ii = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hi = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = Parameter(torch.Tensor(hidden_sz))
        # forget gate
        self.W_if = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hf = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = Parameter(torch.Tensor(hidden_sz))
        # ???
        self.W_ig = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hg = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.W_hgs = Parameter(torch.Tensor(hidden_sz*2, hidden_sz))
        self.b_g = Parameter(torch.Tensor(hidden_sz))
        self.b_gs = Parameter(torch.Tensor(hidden_sz))

        self.W_igc = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hgc = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.W_hgcs = Parameter(torch.Tensor(hidden_sz*2, hidden_sz))
        self.b_gc = Parameter(torch.Tensor(hidden_sz))

        # output gate
        self.W_io = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_ho = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = Parameter(torch.Tensor(hidden_sz))
         
        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
         
    def forward(self, x: torch.Tensor, 
                h_ts: torch.Tensor,
                init_states: Optional[Tuple[torch.Tensor]]=None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = torch.zeros(self.hidden_size).to(x.device), torch.zeros(self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states
        for t in range(seq_sz): # iterate over the time steps
            x_t = x[:, t, :]
            i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_f)

            g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_g)
            g_ts = torch.tanh(x_t @ self.W_ig + h_ts[:, t, :] @ self.W_hgs + self.b_gs)

            cs_t = torch.tanh(x_t @ self.W_igc + h_t @ self.W_hgc @ g_t + h_ts[:, t, :] @ self.W_hgcs @ g_ts + self.b_gc)

            o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_o)

            c_t = f_t * c_t + i_t * cs_t

            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h_t, c_t)'''