class soft_LSTM(nn.Module):
    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_dim, #512
                    hidden_size=num_hidden, #hyper
                    num_layers=layers, #1
                    dropout=dropout_lstm, 
                    bidirectional=True,
                    batch_first=True)
        self.LSTM_m = NaiveLSTM(input_sz = input_dim, hidden_sz = num_hidden)
        self.LSTM_n = NaiveLSTM(input_sz = input_dim, hidden_sz = num_hidden)
        self.device = device
        self.num_hidden = num_hidden
    def forward(self, x):
        output_bi, (hidden_bi, c_bi) = self.rnn(x)
        meta = self.LSTM_m(x = x, bi_direc = hidden_bi)
        hyper = self.LSTM_n(x = x, bi_direc = hidden_bi)
        return meta, hyper

from enum import IntEnum

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2
 
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
                init_states: Optional[Tuple[torch.Tensor]]=None,
                lengths: torch.Tensor,
                h_ts: torch.Tensor):# -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
            g_ts = torch.tanh(x_t @ self.W_ig + h_ts @ self.W_hgs + self.b_gs)

            cs_t = torch.tanh(x_t @ self.W_igc + h_t @ self.W_hgc @ g_t +  + h_ts @ self.W_hgc @ g_ts + self.b_gc)

            o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_o)

            c_t = f_t * c_t + i_t * cs_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))
            if batch_sizes
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h_t, c_t)