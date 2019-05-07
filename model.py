import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F

class BiLSTMEncoder(nn.Module):
    def __init__(self, embed_dim,hidden_dim,layers,dropout_lstm, dropout_input=0.2):
        super(BiLSTMEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dropout_input = dropout_input
        self.dropout_lstm = dropout_lstm
        self.rnn = nn.LSTM(input_size=embed_dim, #1024
                           hidden_size=hidden_dim, #hyper
                           num_layers=layers, #1
                           dropout=dropout_lstm, 
                           bidirectional=True,
                           batch_first=True)
        self.input_dropout = nn.Dropout(dropout_input)
    def forward(self, inputs, lengths):
        batch_size = inputs.size()[1]
        embedded_input = self.input_dropout(inputs)
        (sorted_input, sorted_lengths, input_unsort_indices, _) = sort_batch_by_length(inputs, lengths)
        packed_input = pack_padded_sequence(sorted_input, sorted_lengths.data.tolist(), batch_first=True)
        embedding, _ = self.rnn(packed_input)
        embedding, _ = pad_packed_sequence(embedding, batch_first=True)
        embedding = embedding[input_unsort_indices]
        # Tried batch_norm
        #embedding = self.batch_norm(embedding)
        return embedding

class Metaphor(nn.Module):
    def __init__(self, dropout, num_classes, hidden_dim):
        super(Metaphor, self).__init__()
        self.fcl = nn.Linear(hidden_dim*2, num_classes)
        self.linear_dropout = nn.Dropout(dropout)
    
    def forward(self, output):

        input_encoding = self.linear_dropout(output)
        unnormalized_output = self.fcl(input_encoding)
        normalized_output = F.log_softmax(unnormalized_output, dim=-1)

        return normalized_output

class MainModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, layers, dropout_lstm, dropout_input, dropout_FC, num_classes):
        super(MainModel, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dropout_input = dropout_input
        self.dropout_FC = dropout_FC
        self.dropout_lstm = dropout_lstm

        self.embbedding = BiLSTMEncoder(embed_dim,hidden_dim,layers,dropout_lstm,dropout_input)
        self.metafor_classifier = Metaphor(dropout_FC, num_classes, hidden_dim)

    def forward(self, inputs, lengths):

        out_embedding = self.embbedding.forward(inputs, lengths)
        normalized_output = self.metafor_classifier(out_embedding)

        return normalized_output 

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