import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class BiLSTMEncoder(nn.Module):
    def __init__(self, embed_dim,hidden_dim,layers,dropout_lstm, dropout_input=0.2):
        super(BiLSTMEncoder, self).__init__()
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
    def forward(self, inputs, lengths):
        embedded_input = self.input_dropout(inputs)

        (sorted_input, sorted_lengths, input_unsort_indices, _) = sort_batch_by_length(embedded_input, lengths)
        packed_input = pack_padded_sequence(sorted_input, sorted_lengths.data.tolist(), batch_first=True)
        # if torch.cuda.is_available():
        #     packed_input = packed_input.to(device=torch.device('cuda'))
        embedding, _ = self.rnn(packed_input)
        embedding, _ = pad_packed_sequence(embedding, batch_first=True)
        embedding = embedding[input_unsort_indices]
        return embedding

class MetaphorModel(nn.Module):
    def __init__(self, hidden_dim, dropout_FC, num_classes):
        super(MetaphorModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout_FC = dropout_FC
        self.metafor_classifier = Metaphor(dropout_FC, num_classes, hidden_dim)
          
    def forward(self, out_embedding, lengths):
        normalized_output = self.metafor_classifier(out_embedding)
        return normalized_output

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

class Hyperpartisan(nn.Module):
    def __init__(self, dropout, num_classes, hidden_dim):
        super(Hyperpartisan, self).__init__()
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
        self.self_attention = SelfAttention(2*hidden_dim)
        self.embbedding = BiLSTMEncoder(embed_dim,hidden_dim,layers,dropout_lstm,dropout_input)
        self.metafor_classifier = Metaphor(dropout_FC, num_classes, hidden_dim)
        if torch.cuda.is_available():
            self.embbedding.to(device=torch.device('cuda'))
            self.embbedding.to(device=torch.device('cuda'))
    def forward(self, inputs, lengths):

        out_embedding = self.embbedding.forward(inputs, lengths)
        #out_attention, attention, weighted = self.self_attention(out_embedding, lengths)
        normalized_output = self.metafor_classifier(out_embedding)

        return normalized_output 

class ModelHyper(nn.Module):
    def __init__(self, embed_dim, hidden_dim, layers, dropout_lstm, dropout_input, dropout_FC, dropout_lstm_hyper,dropout_input_hyper,dropout_attention,num_classes):
        super(ModelHyper, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dropout_input = dropout_input
        self.dropout_FC = dropout_FC
        self.dropout_lstm = dropout_lstm
        self.self_attention = SelfAttention(2*hidden_dim, dropout_attention)
        self.self_attention_sentence = SelfAttention(2*hidden_dim, dropout_attention)
        self.embbedding = BiLSTMEncoder(embed_dim,hidden_dim,layers,dropout_lstm,dropout_input)
        self.doc_embbedding = BiLSTMEncoder(2*hidden_dim,hidden_dim,layers,dropout_lstm_hyper,dropout_input_hyper)
        self.metafor_classifier = Metaphor(dropout_FC, num_classes, hidden_dim)
        self.doc_classifier = Metaphor(dropout_FC, num_classes, hidden_dim)
        if torch.cuda.is_available():
            self.embbedding.to(device=torch.device('cuda'))
            self.metafor_classifier.to(device=torch.device('cuda'))
    def forward(self, inputs, lengths, doc_lengths):
        start = time.time()
        squezeed = torch.cat((inputs), 0)
        squezeed_lengths = torch.FloatTensor([val for sublist in lengths for val in sublist])
        if torch.cuda.is_available():
            squezeed = squezeed.to(device=torch.device('cuda'))
            squezeed_lengths = squezeed_lengths.to(device=torch.device('cuda'))
        predicted = self.embbedding(squezeed, squezeed_lengths)
        end = time.time()
        print(end - start, ' First layer')
        #normalized_output = self.metafor_classifier(out_embedding)
        #averaged_docs = torch.div((predicted.sum(dim=1)), squezeed_lengths.view(-1,1), out=None)
        averaged_docs, attention, weighted = self.self_attention_sentence(predicted, squezeed_lengths.int())
        predicted_docs = torch.split(averaged_docs, split_size_or_sections=list(doc_lengths))
        predicted_docs = pad_sequence(predicted_docs, batch_first=True, padding_value=0)
        end = time.time()
        print(end - start, ' Average sentences and pad doc')
        out_embedding = self.doc_embbedding.forward(predicted_docs, doc_lengths)
        end = time.time()
        print(end - start, ' Second Layer')
        prediction, attention, weighted = self.self_attention(out_embedding, doc_lengths)
        end = time.time()
        print(end - start, ' Attention Layer')
        class_prediction = self.doc_classifier(prediction)
        end = time.time()
        print(end - start, ' Last Layer')
        return class_prediction 

class HyperModel(nn.Module):
    
    def __init__(self, hidden_dim, layers, dropout_FC, dropout_lstm_hyper,dropout_input_hyper,dropout_attention,num_classes):
       
        super(HyperModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dropout_FC = dropout_FC
        self.self_attention = SelfAttention(2*hidden_dim, dropout_attention)
        self.self_attention_sentence = SelfAttention(2*hidden_dim, dropout_attention)
        self.doc_embbedding = BiLSTMEncoder(2*hidden_dim,hidden_dim,layers,dropout_lstm_hyper,dropout_input_hyper)
        self.doc_classifier = Metaphor(dropout_FC, num_classes, hidden_dim)
        self.new_embeds = SelfAttention_metaphors(2, 0)

    def forward(self, meta_pred, predicted, squezeed_lengths = torch.LongTensor(1).to(device=torch.device('cuda')), doc_lengths = torch.LongTensor(1).to(device=torch.device('cuda'))):
        
        start = time.time()
        end = time.time()
        print(end - start, ' First layer')

        averaged_docs, attention, weighted = self.new_embeds(predicted, meta_pred, squezeed_lengths.int())    
        averaged_docs, attention, weighted = self.self_attention_sentence(weighted, squezeed_lengths.int())
        predicted_docs = torch.split(averaged_docs, split_size_or_sections=list(doc_lengths))
        predicted_docs = pad_sequence(predicted_docs, batch_first=True, padding_value=0)
        end = time.time()
        print(end - start, ' Average sentences and pad doc')
        out_embedding = self.doc_embbedding.forward(predicted_docs, doc_lengths)
        end = time.time()
        print(end - start, ' Second Layer')
        prediction, attention, weighted = self.self_attention(out_embedding, doc_lengths)
        end = time.time()
        print(end - start, ' Attention Layer')
        class_prediction = self.doc_classifier(prediction)
        end = time.time()
        print(end - start, ' Last Layer')
        return class_prediction 

class HyperSoftModel(nn.Module):
    
    def __init__(self, hidden_dim, layers, dropout_FC, dropout_lstm_hyper,dropout_input_hyper,dropout_attention,num_classes):
       
        super(HyperSoftModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dropout_FC = dropout_FC
        self.self_attention = SelfAttention(2*hidden_dim, dropout_attention)
        self.self_attention_sentence = SelfAttention(hidden_dim, dropout_attention)
        self.doc_embbedding = BiLSTMEncoder(hidden_dim,hidden_dim,layers,dropout_lstm_hyper,dropout_input_hyper)
        self.doc_classifier = Metaphor(dropout_FC, num_classes, hidden_dim)
    
    def forward(self, predicted, squezeed_lengths, doc_lengths):
        averaged_docs, attention, weighted = self.self_attention_sentence(predicted, squezeed_lengths.int())
        predicted_docs = torch.split(averaged_docs, split_size_or_sections=list(doc_lengths))
        predicted_docs = pad_sequence(predicted_docs, batch_first=True, padding_value=0)
        out_embedding = self.doc_embbedding.forward(predicted_docs, doc_lengths)
        prediction, attention, weighted = self.self_attention(out_embedding, doc_lengths)
        class_prediction = self.doc_classifier(prediction)
        return class_prediction

# Self-attention layer from https://gist.github.com/cbaziotis/94e53bdd6e4852756e0395560ff38aa4
class SelfAttention(nn.Module):
    def __init__(self, attention_size,
                 batch_first=True,
                 layers=1,
                 dropout=.0,
                 non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        modules = []
        for i in range(layers - 1):
            modules.append(nn.Linear(attention_size, attention_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))

        # last attention layer must output 1
        modules.append(nn.Linear(attention_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))

        self.attention = nn.Sequential(*modules)

        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def get_mask(attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = Variable(torch.ones(attentions.size())).detach()
        mask = mask.to(device)

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, inputs, lengths):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.attention(inputs).squeeze()
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        # construct a mask, based on sentence lengths
        mask = self.get_mask(scores, lengths)

        # apply the mask - zero out masked timesteps
        masked_scores = scores * mask

        # re-normalize the masked scores
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        representations = weighted.sum(1).squeeze()

        return representations, scores, weighted

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

class SelfAttention_metaphors(nn.Module):
    def __init__(self, attention_size,
                 batch_first=True,
                 layers=1,
                 dropout=.0,
                 non_linearity="tanh"):
        super(SelfAttention_metaphors, self).__init__()

        self.batch_first = batch_first

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        modules = []
        for i in range(layers - 1):
            modules.append(nn.Linear(attention_size, attention_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))

        # last attention layer must output 1
        modules.append(nn.Linear(attention_size, 1))   #2 dimensions per word to 1 
        modules.append(activation)
        modules.append(nn.Dropout(dropout))

        self.attention = nn.Sequential(*modules)

        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def get_mask(attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = Variable(torch.ones(attentions.size())).detach()
        mask = mask.to(device)

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, inputs, metaphor , lengths):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.attention(metaphor).squeeze()            
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        # construct a mask, based on sentence lengths
        mask = self.get_mask(scores, lengths)

        # apply the mask - zero out masked timesteps
        masked_scores = scores * mask

        # re-normalize the masked scores
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        representations = weighted.sum(1).squeeze()

        return representations, scores, weighted


class multitask_soft_model(nn.Module):
  
  def __init__(self, encoder_param, hyper_param, meta_param):
    
    super(multitask_soft_model, self).__init__()

    self.embedding = BiLSTM_SOFT_Encoder(embed_dim = encoder_param['embed_dim'],
                                    hidden_dim = encoder_param['hidden_dim'],
                                    layers = encoder_param['layers'],
                                    dropout_lstm = encoder_param['dropout_lstm'],
                                    dropout_input = encoder_param['dropout_input'])
    
    self.embedding.to(device = 'cuda')
    self.metaphor_model = MetaphorModel(hidden_dim = meta_param['hidden_dim'], 
                                    dropout_FC = meta_param['dropout_FC'],#0.1,
                                    num_classes = 2)
    
    self.hyper_model = HyperSoftModel(hidden_dim = hyper_param['hidden_dim'],
                                  layers = hyper_param['layers'],
                      dropout_FC=hyper_param['dropout_FC'],
                      dropout_lstm_hyper = hyper_param['dropout_lstm_hyper'],
                      dropout_input_hyper = hyper_param['dropout_lstm_hyper'],
                      dropout_attention = hyper_param['dropout_lstm_hyper'],
                      num_classes = 2)

    
  def forward(self, input_data, length_data = torch.LongTensor(1).cuda, length_doc = torch.LongTensor(1), is_doc = True):

    if torch.cuda.is_available():
        input_data = input_data.to(device=torch.device('cuda'))
        length_data = length_data.to(device=torch.device('cuda'))


    out_embedding_meta, out_embedding_hyper = self.embedding(input_data, length_data,is_doc)
    if torch.cuda.is_available():
        out_embedding_meta = out_embedding_meta.to(device = torch.device('cuda'))
    meta_pred = self.metaphor_model(out_embedding_meta, length_data)
    if is_doc:
        if torch.cuda.is_available():
            length_doc = length_doc.to(device=torch.device('cuda'))
        hyp_pred = self.hyper_model(out_embedding_hyper, length_data, length_doc)
    else:
        hyp_pred = None
    
    return meta_pred, hyp_pred

class multitask_model(nn.Module):
  def __init__(self, encoder_param, hyper_param, meta_param):
    super(multitask_model, self).__init__()
    self.embedding = BiLSTMEncoder(embed_dim = encoder_param['embed_dim'],
                                    hidden_dim = encoder_param['hidden_dim'],
                                    layers = encoder_param['layers'],
                                    dropout_lstm = encoder_param['dropout_lstm'],
                                    dropout_input = encoder_param['dropout_input'])
    self.metaphor_model = MetaphorModel(hidden_dim = meta_param['hidden_dim'], 
                                    dropout_FC = meta_param['dropout_FC'],#0.1,
                                    num_classes = 2)
    self.hyper_model = HyperModel(hidden_dim = hyper_param['hidden_dim'],
                                  layers = hyper_param['layers'],
                      dropout_FC=hyper_param['dropout_FC'],
                      dropout_lstm_hyper = hyper_param['dropout_lstm_hyper'],
                      dropout_input_hyper = hyper_param['dropout_lstm_hyper'],
                      dropout_attention = hyper_param['dropout_lstm_hyper'],
                      num_classes = 2)
  def forward(self, input_data, length_data = torch.LongTensor(1), length_doc = torch.LongTensor(1), is_doc = True):

    if torch.cuda.is_available():
        input_data = input_data.to(device=torch.device('cuda'))
        length_data = length_data.to(device=torch.device('cuda'))

    out_embedding = self.embedding(input_data, length_data)
    if torch.cuda.is_available():
        out_embedding = out_embedding.to(device = torch.device('cuda'))

    meta_pred = self.metaphor_model(out_embedding, length_data)
    if is_doc:
        if torch.cuda.is_available():
            length_doc = length_doc.to(device=torch.device('cuda'))
        hyp_pred = self.hyper_model(meta_pred, out_embedding, length_data, length_doc)
    else:
        hyp_pred = None
    
    return meta_pred, hyp_pred