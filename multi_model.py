class MetaphorModel(nn.Module):
    
    def __init__(self, hidden_dim, dropout_FC, num_classes):
        
        super(MetaphorModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout_FC = dropout_FC
        self.self_attention = SelfAttention(2*hidden_dim)
        
#         self.embbedding = BiLSTMEncoder(embed_dim,hidden_dim,layers,dropout_lstm,dropout_input)
        self.metafor_classifier = Metaphor(dropout_FC, num_classes, hidden_dim)
#         if torch.cuda.is_available():
#             self.embbedding.to(device=torch.device('cuda'))
#             self.embbedding.to(device=torch.device('cuda'))
          
    def forward(self, out_embedding, inputs, lengths):

#         out_embedding = self.embbedding.forward(inputs, lengths)
        
#         out_attention, attention, weighted = self.self_attention(out_embedding, lengths)
        normalized_output = self.metafor_classifier(out_embedding)

        return normalized_output
        
class HyperModel(nn.Module):
    
    def __init__(self, hidden_dim, layers, dropout_FC, dropout_lstm_hyper,dropout_input_hyper,dropout_attention,num_classes):
       
        super(HyperModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dropout_FC = dropout_FC
        self.self_attention = SelfAttention(2*hidden_dim, dropout_attention)
        self.self_attention_sentence = SelfAttention(2*hidden_dim, dropout_attention)
#         self.embbedding = BiLSTMEncoder(embed_dim,hidden_dim,layers,dropout_lstm,dropout_input)
        self.doc_embbedding = BiLSTMEncoder(2*hidden_dim,hidden_dim,layers,dropout_lstm_hyper,dropout_input_hyper)
        self.metafor_classifier = Metaphor(dropout_FC, num_classes, hidden_dim)
        self.doc_classifier = Metaphor(dropout_FC, num_classes, hidden_dim)
        if torch.cuda.is_available():
#             self.embbedding.to(device=torch.device('cuda'))
            self.metafor_classifier.to(device=torch.device('cuda'))
    
    def forward(self, predicted, squezeed_lengths, inputs, lengths, doc_lengths):
        
        start = time.time()
#         squezeed = torch.cat((inputs), 0)
#         squezeed_lengths = torch.FloatTensor([val for sublist in lengths for val in sublist])
#         if torch.cuda.is_available():
#             squezeed = squezeed.to(device=torch.device('cuda'))
#             squezeed_lengths = squezeed_lengths.to(device=torch.device('cuda'))
#         predicted = self.embbedding(squezeed, squezeed_lengths)
        end = time.time()
        print(end - start, ' First layer')

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
class multitask_model(nn.Module):
  
  def __init__(self, encoder_param, hyper_param, meta_param):
    
    super(multitask_model, self).__init__()
    
    self.embedding = BiLSTMEncoder(embed_dim = encoder_param['embed_dim'],
                                    hidden_dim = encoder_param['hidden_dim'],
                                    layers = encoder_param['layers'],
                                    dropout_lstm = encoder_param['dropout_lstm'],
                                    dropout_input = encoder_param['dropout_input'])
    
    self.embedding.to(device = 'cuda')
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

    
  def forward(self, is_doc, input_data, length_data, length_doc):
    
    if is_doc:
        squezeed = torch.cat((input_data), 0)
        squezeed_lengths = torch.FloatTensor([val for sublist in length_data for val in sublist])
    
        if torch.cuda.is_available():
            squezeed = squezeed.to(device=torch.device('cuda'))
            squezeed_lengths = squezeed_lengths.to(device=torch.device('cuda'))

        out_embedding = self.embedding(squezeed, squezeed_lengths)
        if torch.cuda.is_available():
            out_embedding = out_embedding.to(device = torch.device('cuda'))
    else:
        if torch.cuda.is_available():
            input_data = input_data.to(device=torch.device('cuda'))
            length_data = length_data.to(device=torch.device('cuda'))
        out_embedding = self.embedding(input_data, length_data)
        if torch.cuda.is_available():
            out_embedding = out_embedding.to(device = torch.device('cuda'))
        hyp_pred = None

    meta_pred = self.metaphor_model(out_embedding, input_data, length_data)
    
    if is_doc:
        if torch.cuda.is_available():
            length_doc = length_doc.to(device=torch.device('cuda'))
        hyp_pred = self.hyper_model(out_embedding, squezeed_lengths, input_data, length_data, length_doc)
    
    return meta_pred, hyp_pred
