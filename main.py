import ast
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasets import SentenceDataset,DocumentDataset
from Data.Metaphors.embeddings import extract_emb
import torch.utils.data as data_utils
from model import BiLSTMEncoder,MainModel
def load_elmo_dataset(path, max_len=200):
    '''
    load ELMo embedding from tsv file.
    :param path: tsv file path.
    :param to_pickle: Convert elmo embeddings to .npy file, avoid read and pad every time.
    :return: elmo embedding and its label.
    '''
    X = []
    label = []
    ids = []
    i = 0
    with open(path, 'rb') as inf:
        for line in inf:
            gzip_fields = line.decode('utf-8').split('\t')
            gzip_id = gzip_fields[0]
            gzip_label = gzip_fields[1]
            elmo_embd_str = gzip_fields[4].strip()
            elmo_embd_list = ast.literal_eval(elmo_embd_str)
            elmo_embd_array = np.array(elmo_embd_list)
            padded_seq = sequence.pad_sequences([elmo_embd_array], maxlen=max_len, dtype='float32')[0]
            X.append(padded_seq)
            label.append(gzip_label)
            ids.append(gzip_id)
            i += 1
            print(i)
    # transform label to variable
    Y = l_encoder.fit_transform(label)
    #tensor_data = torch.from_numpy(np.array(X))
    #tensor_target = torch.from_numpy(np.array(Y))
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    batch_size = 64
    hidden_dim = 512
    emb_file = 'Data/Metaphors/meta_embeds.npy'
    lab_file = 'Data/Metaphors/meta_labels.npy'
    data, labels = extract_emb(emb_file, lab_file)
    #data, labels = load_elmo_dataset(path,200)
    loader = SentenceDataset(data, labels, 200)
    loader_doc = DocumentDataset(data, labels, 200)
    loader_dataset = data_utils.DataLoader(loader, batch_size=batch_size, shuffle=True,
                                  collate_fn=SentenceDataset.collate_fn)
    model = MainModel(embed_dim=1024, hidden_dim = hidden_dim, layers = 1, dropout_lstm = 0, dropout_input=0, dropout_FC=0, num_classes = 2)
    for data, length, label in loader_dataset:
        prediction = model.forward(data, length)
        print(prediction.shape)
        break