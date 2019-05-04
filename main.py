import ast
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasets import SentenceDataset

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
    tensor_data = torch.from_numpy(np.array(X))
    tensor_target = torch.from_numpy(np.array(Y))
    return tensor_data, tensor_target

if __name__ == "__main__":
    batch_size = 64
    data, labels = load_elmo_dataset(path,200)
    loader = TextDataset(data, labels, 200)
    loader_dataset = data_utils.DataLoader(loader, batch_size=batch_size, shuffle=True,
                                  collate_fn=TextDataset.collate_fn)