import torch
import numpy as np
import mmap
from torch.utils.data import Dataset
import torch.nn as nn
from torch.autograd import Variable

class SentenceDataset(Dataset):
    def __init__(self, embedded_text, labels, max_sequence_length=100):
        """
        :param embedded_text:
        :param labels: a list of ints
        :param max_sequence_length: an int
        """
        if len(embedded_text) != len(labels):
            raise ValueError("Differing number of sentences and labels!")
        # A list of numpy arrays, where each inner numpy arrays is sequence_length * embed_dim
        # embedding for each word is : elmo
        self.embedded_text = embedded_text
        # A list of ints, where each int is a label of the sentence at the corresponding index.
        self.labels = labels
        # Truncate examples that are longer than max_sequence_length.
        # Long sequences are expensive and might blow up GPU memory usage.
        self.max_sequence_length = max_sequence_length


    def __getitem__(self, idx):
        """
        Return the Dataset example at index `idx`.
        Returns
        -------
        example_text: numpy array
        length: int
            The length of the (possibly truncated) example_text.
        example_label: int 0 or 1
            The label of the example.
        """
        example_text = self.embedded_text[idx]
        example_label = self.labels[idx]
        # Truncate the sequence if necessary
        example_text = example_text[:self.max_sequence_length]
        example_length = example_text.shape[0]

        return example_text, example_length, example_label

    def __len__(self):
        """
        Return the number of examples in the Dataset.
        """
        return len(self.labels)

    @staticmethod
    def collate_fn(batch):
        """
        Given a list of examples (each from __getitem__),
        combine them to form a single batch by padding.
        Returns:
        -------
        batch_padded_example_text: LongTensor
          LongTensor of shape (batch_size, longest_sequence_length) with the
          padded text for each example in the batch.
        length: LongTensor
          LongTensor of shape (batch_size,) with the unpadded length of the example.
        example_label: LongTensor
          LongTensor of shape (batch_size,) with the label of the example.
        """
        batch_padded_example_text = []
        batch_lengths = []
        batch_labels = []

        # Get the length of the longest sequence in the batch
        max_length = max(batch, key=lambda example: example[1])[1]

        # Iterate over each example in the batch
        for text, length, label in batch:
            # Unpack the example (returned from __getitem__)

            # Amount to pad is length of longest example - length of this example.
            amount_to_pad = max_length - length
            # Tensor of shape (amount_to_pad,), converted to LongTensor
            pad_tensor = torch.zeros(amount_to_pad, text.shape[1])

            # Append the pad_tensor to the example_text tensor.
            # Shape of padded_example_text: (padded_length, embeding_dim)
            # top part is the original text numpy,
            # and the bottom part is the 0 padded tensors

            # text from the batch is a np array, but cat requires the argument to be the same type
            # turn the text into a torch.FloatTenser, which is the same type as pad_tensor
            text = torch.Tensor(text)
            padded_example_text = torch.cat((text, pad_tensor), dim=0)
            padded_example_label = np.append(label,np.zeros(amount_to_pad))

            # Add the padded example to our batch
            batch_padded_example_text.append(padded_example_text)
            batch_lengths.append(length)
            batch_labels.append(padded_example_label)

        tensor_text = torch.stack(batch_padded_example_text)
        tensor_labels = torch.LongTensor(batch_labels)
        tensor_lengths = torch.LongTensor(batch_lengths)
        # Stack the list of LongTensors into a single LongTensor
        return (tensor_text,
                tensor_lengths,
                tensor_labels)

class DocumentDataset(Dataset):
    def __init__(self, embedded_docs, labels, max_sequence_length=100):
        """
        :param embedded_text:
        :param labels: a list of ints
        :param max_sequence_length: an int
        """
        if len(embedded_docs) != len(labels):
            raise ValueError("Differing number of sentences and labels!")
        # A list of numpy arrays, where each inner numpy arrays is sequence_length * embed_dim
        # embedding for each word is : elmo
        if len(embedded_docs.shape) == 1:
            embedded_docs = embedded_docs.reshape((-1,1))
        self.embedded_docs = embedded_docs
        # A list of ints, where each int is a label of the sentence at the corresponding index.
        self.labels = labels
        # Truncate examples that are longer than max_sequence_length.
        # Long sequences are expensive and might blow up GPU memory usage.
        self.max_sequence_length = max_sequence_length


    def __getitem__(self, idx):
        """
        Return the Dataset example at index `idx`.
        Returns
        -------
        example_text: numpy array
        length: int
            The length of the (possibly truncated) example_text.
        example_label: int 0 or 1
            The label of the example.
        """
        example_text = self.embedded_docs[idx]
        sentences_lengths = [len(sentence) for sentence in example_text]
        example_label = self.labels[idx]
        # Truncate the sequence if necessary
        example_text = example_text[:self.max_sequence_length]
        example_length = example_text.shape[0]

        return example_text, example_length, example_label, sentences_lengths

    def __len__(self):
        """
        Return the number of examples in the Dataset.
        """
        return len(self.labels)

    @staticmethod
    def collate_fn(batch):
        """
        Given a list of examples (each from __getitem__),
        combine them to form a single batch by padding.
        Returns:
        -------
        batch_padded_example_text: LongTensor
          LongTensor of shape (batch_size, longest_sequence_length) with the
          padded text for each example in the batch.
        length: LongTensor
          LongTensor of shape (batch_size,) with the unpadded length of the example.
        example_label: LongTensor
          LongTensor of shape (batch_size,) with the label of the example.
        """
        batch_padded_example_text = []
        batch_lengths = []
        batch_labels = []
        batch_sen_lengths = []

        # Get the length of the longest sequence in the batch
        max_length = max(batch, key=lambda example: example[1])[1]
        max_length_sen = max(max(batch, key=lambda example: example[4])[1])

        # Iterate over each example in the batch
        for text, length, label, sen_len in batch:
            doc_sentences = []
            # Unpack the example (returned from __getitem__)
            for sentence in text:
                amount_to_pad = max_length_sen - len(sentence)
                pad_tensor = torch.zeros(amount_to_pad, sentence.shape[1])
                sentence = torch.Tensor(sentence)
                padded_sentence_text = torch.cat((sentence, pad_tensor), dim=0)
                doc_sentences.append(padded_sentence_text)
            amount_to_pad = max_length - length
            for i in range(amount_to_pad):
                pad_tensor = torch.zeros(max_length_sen, sentence.shape[1])
                doc_sentences.append(pad_tensor)
            batch_padded_example_text.append(torch.stack(doc_sentences))
            # Add the padded example to our batch
            batch_lengths.append(length)
            batch_labels.append(label)
            batch_sen_lengths.append(sen_len)

        # Stack the list of LongTensors into a single LongTensor
        return (torch.stack(batch_padded_example_text),
                torch.LongTensor(batch_lengths),
                torch.LongTensor(batch_labels),
                torch.LongTensor(batch_lengths_sent))