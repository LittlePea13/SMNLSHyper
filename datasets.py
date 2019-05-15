import torch
import numpy as np
import mmap
from torch.utils.data import Dataset
import torch.nn as nn
from torch.autograd import Variable
import random

class SentenceDataset(Dataset):
    def __init__(self, embedded_text, labels, max_sequence_length=100):
        """
        :param embedded_text:
        :param labels: a list of ints
        :param max_sequence_length: an int
        """
        print('Number of sentences', len(embedded_text))
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
        number_words = sum(len(element) for element in labels)
        print('number of words', number_words)

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
        #if len(embedded_docs.shape) == 1:
        #    embedded_docs = embedded_docs.reshape((-1,1))
        max_length_sen = [max([len(sentence) for sentence in element]) for element in embedded_docs]
        indexes = np.argsort(max_length_sen)
        self.embedded_docs = embedded_docs[indexes]
        self.doc_lens = [len(element)*max([len(sentence) for sentence in element]) for element in embedded_docs[indexes]]
        # A list of ints, where each int is a label of the sentence at the corresponding index.
        self.labels = labels[indexes]
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
        # Truncate the sequence if necessary fix this, now is doing documents
        example_text = [text[:self.max_sequence_length] for text in example_text]
        example_length = len(example_text)

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
        max_length = max([element[1] for element in batch])
        #print(max(batch, key=lambda example: example[1]))
        #print(max(batch, key=lambda example: example[2]))
        #print(max_length)
        #print(max(batch, key=lambda example: example[3]))
        max_length_sen = max([max(element[3]) for element in batch])

        # Iterate over each example in the batch
        for text, length, label, sen_len in batch:
            doc_sentences = []
            # Unpack the example (returned from __getitem__)
            for sentence in text:
                amount_to_pad = max_length_sen - len(sentence)
                sentence = torch.Tensor(sentence)
                pad_tensor = torch.zeros(amount_to_pad, sentence.shape[1])
                padded_sentence_text = torch.cat((sentence, pad_tensor), dim=0)
                doc_sentences.append(padded_sentence_text)
            '''amount_to_pad = max_length - length
            for i in range(amount_to_pad):
                pad_tensor = torch.zeros(max_length_sen, sentence.shape[1])
                doc_sentences.append(pad_tensor)
            sentences_len = torch.Tensor(sen_len)
            padded_senlen = torch.cat((sentences_len, torch.zeros(amount_to_pad)), dim=0)'''
            batch_padded_example_text.append(torch.stack(doc_sentences))
            # Add the padded example to our batch
            batch_lengths.append(length)
            batch_labels.append(label)
            batch_sen_lengths.append(torch.FloatTensor(sen_len))
        # Stack the list of LongTensors into a single LongTensor
        return (torch.cat((batch_padded_example_text), 0),
                torch.LongTensor(batch_lengths),
                torch.LongTensor(batch_labels),
                torch.cat(batch_sen_lengths, 0))

class Sampler(object):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class AdaptSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, lengths, batch_size,max_size):

        self.lengths = lengths
        self.batch_size = batch_size
        self.max_size = max_size

    def __iter__(self):
        batch = []
        for idx in range(len(self.lengths)):
            batch.append(idx)
            if sum(self.lengths[batch[0]:idx+1]) > self.max_size or len(batch) == self.batch_size:
                last = batch.pop()
                yield random.sample(batch, len(batch))
                batch = [last]
        if len(batch) > 0:
            yield batch

    def __len__(self):
        batch = []
        len_ = 0
        for idx in range(len(self.lengths)):
            batch.append(idx)
            if sum(self.lengths[batch[0]:idx+1]) > self.max_size or len(batch) == self.batch_size:
                last = batch.pop()
                len_ += 1
                batch = [last]
        if len(batch) > 0:
            len_ += 1
        return len_