import torch
import config
from torch import optim
import torch.nn as nn
from encode import NumEncoder
from decode import NumDecoder
from num_seq2seq import Seq2Seq
from dataset import data_loader as train_dataloader
from word_sequence import num_sequence
import numpy as np
import random

encoder = NumEncoder()
decoder = NumDecoder()
model = Seq2Seq(encoder, decoder)
model.load_state_dict(torch.load("model/seq2seq_model.pkl"))


def evalaute():
    for i in range(10):
        test_words = random.randint(1, 100000000)
        test_word_len = [len(str(test_words))]
        _test_words = torch.LongTensor([num_sequence.transform(test_words)])

        decoded_incdices = model.evaluation(_test_words, test_word_len)
        # 转化为字符串
        result = num_sequence.inverse_transform(decoded_incdices)
        print(test_words, ">>>>>", "".join(result), str(test_words) + "0" == "".join(result))


if __name__ == '__main__':
    evalaute()