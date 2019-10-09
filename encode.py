import torch.nn as nn
from word_sequence import num_sequence
import config


class NumEncoder(nn.Module):
    def __init__(self):
        super(NumEncoder,self).__init__()
        self.vocab_size = len(num_sequence)
        self.dropout = config.dropout
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=config.embedding_dim, padding_idx=num_sequence.PAD)
        self.gru = nn.GRU(input_size=config.embedding_dim,
                          hidden_size=config.hidden_size,
                          num_layers=1,
                          batch_first=True)


    def forward(self, input, input_length):
        embeded = self.embedding(input)

        embeded = nn.utils.rnn.pack_padded_sequence(embeded, lengths=input_length, batch_first=True)

        out, hidden = self.gru(embeded)

        out, outputs_length = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, padding_value=num_sequence.PAD)

        return out, hidden