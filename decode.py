import torch
import torch.nn as nn
import config
import random
import torch.nn.functional as F
from word_sequence import num_sequence


class NumDecoder(nn.Module):
    def __init__(self):
        super(NumDecoder,self).__init__()
        self.max_seq_len = config.max_len
        self.vocab_size = len(num_sequence)
        self.embedding_dim = config.embedding_dim
        self.dropout = config.dropout

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,embedding_dim=self.embedding_dim, padding_idx=num_sequence.PAD)
        self.gru = nn.GRU(input_size=self.embedding_dim,
                          hidden_size=config.hidden_size,
                          num_layers=1,
                          batch_first=True,
                          dropout=self.dropout)
        self.log_softmax = nn.LogSoftmax()
        self.fc = nn.Linear(config.hidden_size, self.vocab_size)

    def forward(self, encoder_hidden,  target, target_length):

        decoder_input = torch.LongTensor([[num_sequence.SOS]]*config.batch_size)

        decoder_outputs = torch.zeros(config.batch_size, config.max_len, self.vocab_size)

        decoder_hidden = encoder_hidden

        for t in range(config.max_len):
            decoder_output_t , decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[:,t,:] = decoder_output_t

            use_teacher_forcing = random.random() > 0.5
            if use_teacher_forcing:
                decoder_input = target[:,t].unsqueeze(1)
            else:
                value, index = torch.topk(decoder_output_t, 1)
                decoder_input = index

        return decoder_outputs, decoder_hidden


    def forward_step(self, decoder_input, decoder_hidden):

        embeded = self.embedding(decoder_input)

        out, decoder_hidden = self.gru(embeded, decoder_hidden)

        out = out.squeeze(0)

        out = F.log_softmax(self.fc(out), dim=-1)
        out = out.squeeze(1)

        return out, decoder_hidden

    def evaluation(self, encoder_hidden):  # [1, 20, 14]
        batch_size = encoder_hidden.size(1)  # 评估的时候和训练的batch_size不同，不适用config的配置

        decoder_input = torch.LongTensor([[num_sequence.SOS] * batch_size])
        decoder_outputs = torch.zeros(batch_size, config.max_len, self.vocab_size)  # [batch_size，seq_len,vocab_size]
        decoder_hidden = encoder_hidden

        # 评估，不再使用teacher forcing，完全使用预测值作为下一次的输入
        for t in range(config.max_len):
            decoder_output_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[:, t, :] = decoder_output_t
            value, index = torch.topk(decoder_output_t, 1)  # index [20,1]
            decoder_input = index.transpose(0, 1)

        # 获取输出的id
        decoder_indices = []
        for i in range(decoder_outputs.size(1)):
            value, indices = torch.topk(decoder_outputs[:, i, :], 1)
            decoder_indices.append(int(indices[0][0].data))
        return decoder_indices