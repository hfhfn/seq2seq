class NumSequence:
    UNK_TAG = 'UNK' # 未知词
    PAD_TAG = 'PAD' # 填充词
    EOS_TAG = 'EOS' # 句子的开始
    SOS_TAG = 'SOS' # 句子的结束

    UNK = 0
    PAD = 1
    EOS = 2
    SOS = 3

    def __init__(self):
        self.dict = {
            self.UNK_TAG : self.UNK,
            self.PAD_TAG : self.PAD,
            self.EOS_TAG : self.EOS,
            self.SOS_TAG : self.SOS,
        }
        for i in range(10):
            self.dict[str(i)] = len(self.dict)
            self.index2word = dict(zip(self.dict.values(), self.dict.keys()))

    def __len__(self):
        return len(self.dict)

    def transform(self, sequence, max_len=None, add_eos=False):
        sequence_list = list(str(sequence))
        seq_len = len(sequence_list)+1 if add_eos else len(sequence_list)

        if add_eos and max_len is not None:
            assert max_len >= seq_len, 'max_len 需要大于seq+eos的长度'
        _sequence_index = [self.dict.get(i, self.UNK) for i in sequence_list]
        if add_eos:
            _sequence_index += [self.EOS]
        if max_len is not None:
            sequence_index = [self.PAD] * max_len
            sequence_index[:seq_len] = _sequence_index
            return sequence_index
        else:
            return _sequence_index

    def inverse_transform(self, sequence_index):
        result = []
        for i in sequence_index:
            if i == self.EOS:
                break
            result.append(self.index2word.get(int(i), self.UNK_TAG))
        return result

num_sequence = NumSequence()

if __name__ == '__main__':
    num_sequence = NumSequence()
    print(num_sequence.dict)
    print(num_sequence.index2word)
    # to_index = ''.join(num_sequence.transform('1231230', add_eos=True))
    to_index = num_sequence.transform('1231230', add_eos=True)
    print(to_index)
    print(''.join(num_sequence.inverse_transform(to_index)))
