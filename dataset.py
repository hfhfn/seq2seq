from torch.utils.data import Dataset, DataLoader
import numpy as np
from word_sequence import num_sequence
import torch
import config

class RandomDataset(Dataset):
    def __init__(self):
        super(RandomDataset, self).__init__()
        self.total_data_size = 500000
        np.random.seed(10)
        self.total_data = np.random.randint(1, 100000000, size=[self.total_data_size])

    def __getitem__(self, idx):
        input = str(self.total_data[idx])
        return input, input+'0', len(input), len(input)+1

    def __len__(self):
        return self.total_data_size


def collate_fn(batch):
    batch = sorted(batch, key=lambda x:x[3], reverse=True)
    input, target, input_length, target_length = zip(*batch)

    input = torch.LongTensor([num_sequence.transform(i, max_len=config.max_len) for i in input])
    target = torch.LongTensor([num_sequence.transform(i, max_len=config.max_len, add_eos=True) for i in target])
    input_length = torch.LongTensor(input_length)
    target_length = torch.LongTensor(target_length)

    return input, target, input_length, target_length


data_loader = DataLoader(dataset=RandomDataset(), batch_size=config.batch_size, collate_fn=collate_fn, drop_last=True)
