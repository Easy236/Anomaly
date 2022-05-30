import torch


def create_dataset(sequences):
    dataset = [torch.tensor(s).unsqueeze(1) for s in sequences]
    n_seq, seq_len, n_features = torch.stack(dataset).shape
    return dataset, seq_len, n_features
