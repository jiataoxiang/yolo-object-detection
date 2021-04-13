import torch


if __name__ == "__main__":
    predictions = torch.tensor([[1, 2], [3, 4]])

    print(predictions[..., 1:2])
    print(predictions[..., 1].unsqueeze(-1))