import os

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


# Define the Model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Define the Training Function
def train(rank, world_size):
    # Initialize the process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )

    # Prepare dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    # Model and optimizer
    model = SimpleModel().to(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # Initialize Accelerator
    accelerator = Accelerator()
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Training loop
    for epoch in range(1, 5):
        model.train()
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            accelerator.backward(loss)
            optimizer.step()
            if batch_idx % 10 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}"
                )

    torch.distributed.destroy_process_group()


# Main Function to Launch Processes
def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
