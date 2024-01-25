import logging
import os

import torch
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
from accelerate import Accelerator
from torch.utils.data import DataLoader, DistributedSampler

logging.basicConfig(level=logging.INFO, handlers=[])


def worker(rank, world_size, model, optimizer, train_dataset):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["LOCAL_RANK"] = str(rank)

    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )

    # Check if the process group is initialized correctly
    if torch.distributed.is_initialized():
        logging.info(
            f"Process Group Initialized: Rank {torch.distributed.get_rank()}, World Size {torch.distributed.get_world_size()}, GPUs: {torch.cuda.device_count()}"
        )
    else:
        logging.info("Failed to initialize Process Group")

    accelerator = Accelerator()
    logging.info(f"accelerator.state: {accelerator.state}")
    logging.info(f"accelerator.device: {accelerator.device}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    for epoch in range(10):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data.view(data.size(0), -1))
            loss = loss_fn(output, target)
            accelerator.backward(loss)
            optimizer.step()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    model = torch.nn.Linear(784, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    world_size = torch.cuda.device_count()
    logging.info(f"Spawning {world_size} processes...")
    mp.spawn(
        worker,
        args=(world_size, model, optimizer, train_dataset),
        nprocs=world_size,
        join=True,
    )
