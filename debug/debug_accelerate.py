import hashlib
import io
import logging
import os

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from accelerate import Accelerator
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, handlers=[])


def worker(rank, world_size, model_serialized, from_optimizer, train_dataset):
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
    # logging.info(f"accelerator.state: {accelerator.state}")
    logging.info(f"accelerator.device: {accelerator.device}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    loss_fn = torch.nn.NLLLoss()

    logging.info(
        f"{rank}: model_serialized sha: {hashlib.sha256(model_serialized.get_obj()).hexdigest()}"
    )
    model = torch.load(io.BytesIO(model_serialized.get_obj()))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # optimizer.load_state_dict(from_optimizer.state_dict())
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    for epoch in range(5):
        logging.info(f"{rank}: Epoch {epoch}, train_loader length: {len(train_loader)}")
        model.train()
        losses = []
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            losses.append(loss.item())
            accelerator.backward(loss)
            optimizer.step()
        logging.info(f"{rank}: Epoch {epoch}, loss: {sum(losses)/len(losses)}")

    if accelerator.is_main_process:
        logging.info(f"{rank}: Saving model...")
        out_model = accelerator.unwrap_model(model).to("cpu")
        model_buffer = io.BytesIO()
        torch.save(out_model, model_buffer)
        model_serialized.get_obj()[:] = model_buffer.getvalue()
        logging.info(
            f"{rank}: model_serialized sha: {hashlib.sha256(model_serialized.get_obj()).hexdigest()}"
        )

    torch.distributed.destroy_process_group()


def evaluate(model, eval_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in eval_loader:
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    logging.info(f"Accuracy: {accuracy}%")


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1)
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        # Output layer
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    logging.info(f"train_dataset length: {len(train_dataset)}")

    eval_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

    model = SimpleCNN()

    evaluate(model, eval_loader)

    model_buffer = io.BytesIO()
    torch.save(model, model_buffer)
    model_serialized = mp.Array("b", model_buffer.getvalue())
    logging.info(
        f"model_serialized sha: {hashlib.sha256(model_serialized.get_obj()).hexdigest()}"
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    world_size = torch.cuda.device_count()
    logging.info(f"Spawning {world_size} processes...")
    mp.spawn(
        worker,
        args=(world_size, model_serialized, optimizer, train_dataset),
        nprocs=world_size,
        join=True,
    )

    logging.info(
        f"model_serialized sha: {hashlib.sha256(model_serialized.get_obj()).hexdigest()}"
    )
    out_model = torch.load(io.BytesIO(model_serialized.get_obj()))
    logging.info(f"out_model device: {out_model.device}")
    evaluate(out_model, eval_loader)
