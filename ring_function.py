from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from types import SimpleNamespace
from syftbox.lib import Client, SyftPermission
from torch.utils.data import DataLoader, TensorDataset
import shutil
import os

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def ring_function(ring_data: SimpleNamespace, secret_path: Path):
    client = Client.load()
    mnist_path = ""
    with open(secret_path, 'r') as secret_file:
        mnist_path = secret_file.read().strip()


    if ring_data.current_index >= len(ring_data.ring) - 1:
        done_pipeline_path: Path = (
            Path(client.datasite_path) / "app_pipelines" / "ring" / "done"
        )
        shutil.move(ring_data.model, str(done_pipeline_path))
        return 0

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the saved MNIST subset
    images, labels = torch.load(mnist_path)

    # Create a TensorDataset
    dataset = TensorDataset(images, labels)

    # Create a DataLoader for the dataset
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # train_dataset = datasets.MNIST(
    #     mnist_path, train=True, download=True, transform=transform
    # )
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = SimpleNN()  # Initialize model

    # Load serialized model if present
    if hasattr(ring_data, "model"):
        state_dict = torch.load(ring_data.model)
        model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=float(ring_data.learning_rate))

    print("\n\n Training...\n\n ")
    # Training loop
    for epoch in range(int(ring_data.iterations)):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    print("\n\n Done...\n\n ")

    next_index = ring_data.current_index + 1
    next_person = ring_data.ring[next_index]

    destination_datasite_path = Path(client.sync_folder) / next_person
    new_model_path = (
        destination_datasite_path
        / "app_pipelines"
        / "ring"
        / "running"
        / ring_data.model
    )

    print(f"\n\n Saving it in {str(new_model_path)}\n\n")
    # Serialize the model
    os.makedirs(os.path.dirname(str(new_model_path)), exist_ok=True)
    torch.save(model.state_dict(), str(new_model_path))
    return 0
