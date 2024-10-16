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
import shutil


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
        mnist_path = secret_file.read()

    print("My Mnist dataset path is: ", mnist_path)

    if ring_data.current_index >= len(ring_data.ring) - 1:
        done_pipeline_path: Path = (
            Path(client.datasite_path) / "app_pipelines" / "ring" / "done"
        )
        shutil.move(ring_data.model, str(done_pipeline_path))
        return 0

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    print("Loading mnist dataset file")
    train_dataset = datasets.MNIST(
        mnist_path, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = SimpleNN()  # Initialize model

    print("Loading the model file ...")
    # Load serialized model if present
    if hasattr(ring_data, "model"):
        state_dict = torch.load(ring_data.model)
        model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=float(ring_data.learning_rate))

    print("\n\n Training...\n\n ")
    try:
        # Training loop
        for epoch in range(int(ring_data.iterations)):
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if epoch % 100 == 0:
                    print("\n\n Hello World \n\n")
    except Exception as e:
        print(f"Ops! Something went wrong {str(e)}")
        return 0
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
    torch.save(model.state_dict(), str(new_model_path))
    return 0
