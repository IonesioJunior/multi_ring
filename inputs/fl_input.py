import json

import torch
import torch.nn as nn


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


# Define colors
RED = "\033[0;31m"
GREEN = "\033[0;32m"
BLUE = "\033[0;34m"
NC = "\033[0m"  # No Color


def get_inputs():
    # print(f"{BLUE}Please, fill the fields in order to setup your peer for the ring app:{NC}\n")
    is_lead = input(f"{BLUE}Will you be the leading this ring round (y/N)?{NC}")

    members = []
    if is_lead.lower() == "y":
        leader = input(f"{BLUE}Add your email as a ring member: {NC}")
        members.append(leader)
        while True:
            member = input(
                f"{BLUE}Add new a ring member (Leave empty to stop adding peers): {NC}"
            )
            if not member:
                break
            members.append(member)

        members.append(leader)

        lr = float(input(f"{BLUE}Set the learning rate: {NC}"))

        iterations = int(input(f"{BLUE}Set the number of iterations: {NC}"))

        model = SimpleNN()
        serialized_model = torch.save(model.state_dict(), "mnist_model.pth")


        with open("data.json", "w") as file:
            print("Writing the data.json file ...")
            file.write(
                json.dumps(
                    {
                        "ring": members,
                        "data": 0,
                        "current_index": 0,
                        "iterations": iterations,
                        "learning_rate": lr,
                        "model": "mnist_model.pth",
                    }
                )
            )

    secret_number = input(
        f"{BLUE}ADD the path were your MNIST dataset subset lives: {NC}"
    )

    # Write the secret number to a file
    with open("secret.txt", "w") as file:
        file.write(secret_number)

    # Confirm the action
    print(f"\n\n{GREEN}Everything has been set! Have good fun!{NC}\n\n")
