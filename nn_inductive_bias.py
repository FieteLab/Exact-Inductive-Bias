import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import compute_inductive_bias

# Set random seeds for reproducibility
torch.manual_seed(99)
torch.cuda.manual_seed(99)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define the model architecture
class Net(nn.Module):
    def __init__(self, width, depth, input_size, output_size):
        super(Net, self).__init__()

        # Define the layers
        self.input_size = input_size
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, width))
        for _ in range(depth):
            self.layers.append(nn.Linear(width, width))
        self.layers.append(nn.Linear(width, output_size))

    def forward(self, x):
        x = x.view(-1, self.input_size)
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x


# Function to train the model
def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0.0
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        target_one_hot = torch.zeros_like(output)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        loss = criterion(output, target_one_hot)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader.dataset)


# Function to evaluate the model
def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            target_one_hot = torch.zeros_like(output)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1)
            loss = criterion(output, target_one_hot)
            total_loss += loss.item()
            breakpoint()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return total_loss / len(test_loader.dataset), correct / len(test_loader.dataset)


def run(train_dataset, test_dataset, input_size, output_size, width, depth, epochs, batch_size, num_trials):
    test_losses = []

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Perform multiple trainings with different random initializations and training data orderings
    for trial in range(num_trials):
        print('Trial ', trial + 1)
        # Initialize the model
        model = Net(width, depth, input_size, output_size).to(device)

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        for epoch in range(epochs):
            train_loss = train(model, train_loader, criterion, optimizer)
            print(f"Epoch {epoch + 1} Training Loss: {train_loss:.4f}")

            # Evaluate the model on the test set
            test_loss, test_accuracy = evaluate(model, test_loader, criterion)
            print(f"Test Loss: {test_loss:.4f} Test Accuracy: {test_accuracy:.4f}")

        # Append the test loss to the list
        test_losses.append(test_loss)

    # Print the list of test set losses
    print("Test Set Losses:")
    print(test_losses)
    return test_losses


def main(train_dataset, test_dataset, input_size, output_size, width, depth, epochs,
         batch_size, num_trials, target_error):
    """
    train_dataset: training dataset
    test_dataset: test dataset
    input_size: input size
    output_size: output size
    width: width of the hidden layers
    depth: number of hidden layers
    epochs: number of training epochs
    batch_size: batch size
    num_trials: number of trials to run
    target_error: target error for inductive bias calculation
    """
    model_errors = run(train_dataset, test_dataset, input_size, output_size, width, depth, epochs, batch_size,
                       num_trials)
    return compute_inductive_bias(model_errors, target_error)