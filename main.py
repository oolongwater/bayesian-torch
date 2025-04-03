import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from bayesian_torch.layers import LSTMReparameterization, LinearReparameterization

# Define the Bayesian RNN model
class BayesianRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(BayesianRNN, self).__init__()
        self.lstm = LSTMReparameterization(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            prior_mean=0.0,
            prior_variance=1.0,
            posterior_mu_init=0.0,
            posterior_rho_init=-3.0,
        )
        self.fc = LinearReparameterization(
            in_features=hidden_size,
            out_features=output_size,
            prior_mean=0.0,
            prior_variance=1.0,
            posterior_mu_init=0.0,
            posterior_rho_init=-3.0,
        )

    def forward(self, x):
        kl_sum = 0
        x, (hn, cn), kl = self.lstm(x)
        kl_sum += kl
        x = x[:, -1, :]  # Take the last time step
        x, kl = self.fc(x)
        kl_sum += kl
        return x, kl_sum

# Training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, kl_sum = model(data)
        loss = criterion(output, target) + kl_sum
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Testing function
def test(model, test_loader, criterion, device, num_monte_carlo=50):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            mc_outputs = []
            for _ in range(num_monte_carlo):
                output, _ = model(data)
                mc_outputs.append(output)
            mc_outputs = torch.stack(mc_outputs)
            mean_output = mc_outputs.mean(dim=0)
            loss = criterion(mean_output, target)
            total_loss += loss.item()
            pred = mean_output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    return total_loss / len(test_loader), accuracy

# Main function
def main():
    parser = argparse.ArgumentParser(description="Train and Test Bayesian RNN")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="Mode: train or test")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-monte-carlo", type=int, default=50, help="Number of Monte Carlo samples for testing")
    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model, criterion, and optimizer
    input_size = 28  # MNIST images are 28x28
    hidden_size = 128
    output_size = 10  # 10 classes for MNIST
    model = BayesianRNN(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train or test
    if args.mode == "train":
        for epoch in range(1, args.epochs + 1):
            train_loss = train(model, train_loader, optimizer, criterion, device)
            print(f"Epoch {epoch}/{args.epochs}, Train Loss: {train_loss:.4f}")
        torch.save(model.state_dict(), "bayesian_rnn.pth")
    elif args.mode == "test":
        model.load_state_dict(torch.load("bayesian_rnn.pth"))
        test_loss, accuracy = test(model, test_loader, criterion, device, args.num_monte_carlo)
        print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()