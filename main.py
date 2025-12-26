import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from models.mlp import MLP5Layer
from optimizer.custom_sgd import CustomSGD
from utils.data_loader import get_mnist_loaders
from utils.logger import Logger

def main():
    parser = argparse.ArgumentParser(description='MNIST Classification with Custom SGD')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger()
    logger.log(f"Args: {args}")

    train_loader, test_loader = get_mnist_loaders(args.batch_size)
    model = MLP5Layer().to(device)
    optimizer = CustomSGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch}"):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Testing
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100. * correct / len(test_loader.dataset)
        logger.log(f"Epoch: {epoch}, Loss: {train_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()