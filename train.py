import torch
import torch.nn as nn
import torch.optim as optim

from model import BrainTumorCNN
from dataset import get_dataloader
from config import train_dir, epochs, learning_rate, weights_path


def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = BrainTumorCNN().to(device)

    loader = get_dataloader(train_dir)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), weights_path)
    print("Model saved successfully.")


if __name__ == "__main__":
    train_model()