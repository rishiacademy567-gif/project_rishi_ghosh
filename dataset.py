from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import resize_x, resize_y, batch_size

transform = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.ToTensor()
])

class BrainTumorDataset:
    pass

def get_dataloader(path, shuffle=True):
    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader