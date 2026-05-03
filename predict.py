import torch
from PIL import Image
from torchvision import transforms
from model import BrainTumorCNN
from config import resize_x, resize_y, class_names, weights_path

transform = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.ToTensor()
])


def predict_batch(list_of_img_paths):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BrainTumorCNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    results = []
    for path in list_of_img_paths:
        img = Image.open(path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img)
            pred = out.argmax(1).item()
        results.append(class_names[pred])
    return results