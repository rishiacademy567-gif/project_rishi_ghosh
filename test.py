import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from model import BrainTumorCNN
from dataset import get_dataloader
from config import test_dir, weights_path, class_names

def evaluate_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BrainTumorCNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    loader = get_dataloader(test_dir, shuffle=False)
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    print('Accuracy:', accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=class_names))
    print('Confusion Matrix:', confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
    evaluate_model()