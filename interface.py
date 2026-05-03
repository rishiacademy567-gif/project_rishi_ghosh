from model import BrainTumorCNN as TheModel
from train import train_model as the_trainer
from predict import predict_batch as the_predictor
from dataset import BrainTumorDataset as TheDataset
from dataset import get_dataloader as the_dataloader
from config import batch_size as the_batch_size
from config import epochs as total_epochs