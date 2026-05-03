batch_size = 16
epochs = 10
learning_rate = 0.001
resize_x = 128
resize_y = 128
input_channels = 3
num_classes = 4

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

train_dir = 'dataset/Training'
test_dir = 'dataset/Testing'

weights_path = 'checkpoints/final_weights.pth'