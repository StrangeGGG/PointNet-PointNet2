import torch
from PointNet import PointNet, pointnetloss
from torch.utils.data import DataLoader
from Dataset import PointCloudDataset
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
from Temp import classes
from sklearn.metrics import classification_report
import pandas as pd

# Load the dataset
data_path = 'ModelNet40'

test_dataset = PointCloudDataset(data_path, valid=True, get_testset=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=64, num_workers=0, pin_memory=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
pointnet = PointNet()
pointnet.to(device)

# Load a pre-trained model if it exists
if os.path.exists('save_10.pth'):
    pointnet.load_state_dict(torch.load('save_10.pth'))
    print('Loaded Pre-trained PointNet Model!')

pointnet.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for data in tqdm(test_loader, desc="Testing", unit="batch"):
        inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
        outputs, __, __ = pointnet(inputs.transpose(1, 2))
        _, preds = torch.max(outputs.data, 1)
        all_preds += list(preds.cpu().numpy())  # Move predictions back to CPU for numpy
        all_labels += list(labels.cpu().numpy())  # Same for labels

cm = confusion_matrix(all_labels, all_preds)
total_accuracy = np.trace(cm) / np.sum(cm)
print(f"\nTotal Accuracy: {total_accuracy:.4f} ({total_accuracy*100:.2f}%)")

# function from https://deeplizard.com/learn/video/0LhiS6yu2qQ
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plt.figure(figsize=(16,16))
plot_confusion_matrix(cm, list(classes.keys()), normalize=False)
plt.savefig('confusion_matrix_10.png', bbox_inches='tight', dpi=300)
plt.close()

report_dict = classification_report(all_labels, all_preds, target_names=list(classes.keys()), output_dict=True, zero_division=0)

# 转换为DataFrame（排除'accuracy'和'avg'行）
df = pd.DataFrame(report_dict).transpose()
df.to_csv('classification_report_non_normalized_10.csv', index_label='class')