import torch
from PointNet2OnlySA1 import PointNet2OnlySA1
from torch.utils.data import DataLoader, random_split
from Dataset import PointCloudDataset
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Load the dataset
data_path = 'ModelNet40'

train_dataset = PointCloudDataset(data_path)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_subset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(dataset=val_subset, batch_size=32, num_workers=0, pin_memory=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
pointnet2 = PointNet2OnlySA1()
pointnet2.to(device)
optimizer = torch.optim.Adam(pointnet2.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
max_grad_norm = 1.0

# Load a pre-trained model if it exists
# if os.path.exists('save.pth'):
#     pointnet.load_state_dict(torch.load('save.pth'))
#     print('Loaded Pre-trained PointNet Model!')

def train(model, train_loader, val_loader=None, epochs=10):

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    mini_batch_train_losses = []
    mini_batch_val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_loss = 0.0
        total = 0
        correct = 0

        # Wrap train_loader with tqdm for progress bar
        train_progress = tqdm(enumerate(train_loader, 0),
                             total=len(train_loader),
                             desc=f'Epoch {epoch + 1}/{epochs}',
                             leave=True)

        # for i, data in enumerate(train_loader, 0):
        for i, data in train_progress:
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            inputs = inputs.contiguous()
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            mini_batch_train_losses.append(loss.item())

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:    # print every 5 mini-batches
                # print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                #     (epoch + 1, i + 1, len(train_loader), running_loss / 5))
                train_progress.write(
                    f'[Epoch: {epoch + 1}, Batch: {i + 1}/{len(train_loader)}], loss: {running_loss / 5:.3f}'
                )
                running_loss = 0.0

        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(100 * correct/total)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}, "
              f"Accuracy: {100 * correct/total:.2f}%")

        model.eval()
        correct = total = 0

        val_loss = 0.0

        # validation
        if val_loader:
            val_progress = tqdm(val_loader, desc='Validating', leave=False)
            with torch.no_grad():
                # for data in val_loader:
                for data in val_progress:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    # outputs, __, __ = model(inputs.transpose(1,2))

                    inputs = inputs.contiguous()

                    outputs = model(inputs)

                    # Calculate validation loss
                    loss = criterion(outputs, labels)

                    mini_batch_val_losses.append(loss.item())

                    val_loss += loss.item()

                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            val_acc = 100. * correct / total
            print('Valid accuracy: %d %%' % val_acc)
            val_accuracies.append(val_acc)
            val_losses.append(val_loss / len(val_loader))
            scheduler.step(val_loss / len(val_loader))

    # save the model
    torch.save(model.state_dict(), "onlysa1_model.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot
    plt.savefig('onlysa1_loss_curve.png')
    plt.close()

    # plt.figure(figsize=(10, 5))
    # plt.plot(mini_batch_train_losses, label='Training Loss')
    # plt.title('Training  Loss')
    # plt.xlabel('mini-batches')
    # plt.ylabel('Loss')
    # plt.legend()
    #
    # # Save the plot
    # plt.savefig('PointNetPlusPlus_training_loss.png')
    # plt.close()
    #
    # plt.figure(figsize=(10, 5))
    # plt.plot(mini_batch_val_losses, label='Validation Loss')
    # plt.title('Validation Loss')
    # plt.xlabel('mini-batches')
    # plt.ylabel('Loss')
    # plt.legend()
    #
    # # Save the plot
    # plt.savefig('PointNetPlusPlus_validation_loss.png')
    # plt.close()


train(pointnet2, train_loader, valid_loader, epochs=10)