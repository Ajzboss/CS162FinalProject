import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
#for confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,attnetion_mask, y,original_text) in enumerate(dataloader):
        #print(X)
        X, y = X.to(device), y.to(device)
        if torch.isnan(X).any() or torch.isinf(X).any():
                raise ValueError("Input data contains NaN or infinite values.")
        # Compute prediction error
        # print(f"Device: {device}")
        pred = model(X).to(device)

        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        # for param in model.parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
        #             print(f"Param.grad that is nan or inf:{param.grad} ")
        #torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite =True)
        optimizer.step()
        optimizer.zero_grad()
        #print(loss)
        loss, current = loss.item(), (batch + 1) * len(X)
        #print(f"loss: {loss}  [{current}/{size}]")

        if batch % 128 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            #     print("Using GPU:", torch.cuda.get_device_name(device))
            #     print("GPU Memory Usage:")
            #     print("Allocated:", round(torch.cuda.memory_allocated(device)/1024**3,1), "GB")
            #     print("Cached:   ", round(torch.cuda.memory_reserved(device)/1024**3,1), "GB")
        del loss,current,pred
    #print("Batching Complete")


import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def test(dataloader, model, loss_fn):
    model.eval()
    device = next(model.parameters()).device
    test_loss, correct = 0.0, 0
    all_pred, all_labels = [], []
    
    with torch.no_grad():
        num_batches = len(dataloader)
        size = len(dataloader.dataset)
        
        for i, (X, _, y, _) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)

            # Accumulate predictions and labels
            all_pred.extend(torch.argmax(pred, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            # Update loss and accuracy
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # Progress logging
            print(f"Progress: {i+1}/{num_batches}, Accuracy: {(100 * correct / size):.1f}%, Avg Loss: {test_loss / (i+1):.6f}")
        
        # Final metrics
        test_loss /= num_batches
        accuracy = correct / size

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_pred)
        class_names = ['Human', 'Machine']
        confusion_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        confusion_df.index.name = 'True Label'
        confusion_df.columns.name = 'Predicted Label'
        sns.heatmap(confusion_df, annot=True, fmt='3g', cmap="Blues")
        plt.title("Confusion Matrix")
        plt.show()

        print(f"\nTest Accuracy: {100 * accuracy:.1f}%")
        print(f"Average Test Loss: {test_loss:.6f}")
