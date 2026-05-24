import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import os

def evaluate_model(model, loader, device, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            
            preds = torch.round(torch.sigmoid(outputs))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, acc, prec, rec, f1

def train():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    BATCH_SIZE = 32
    EPOCHS = 10
    
    # Data Augmentation for training, simple Resize & Normalize for validation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    base_dir = "1GB_dataset/ela"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    
    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    print("Loading pre-trained EfficientNet-B0...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Replace classifier head first
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    
    # Unfreeze the whole model so all layers can adapt to ELA noise patterns
    for param in model.parameters():
        param.requires_grad = True
    
    model = model.to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Use differential learning rates: fine-tune features slowly, train classifier faster
    optimizer = optim.Adam([
        {'params': model.features.parameters(), 'lr': 5e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ])
    
    # Cosine Annealing scheduler to smoothly decay the learning rates
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_f1 = 0.0
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float()
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            if (i+1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        train_loss = running_loss / len(train_dataset)
        
        # Validation phase
        val_loss, acc, prec, rec, f1 = evaluate_model(model, val_loader, DEVICE, criterion)
        
        # Step the learning rate scheduler
        scheduler.step()
        
        epoch_time = time.time() - start_time
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        print(f"Epoch [{epoch+1}/{EPOCHS}] ({epoch_time:.0f}s) | LRs: {current_lrs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
        print("-" * 50)
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved new best model!")
            
    print(f"Training completed. Best F1 Score: {best_f1:.4f}")
    
    # Load best model for final evaluation output
    model.load_state_dict(torch.load("best_model.pth"))
    val_loss, acc, prec, rec, f1 = evaluate_model(model, val_loader, DEVICE, criterion)
    
    # Save final metrics
    with open("results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Best Validation F1: {best_f1:.4f}\n")
if __name__ == "__main__":
    train()
