import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELearner(nn.Module):
    def __init__(self):
        super().__init__()

    def trainOnData(self,train_loader,val_loader,optimizer,epochs,patience):
        
        criterion = nn.BCEWithLogitsLoss()
        
        scaler = torch.amp.GradScaler("cuda")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(device)

        count = patience
        best_val_loss = float('inf')
        for epoch in range(epochs):

            running_loss = 0.0

            self.train()

            for inputs,labels in train_loader:

                inputs = inputs.to(device, non_blocking = True)
                labels = labels.to(device, non_blocking = True)

                optimizer.zero_grad()

                with torch.amp.autocast("cuda"):
                    outputs = self(inputs).squeeze(1)
                    loss = criterion(outputs,labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)

            running_loss = 0.0
            self.eval()

            with torch.no_grad():
                for inputs,labels in val_loader:
                    inputs = inputs.to(device, non_blocking = True)
                    labels = labels.to(device, non_blocking = True)
                    
                    outputs = self(inputs).squeeze(1)
                    loss = criterion(outputs,labels)
                    running_loss += loss.item()

            val_loss = running_loss / len(val_loader)

            if val_loss > best_val_loss + 1e-3:
                count -= 1

                if count == 0:
                    print("Early stopping")
                    break;
            else:
                count = patience
                best_val_loss = val_loss

            print(f"Epoch {epoch:4} | Train Loss = {train_loss:.6f}, Validation Loss = {val_loss:.6f}")


    def testOnData(self, test_loader):

        self.eval() 

        criterion = nn.BCEWithLogitsLoss()  

        total = 0
        correct = 0
        running_loss = 0

        TP = 0
        FP = 0
        FN = 0

        with torch.no_grad():

            for inputs, labels in test_loader:

                outputs = self(inputs).squeeze(1)  

                loss = criterion(outputs, labels)
                running_loss += loss.item()

                preds = (outputs >= 0.5).int()
                labels_int = labels.int()

                # Accuracy
                correct += (preds == labels_int).sum().item()
                total += labels.size(0)

                # Metrics
                TP += ((preds == 1) & (labels_int == 1)).sum().item()
                FP += ((preds == 1) & (labels_int == 0)).sum().item()
                FN += ((preds == 0) & (labels_int == 1)).sum().item()

        accuracy = correct / total
        avg_loss = running_loss / len(test_loader)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )

        print(f"\nTest Loss  = {avg_loss:.4f}")
        print(f"Accuracy   = {accuracy:.4f}")
        print(f"Precision  = {precision:.4f}")
        print(f"Recall     = {recall:.4f}")
        print(f"F1 Score   = {f1:.4f}")

        
    def predictReview(self,input):

        self.eval()

        with torch.no_grad():
            prob = torch.sigmoid(self(input)).item()

            print("Prob = ", torch.sigmoid(self(input)).item())

        pred = "Fake" if prob >= 0.4 else "Genuine"

        return prob,pred



class TextCNN(BCELearner):
    def __init__(self, vocab_size, embed_dim, num_filters,dropout_rate):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size=3)
        self.conv2 = nn.Conv1d(embed_dim, num_filters, kernel_size=4)
        self.conv3 = nn.Conv1d(embed_dim, num_filters, kernel_size=5)
        
        self.fc = nn.Linear(num_filters * 3, 1)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.embedding(x)            #[n,l,c] 
        x = x.permute(0, 2, 1)           #[n,c,l]
        
        x1 = F.relu(self.conv1(x))
        x1 = F.max_pool1d(x1, x1.shape[2]).squeeze(2)
        
        x2 = F.relu(self.conv2(x))
        x2 = F.max_pool1d(x2, x2.shape[2]).squeeze(2)
        
        x3 = F.relu(self.conv3(x))
        x3 = F.max_pool1d(x3, x3.shape[2]).squeeze(2)
        
        x = torch.cat((x1, x2, x3), dim=1)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    

import torch
import torch.nn as nn

class TextLSTM(BCELearner):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout_rate):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        x = self.embedding(x)

        _, (h_n, _) = self.lstm(x)

        forward = h_n[-2]
        backward = h_n[-1]

        x = torch.cat((forward, backward), dim=1)

        x = self.layer_norm(x)
        x = self.dropout(x)

        x = self.fc(x)

        return x

