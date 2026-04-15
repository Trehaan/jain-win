import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from ML.Data.dataset import MyDataset
from sklearn.model_selection import train_test_split
import os


def train_model(name,model,model_config,data_struct,lr,weight_decay,epochs,patience,batch_size=1024,num_workers=0):
    
    train_sequences,val_sequences,train_labels,val_labels = train_test_split(data_struct.train_sequences,
                                                                             data_struct.train_labels,
                                                                             test_size=0.1,
                                                                             random_state=42)

    train_dataset = MyDataset(train_sequences,train_labels)
    val_dataset = MyDataset(val_sequences,val_labels)

    train_loader = DataLoader(train_dataset,
                              batch_size = batch_size,
                              shuffle = True,
                              num_workers = num_workers,
                              pin_memory = False)
    
    val_loader = DataLoader(val_dataset,
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers = num_workers,
                            pin_memory = False)


    optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)

    model.trainOnData(train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    epochs=epochs,
                    patience=patience)


    save_choice = input("\nSave model? (y/n) : ")
    if save_choice == "y":
        save_dict = {
            "model_state_dict" : model.state_dict(),
            "vocab" : data_struct.vocab,
            "config" : model_config
        }
        
        torch.save(save_dict,os.path.join("ML","Models","saved_models",name + "_model.pth"))
        print("Saved Model")
