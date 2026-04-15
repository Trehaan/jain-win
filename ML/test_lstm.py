from ML.Data.dataset import DataStruct
from ML.Models.models import TextLSTM
from ML.Models.test_models import test_model
import torch
import os

lstm_dict = torch.load(os.path.join("ML","Models","saved_models","lstm_model.pth"),weights_only = True)
lstm_config = lstm_dict["config"]
lstm_model = TextLSTM(len(lstm_dict["vocab"]),
                    lstm_config["embed_dim"],
                    lstm_config["hidden_size"],
                    lstm_config["num_layers"],
                    lstm_config["dropout_rate"])

struct = DataStruct()
struct.vocab = lstm_dict["vocab"]
struct.preprocess_test_data()

test_model(lstm_model,lstm_dict,struct)