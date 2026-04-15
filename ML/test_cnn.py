from ML.Data.dataset import DataStruct
from ML.Models.models import TextCNN
from ML.Models.test_models import test_model
import torch
import os

cnn_dict = torch.load(os.path.join("ML","Models","saved_models","cnn_model.pth"),weights_only = True)
cnn_config = cnn_dict["config"]
cnn_model = TextCNN(len(cnn_dict["vocab"]),
                    cnn_config["embed_dim"],
                    cnn_config["num_filters"],
                    cnn_config["dropout_rate"])

struct = DataStruct()
struct.vocab = cnn_dict["vocab"]
struct.preprocess_test_data()

test_model(cnn_model,cnn_dict,struct)