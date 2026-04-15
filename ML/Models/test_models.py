from torch.utils.data import DataLoader
from ML.Data.dataset import MyDataset


def test_model(model,dict,data_struct,batch_size=32):

    model_state_dict = dict["model_state_dict"]

    test_dataset = MyDataset(data_struct.test_sequences,data_struct.text_labels)
    test_loader = DataLoader(test_dataset,batch_size = batch_size,shuffle = False)

    model.load_state_dict(model_state_dict)
    model.testOnData(test_loader=test_loader)

