from ML.Models.train_models import train_model
from ML.Models.models import TextLSTM
from ML.Data.dataset import DataStruct

# Store all dataset information
struct = DataStruct()
struct.build_vocab(min_freq=5,max_vocab=5000)
struct.preprocess_train_data()

lstm_config = {
    "embed_dim": 100,
    "hidden_size": 72,
    "num_layers": 1,
    "dropout_rate": 0.3
}


lstm_model = TextLSTM(
    struct.get_vocab(),
    lstm_config["embed_dim"],
    lstm_config["hidden_size"],
    lstm_config["num_layers"],
    lstm_config["dropout_rate"]
)

train_model(
    name="lstm",
    model=lstm_model,
    model_config=lstm_config,
    data_struct=struct,
    lr=1e-5,
    weight_decay=1e-3,
    epochs=2000,
    patience=35,
    batch_size=512
)
