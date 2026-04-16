from ML.Models.train_models import train_model
from ML.Models.models import TextCNN
from ML.Data.dataset import DataStruct

# Store all dataset information
struct = DataStruct()
struct.build_vocab(min_freq=5,max_vocab=5000)
struct.preprocess_train_data()

cnn_config = {
    "embed_dim" : 100,
    "num_filters" : 64,
    "dropout_rate" : 0.5
}

cnn_model = TextCNN(
    struct.get_vocab(),
    cnn_config["embed_dim"],
    cnn_config["num_filters"],
    cnn_config["dropout_rate"]
)

train_model(name="cnn",
            model=cnn_model,
            model_config=cnn_config,
            data_struct=struct,
            lr=1e-4,
            weight_decay=1e-5,
            epochs=1200,
            patience=20)
