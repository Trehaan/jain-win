from django.shortcuts import render
from ML.Models.models import TextCNN
from ML.Data.data_processer import preprocess_text,tokenize_text,pad_sequence
import torch
import os

# Create your views here.

def index(request):

    context = {"text" : ""}

    if request.method == "POST":
        text = request.POST.get("text")

        cnn_dict = torch.load(os.path.join("ML","Models","saved_models","cnn_model.pth"),weights_only = True)
        cnn_config = cnn_dict["config"]
        vocab = cnn_dict["vocab"]
        cnn_model = TextCNN(len(vocab),
                            cnn_config["embed_dim"],
                            cnn_config["num_filters"],
                            cnn_config["dropout_rate"])
        
        cnn_model.load_state_dict(cnn_dict["model_state_dict"])
        
        cleaned = preprocess_text(text)
        sequence = tokenize_text(cleaned,vocab)
        padded = pad_sequence(sequence,150)

        review = torch.tensor(padded, dtype = torch.long).unsqueeze(0)

        prob,pred = cnn_model.predictReview(review)

        context.update({"prob" : prob, "pred" : pred, "text" : text})

    return render(request,"index.html",context)