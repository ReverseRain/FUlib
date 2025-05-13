import torch
import copy
import os
import matplotlib.pyplot as plt
from flcore.trainmodel.models import *

def drawSimilar():
    model = CNN(in_features=3, num_classes=10, dim=1600)
    head = copy.deepcopy(model.fc)
    model.fc = nn.Identity()
    model = BaseHeadSplit(model, head)

    model_path = os.path.join("models", "Cifar10")
    model_path = os.path.join(model_path, "1" + "_server" + ".pt")
    model = torch.load(model_path)


    model2 = CNN(in_features=3, num_classes=10, dim=1600)
    head2 = copy.deepcopy(model2.fc)
    model2.fc = nn.Identity()
    model2 = BaseHeadSplit(model2, head2)

    model_path = os.path.join("models", "Cifar10")
    model_path = os.path.join(model_path, "retrain_model" + ".pt")
    model2 = torch.load(model_path)


    cos_sims = {}
    layer_names=[name for name, param in model.named_parameters()]
    pretrained_weights = {name: param.data for name, param in model.named_parameters()}
    retrained_weights = {name: param.data for name, param in model2.named_parameters()}
    for layer_name in layer_names:
        vec_pretrained = pretrained_weights[layer_name].flatten()
        vec_retrained = retrained_weights[layer_name].flatten()

        sim = F.cosine_similarity(vec_pretrained.unsqueeze(0), 
                                    vec_retrained.unsqueeze(0), 
                                    dim=1).item()
        cos_sims[layer_name] = sim



    sorted_layers = sorted(cos_sims.items(), key=lambda x: x[1], reverse=True)
    layers = [x[0] for x in sorted_layers]
    scores = [x[1] for x in sorted_layers]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(layers, scores, color='blue', alpha=0.7)


    plt.title("Layer-wise Cosine Similarity between Pretrained and Retrained Models", fontsize=14)
    plt.xlabel("Layer Names", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()