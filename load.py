import torch
import re
from model.contrastive_model import TextEmbedding
import csv
import numpy as np

# Même regex que dans le dataset original
pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(pattern)


def read_smiles(data_path):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            # On suppose que le SMILES est dans la dernière colonne
            smiles = row[-1]
            smiles_data.append(smiles)
    print("Len SMILES", len(smiles_data))
    return smiles_data


def load_vocab(vocab_path="config/vocab.txt"):
    word2index = {}
    with open(vocab_path, "r") as f:
        for line in f:
            token, idx = line.strip().split("\t")
            word2index[token] = int(idx)
    return word2index


def smiles_to_tensor(smiles, word2index, max_length=100):
    tokens = regex.findall(smiles)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    tokens += ["<"] * (max_length + 1 - len(tokens))
    ids = [word2index.get(token, word2index.get("unk", 0)) for token in tokens]
    return torch.LongTensor(ids)


def load_pretrained_weights(model, model_path, exclude_keys=("projection", "pos_emb")):
    pretrained_dict = torch.load(
        model_path, map_location="cpu"
    )  # ou map_location=device
    model_dict = model.state_dict()

    # Filtrer les clés à exclure
    filtered_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if all(exclude not in k for exclude in exclude_keys)
        and k in model_dict
        and v.size() == model_dict[k].size()
    }

    print(f"Nombre de poids chargés : {len(filtered_dict)} / {len(model_dict)}")
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)


def get_embeddings_from_smiles(
    smiles_list,
    model_path,
    output_tag,
    device="cpu",
    vocab_path="config/vocab.txt",
    max_length=100,
):
    # Charger vocabulaire
    word2index = load_vocab(vocab_path)
    vocab_size = max(word2index.values()) + 1

    # Charger modèle
    model = TextEmbedding(vocab_size)
    load_pretrained_weights(model, model_path)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for smiles in smiles_list:
            tensor = smiles_to_tensor(smiles, word2index, max_length)
            tensor = tensor.unsqueeze(0).to(device)  # batch size = 1
            emb = model(tensor, return_embedding=True)
            # print(emb.shape)
            embeddings.append(emb.squeeze(0).cpu().numpy())

    emb = np.vstack(embeddings)
    print(emb.shape)
    np.save(f"ckpt/{output_tag}-CONSMI-smi.npy", embeddings)


filepath = "/home/srkpa/projects/no-name/molcr/utils/query.csv"
filepath = "/home/srkpa/projects/no-name/.data/molcr_drugbank/graph_0.98_0.0_smi.csv"
model_path = "/home/srkpa/projects/no-name/consmi/ckpt/best_steps_checkpoint20230520.pt"
smiles_list = read_smiles(filepath)
if len(smiles_list) < 100:
    output_tag = "CQ"
else:
    output_tag = "DB"
get_embeddings_from_smiles(smiles_list, model_path, output_tag)
