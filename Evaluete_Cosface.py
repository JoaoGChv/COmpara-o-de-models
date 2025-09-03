import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision import transforms

# Importa as classes do nosso projeto. Note que NÃO precisamos importar CosFace aqui.
from data_louder import VggFace2Dataset
from model import FaceRecognitionModel 

# --- CONFIGURAÇÕES ---
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# ATUALIZE o caminho para o seu dataset de TESTE pré-processado (ex: LFW)
PROCESSED_DATA_DIR = '/home/ubuntu/noleak/teste_cosface/test' 
# ATUALIZE o caminho para o arquivo do modelo treinado com COSFACE
MODEL_PATH = '/home/ubuntu/noleak/teste_cosface/Cosface_model_epoch_20.pth' # Exemplo de nome
EMBEDDING_SIZE = 512

# --- FUNÇÕES AUXILIARES (As mesmas do evaluate do baseline) ---

# Explicação: Esta função carrega o modelo treinado e extrai os embeddings de um dataset.
# Ela funciona perfeitamente com a nossa versão atualizada do model.py, que já retorna os embeddings.
def get_embeddings(model, loader):
    model.eval() 
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for i, (images, batch_labels) in enumerate(loader):
            images = images.to(DEVICE)
            
            embeddings_batch = model(images)
            embeddings.append(embeddings_batch.cpu())
            labels.append(batch_labels.cpu())
            
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    return embeddings, labels

# Explicação: Esta função simula a avaliação no LFW comparando pares de embeddings.
# A lógica é a mesma, pois a métrica (similaridade de cosseno) não depende de como o modelo foi treinado.
def evaluate_lfw_style(embeddings, labels):
    # (O código desta função é idêntico ao do evaluate.py do baseline)
    num_pairs = 6000
    pos_pairs, neg_pairs = [], []

    # Criando pares positivos (mesma pessoa)
    unique_labels, counts = labels.unique(return_counts=True)
    labels_with_multiple_images = unique_labels[counts >= 2]
    
    for _ in range(num_pairs // 2):
        person_id = np.random.choice(labels_with_multiple_images.numpy())
        indices = np.where(labels.numpy() == person_id)[0]
        pair = np.random.choice(indices, 2, replace=False)
        pos_pairs.append((pair[0], pair[1]))

    # Criando pares negativos (pessoas diferentes)
    for _ in range(num_pairs // 2):
        person_ids = np.random.choice(unique_labels.numpy(), 2, replace=False)
        person1_idx = np.random.choice(np.where(labels.numpy() == person_ids[0])[0])
        person2_idx = np.random.choice(np.where(labels.numpy() == person_ids[1])[0])
        neg_pairs.append((person1_idx, person2_idx))
            
    # Calcula as similaridades para todos os pares
    pos_sims = [F.cosine_similarity(embeddings[p1].unsqueeze(0), embeddings[p2].unsqueeze(0)).item() for p1, p2 in pos_pairs]
    neg_sims = [F.cosine_similarity(embeddings[p1].unsqueeze(0), embeddings[p2].unsqueeze(0)).item() for p1, p2 in neg_pairs]

    # Encontra a melhor acurácia
    thresholds = np.arange(-1.0, 1.0, 0.01)
    best_accuracy = 0
    for threshold in thresholds:
        correct_pos = sum(1 for sim in pos_sims if sim > threshold)
        correct_neg = sum(1 for sim in neg_sims if sim < threshold)
        accuracy = (correct_pos + correct_neg) / (len(pos_pairs) + len(neg_pairs))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        
    return best_accuracy


if __name__ == '__main__':
    # A normalização do teste não deve ter data augmentation
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Prepara o DataLoader para o dataset de teste
    test_dataset = VggFace2Dataset(root_dir=PROCESSED_DATA_DIR, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    # Explicação: Instanciamos a versão do FaceRecognitionModel que retorna embeddings.
    # Note que ela não precisa mais do argumento 'num_classes', pois não tem a camada classificadora.
    model = FaceRecognitionModel(embedding_size=EMBEDDING_SIZE).to(DEVICE)
    
    # Carrega os pesos do modelo treinado com CosFace
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # Extrai todos os embeddings
    print("Iniciando a inferência do modelo CosFace...")
    print("Extraindo embeddings do dataset de teste...")
    all_embeddings, all_labels = get_embeddings(model, test_loader)
    
    # Calcula a acurácia
    print("Calculando a acurácia de verificação...")
    accuracy = evaluate_lfw_style(all_embeddings, all_labels)
    
    print("-" * 30)
    print(f"Acurácia do Modelo CosFace (LMCL): {accuracy * 100:.2f}%")
    print("-" * 30)