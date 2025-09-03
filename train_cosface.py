import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# Novas bibliotecas para métricas e visualização
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# Importa as classes que vamos usar
from data_louder import VggFace2Dataset
from model import FaceRecognitionModel
from cosface import CosFace 

# --- Configurações e Hiperparâmetros ---
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PROCESSED_DATA_DIR = '/home/ubuntu/noleak/teste_cosface/train' # ATUALIZE
NUM_CLASSES = 480 

# Parâmetros de Treinamento
BATCH_SIZE = 256
NUM_EPOCHS = 20  
LEARNING_RATE = 0.05 
WEIGHT_DECAY = 5e-4

# --- Preparação dos Dados (com divisão para validação) ---
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

full_dataset = VggFace2Dataset(root_dir=PROCESSED_DATA_DIR, transform=train_transforms)
# Explicação: Dividimos o dataset em treino e validação. O artigo de exemplo
# usava 10% para validação. Vamos seguir essa proporção.
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
# Explicação: O loader de validação não precisa de shuffle.
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# --- Inicialização do Modelo e da Perda CosFace ---
model = FaceRecognitionModel().to(DEVICE)
margin_loss = CosFace(in_features=512, out_features=NUM_CLASSES, s=64.0, m=0.35).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([{'params': model.parameters()}, {'params': margin_loss.parameters()}],
                      lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

# Listas para armazenar métricas e plotar a curva de aprendizado
train_losses = []
val_losses = []
val_accuracies = []
val_f1_scores = []

# --- Função de Plotagem ---
def plot_metrics(train_losses, val_losses, val_accuracies, val_f1_scores):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Plotagem da Curva de Perda
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Perda de Treinamento')
    plt.plot(epochs, val_losses, 'r', label='Perda de Validação')
    plt.title('Curva de Aprendizado (Perda)')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    
    # Plotagem das Métricas de Validação
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'g', label='Acurácia de Validação')
    plt.plot(epochs, val_f1_scores, 'm', label='F1-Score de Validação')
    plt.title('Métricas de Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Valor da Métrica')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curve.png')
    plt.show()

# --- Loop de Treinamento Principal ---
print("Iniciando o treinamento com CosFace (LMCL)...")
for epoch in range(NUM_EPOCHS):
    model.train()
    margin_loss.train()
    running_loss = 0.0
    
    # Itera sobre os lotes de treino
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        embeddings = model(images)
        logits = margin_loss(embeddings, labels)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i + 1) % 100 == 0:
            print(f'Época [{epoch+1}/{NUM_EPOCHS}], Passo [{i+1}/{len(train_loader)}], Perda (Loss): {loss.item():.4f}')

    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    # --- Loop de Validação ---
    model.eval()
    margin_loss.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    # Explicação: Desabilita o cálculo de gradientes.
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            embeddings = model(images)
            logits = margin_loss(embeddings, labels)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            
            # Explicação: Para acurácia e F1-score, pegamos a classe com a maior probabilidade.
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_val_loss = val_loss / len(val_loader)
    val_losses.append(epoch_val_loss)
    
    # Calcula as métricas de validação
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro') # 'macro' para lidar com classes desbalanceadas
    val_accuracies.append(accuracy)
    val_f1_scores.append(f1)
    
    print(f"Fim da Época {epoch+1} | Perda Treino: {epoch_train_loss:.4f} | Perda Val.: {epoch_val_loss:.4f}")
    print(f"               | Acurácia Val.: {accuracy:.4f} | F1-Score Val.: {f1:.4f}")

    torch.save(model.state_dict(), f'cosface_model_epoch_{epoch+1}.pth') 

print("Treinamento com CosFace (LMCL) concluído!")
plot_metrics(train_losses, val_losses, val_accuracies, val_f1_scores)