#import torch
#import torch.nn as nn
#from torchvision.models import resnet50, ResNet50_Weights

#class FaceRecognitionModel(nn.Module):
    # Explicação: Em PyTorch, os modelos são classes que herdam de nn.Module.
    # O __init__ define as camadas que o modelo usará.
 #   def __init__(self, num_classes, embedding_size=512):
  #      super(FaceRecognitionModel, self).__init__()
        
        # 1. Carrega uma ResNet-50 pré-treinada no ImageNet.
        #    Usar pesos pré-treinados (weights) acelera muito o treinamento.
   #     self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # 2. O 'backbone' original tem uma camada final ('fc') para classificar 1000 classes.
        #    Nós vamos usá-lo como um extrator de features. A camada antes da 'fc'
        #    gera 2048 features. Nós substituímos a 'fc' por uma camada de identidade,
        #    que simplesmente repassa a entrada, efetivamente nos dando o output de 2048 features.
    #    backbone_output_features = self.backbone.fc.in_features  # Geralmente é 2048
     #   self.backbone.fc = nn.Identity()

        # 3. Definimos nossas próprias camadas "cabeça" (head).
        #    Esta camada de embedding pega as 2048 features do backbone e as projeta
        #    em um espaço de embedding menor (512 dimensões), que é onde a "mágica"
        #    do reconhecimento facial acontece.
      #  self.embedding_layer = nn.Linear(backbone_output_features, embedding_size)
        
        # 4. Esta é a camada classificadora do nosso baseline.
        #    Ela pega o embedding de 512 dimensões e tenta classificar em uma das
        #    'num_classes' (480 no seu caso). É esta camada que será substituída
        #    pela camada CosFace na próxima fase.
       # self.classifier = nn.Linear(embedding_size, num_classes)
        
    # Explicação: O método 'forward' define como os dados fluem através das camadas
    # que definimos no __init__.
    #def forward(self, x):
        # Passa a imagem pelo backbone para extrair as features brutas
     #   x = self.backbone(x)
        
        # Passa as features pela camada de embedding
      #  embedding = self.embedding_layer(x)
        
        # Passa o embedding pelo classificador para obter os logits (saídas brutas)
        #logits = self.classifier(embedding)
        
        # Por enquanto, retornamos apenas os logits para o treinamento com Softmax.
        # No futuro, também vamos precisar do 'embedding'.
       # return embedding

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class FaceRecognitionModel(nn.Module):
    def __init__(self, embedding_size=512): # Não precisamos mais de 'num_classes' aqui
        super(FaceRecognitionModel, self).__init__()
        
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone_output_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.embedding_layer = nn.Linear(backbone_output_features, embedding_size)
        
    def forward(self, x):
        # Passa a imagem pelo backbone
        x = self.backbone(x)
        
        # Retorna o embedding. A camada de classificação agora está fora deste módulo.
        embedding = self.embedding_layer(x)
        return embedding