import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CosFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, input, label):
        # 1. Normaliza as features (input) e os pesos (weight)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # --- LÓGICA CORRIGIDA ---
        # Explicação: Em vez de usar scatter_ que pode ser problemático para o gradiente,
        # vamos usar uma abordagem com "one-hot encoding". É mais explícito e seguro.
        
        # 2. Cria um tensor one-hot com a mesma forma dos cossenos
        one_hot = torch.zeros(cosine.size(), device=input.device)
        # O .long() é importante para garantir que os rótulos sejam do tipo correto para indexação
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # 3. Aplica a margem apenas onde o one-hot é 1 (ou seja, na classe correta)
        #    (cosine - one_hot * self.m) subtrai a margem 'm' somente dos logits da classe correta.
        output = (cosine - one_hot * self.m)
        
        # 4. Escala todos os logits pelo fator 's'
        output *= self.s
        
        return output

    # Adicione este método para manter a compatibilidade se precisar depurar
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'