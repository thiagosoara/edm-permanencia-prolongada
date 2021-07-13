from processing import Y_base_enade_teste
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import pickle

# Carrega o arquivo com os dados
with open('base_enade.pkl', 'rb') as f:  
  X_base_enade_treinamento, Y_base_enade_treinamento, X_base_enade_teste, Y_base_enade_teste = pickle.load(f)

# Geração do Modelo baseado em Regressão Logísticas
logistic_base_enade = LogisticRegression(random_state = 1)
logistic_base_enade.fit(X_base_enade_treinamento,Y_base_enade_treinamento)
previsoes = logistic_base_enade.predict(X_base_enade_teste)

# Printa a precisão geral
print(accuracy_score(Y_base_enade_teste,previsoes))

# Desenha a matriz de confusão (Rodar no Jupyter)
cm = ConfusionMatrix(logistic_base_enade)
cm.fit(X_base_enade_treinamento,Y_base_enade_treinamento)
cm.score(X_base_enade_teste, Y_base_enade_teste)

# Printa estatísticas gerais
print(classification_report(Y_base_enade_teste, previsoes))
