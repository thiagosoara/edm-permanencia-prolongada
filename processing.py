import pandas as pd
import numpy as np
from pandas.core.indexes import base
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle




df_2019 = pd.read_csv('data/microdados_enade_2019.txt', sep=';', na_values=[' ',''])
df_2017 = pd.read_csv('data/MICRODADOS_ENADE_2017.txt', sep=';', na_values=[' ',''])
df_2014 = pd.read_csv('data/MICRODADOS_ENADE_2014.txt', sep=';', na_values=[' ',''])

base_enade_2019 = pd.DataFrame()

#APLICAR FILTRO

# CO_ORGACAD = 10028 (Universidade)
# CURSO = ENGENHARIA DA COMPUTAÇÂO
# AREA = CO_GRUPO = 4003

df_2019 = df_2019.query('(CO_ORGACAD==10028) and (CO_GRUPO==4003) and (CO_CATEGAD==10001 or CO_CATEGAD==10002 or CO_CATEGAD==10003 or CO_CATEGAD==115 or CO_CATEGAD==116 or CO_CATEGAD==895 or CO_CATEGAD==94)')

#df_2019 = df_2019.loc[df_2019['CO_ORGACAD'] == 10028]
#df_2019 = df_2019.loc[df_2019['CO_GRUPO'] == 4003]
#df_2019 = df_2019.loc[df_2019['CO_CATEGAD'].isin([10001,10002,10003,115,116,895,94])]


base_enade_2019['ANO_PROVA'] = df_2019['NU_ANO']
base_enade_2019['ANO_ENTRADA'] = df_2019['ANO_IN_GRAD']
base_enade_2019['SEXO'] = df_2019['TP_SEXO']
base_enade_2019['IDADE'] = df_2019['NU_IDADE']
base_enade_2019['RACA'] = df_2019['QE_I02']
base_enade_2019['ESCOLARIDADE_PAI'] = df_2019['QE_I04']
base_enade_2019['ESCOLARIDADE_MAE'] = df_2019['QE_I05']
base_enade_2019['RENDA_FAMILIAR'] = df_2019['QE_I08']
base_enade_2019['BOLSA_ESTUDANTIL'] = df_2019['QE_I12']
base_enade_2019['INTERCAMBIO'] = df_2019['QE_I14']
base_enade_2019['TRABALHO_DURANTE_GRAD'] = df_2019['QE_I10']
base_enade_2019['COTAS'] = df_2019['QE_I15']
base_enade_2019['DURACAO_PERMANENCIA'] = base_enade_2019['ANO_PROVA'] - base_enade_2019['ANO_ENTRADA'] 
base_enade_2019['PERMANENCIA_PROLONGADA'] =  np.where(base_enade_2019['DURACAO_PERMANENCIA'] >= 6, 1, 0)

base_enade_2017 = pd.DataFrame()

df_2017 = df_2017.query('(CO_ORGACAD==10028) and (CO_GRUPO==4003) and (CO_CATEGAD==1 or CO_CATEGAD==2 or CO_CATEGAD==3)')
#df_2017 = df_2017.loc[df_2017['CO_ORGACAD'] == 10028]
#df_2017 = df_2017.loc[df_2017['CO_GRUPO'] == 4003]

base_enade_2017['ANO_PROVA'] = df_2017['NU_ANO']
base_enade_2017['ANO_ENTRADA'] = df_2017['ANO_IN_GRAD']
base_enade_2017['SEXO'] = df_2017['TP_SEXO']
base_enade_2017['IDADE'] = df_2017['NU_IDADE']
base_enade_2017['RACA'] = df_2017['QE_I02']
base_enade_2017['ESCOLARIDADE_PAI'] = df_2017['QE_I04']
base_enade_2017['ESCOLARIDADE_MAE'] = df_2017['QE_I05']
base_enade_2017['RENDA_FAMILIAR'] = df_2017['QE_I08']
base_enade_2017['BOLSA_ESTUDANTIL'] = df_2017['QE_I12']
base_enade_2017['INTERCAMBIO'] = df_2017['QE_I14']
base_enade_2017['TRABALHO_DURANTE_GRAD'] = df_2017['QE_I10']
base_enade_2017['COTAS'] = df_2017['QE_I15']
base_enade_2017['DURACAO_PERMANENCIA'] = base_enade_2017['ANO_PROVA'] - base_enade_2017['ANO_ENTRADA'] 
base_enade_2017['PERMANENCIA_PROLONGADA'] =  np.where(base_enade_2017['DURACAO_PERMANENCIA'] >= 6, 1, 0)


base_enade_2014 = pd.DataFrame()


df_2014 = df_2014.query('(CO_ORGACAD==10028) and (CO_GRUPO==5809) and (CO_CATEGAD==93 or CO_CATEGAD==116 or CO_CATEGAD==10001 or CO_CATEGAD==10002 or CO_CATEGAD==10003)')
#df_2014 = df_2014.loc[df_2014['CO_ORGACAD'] == 10028]
#df_2014 = df_2014.loc[df_2014['CO_GRUPO'] == 5809]

base_enade_2014['ANO_PROVA'] = df_2014['NU_ANO']
base_enade_2014['ANO_ENTRADA'] = df_2014['ANO_IN_GRAD']
base_enade_2014['SEXO'] = df_2014['TP_SEXO']
base_enade_2014['IDADE'] = df_2014['NU_IDADE']
base_enade_2014['RACA'] = df_2014['QE_I02']
base_enade_2014['ESCOLARIDADE_PAI'] = df_2014['QE_I04']
base_enade_2014['ESCOLARIDADE_MAE'] = df_2014['QE_I05']
base_enade_2014['RENDA_FAMILIAR'] = df_2014['QE_I08']
base_enade_2014['BOLSA_ESTUDANTIL'] = df_2014['QE_I12']
base_enade_2014['INTERCAMBIO'] = df_2014['QE_I14']
base_enade_2014['TRABALHO_DURANTE_GRAD'] = df_2014['QE_I10']
base_enade_2014['COTAS'] = df_2014['QE_I15']
base_enade_2014['DURACAO_PERMANENCIA'] = base_enade_2014['ANO_PROVA'] - base_enade_2014['ANO_ENTRADA'] 
base_enade_2014['PERMANENCIA_PROLONGADA'] =  np.where(base_enade_2014['DURACAO_PERMANENCIA'] >= 6, 1, 0)


#juntar dataframe
print(base_enade_2014.describe(include="all"))
print(base_enade_2017.describe(include="all"))
print(base_enade_2019.describe(include="all"))

base_enade = pd.DataFrame()

base_enade = base_enade.append(base_enade_2014)
base_enade = base_enade.append(base_enade_2017)
base_enade = base_enade.append(base_enade_2019)
base_enade = base_enade.dropna().reset_index(drop=True)

base_enade['IDADE'] = base_enade['IDADE'] - base_enade['DURACAO_PERMANENCIA']

# IMPRIME O CSV PARA BI
print(base_enade)
base_enade.to_csv('result_bi.csv', index=False)

# INICIA TRATAMENTO DOS DADOS PARA ALGORITMOS
base_enade = base_enade.drop(
    columns=['DURACAO_PERMANENCIA', 'ANO_PROVA', 'ANO_ENTRADA'], axis=1,
)

# DIVISÃO ENTRE FEATURES E CLASSE ALVO
X_base_enade = base_enade.iloc[:, 0:10].values
Y_base_enade = base_enade.iloc[:,10].values


# LABEL ENCODER
##CATEGORIAS ORDINAIS
label_encoder_escolaridade_pai = LabelEncoder()
label_encoder_escolaridade_mae = LabelEncoder()
X_base_enade[:,3] = label_encoder_escolaridade_pai.fit_transform(X_base_enade[:,3])
X_base_enade[:,4] = label_encoder_escolaridade_mae.fit_transform(X_base_enade[:,4])
#Renda familiar

##CATEGORIAS 
label_encoder_sexo = LabelEncoder()
label_encoder_raca = LabelEncoder()
label_encoder_renda_familiar = LabelEncoder()
label_encoder_bolsa_estudantil = LabelEncoder()
label_encoder_intercambio = LabelEncoder()
label_encoder_trabalho_durante_grad = LabelEncoder()
label_encoder_cotas = LabelEncoder()
X_base_enade[:,0] = label_encoder_sexo.fit_transform(X_base_enade[:,0])
X_base_enade[:,2] = label_encoder_raca.fit_transform(X_base_enade[:,2])
X_base_enade[:,5] = label_encoder_renda_familiar.fit_transform(X_base_enade[:,5])
X_base_enade[:,6] = label_encoder_bolsa_estudantil.fit_transform(X_base_enade[:,6])
X_base_enade[:,7] = label_encoder_intercambio.fit_transform(X_base_enade[:,7])
X_base_enade[:,8] = label_encoder_trabalho_durante_grad.fit_transform(X_base_enade[:,8])
X_base_enade[:,9] = label_encoder_cotas.fit_transform(X_base_enade[:,9])

# ONE HOT ENCODER
onehotencoder_base_enade = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0,2,5,6,7,8,9])], remainder='passthrough')
X_base_enade = onehotencoder_base_enade.fit_transform(X_base_enade).toarray()

# Padronização (Standardization)
scaler_base_enade = StandardScaler()
X_base_enade = scaler_base_enade.fit_transform(X_base_enade)

# Separação base TREINAMENTO e TESTE
X_base_enade_treinamento, X_base_enade_teste, Y_base_enade_treinamento, Y_base_enade_teste = train_test_split(X_base_enade, Y_base_enade, test_size=0.20, random_state=0)

print(X_base_enade.shape)
print(X_base_enade_treinamento.shape, Y_base_enade_treinamento.shape)
print(X_base_enade_teste.shape, Y_base_enade_teste.shape)

with open('base_enade.pkl', mode = 'wb') as f:
    pickle.dump([X_base_enade_treinamento, Y_base_enade_treinamento, X_base_enade_teste, Y_base_enade_teste], f)
#np.savetxt("result_features_algoritmo.csv", X_base_enade, delimiter=",")
#np.savetxt("result_class_algoritmo.csv", Y_base_enade, delimiter=",")

