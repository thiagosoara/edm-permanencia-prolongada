import pandas as pd
import numpy as np
from pandas.core.indexes import base
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

df_2019 = pd.read_csv('data/microdados_enade_2019.txt', sep=';', na_values=[' ',''])
df_2017 = pd.read_csv('data/MICRODADOS_ENADE_2017.txt', sep=';', na_values=[' ',''])
df_2014 = pd.read_csv('data/MICRODADOS_ENADE_2014.txt', sep=';', na_values=[' ',''])
df_2011 = pd.read_csv('data/MICRODADOS_ENADE_2011.txt', sep=';', na_values=[' ',''])


base_enade_2019 = pd.DataFrame()

#print(df_2019)
#print(df_2017)
#print(df_2014)
#print(df_2011)

#APLICAR FILTRO

# CO_ORGACAD = 10028 (Universidade)
# CURSO = ENGENHARIA DA COMPUTAÇÂO
# AREA = CO_GRUPO = 4003

df_2019 = df_2019.loc[df_2019['CO_ORGACAD'] == 10028]
df_2019 = df_2019.loc[df_2019['CO_GRUPO'] == 4003]

base_enade_2019['ANO_PROVA'] = df_2019['NU_ANO']
base_enade_2019['ANO_ENTRADA'] = df_2019['ANO_IN_GRAD']
base_enade_2019['SEXO'] = df_2019['TP_SEXO']
base_enade_2019['IDADE'] = df_2019['NU_IDADE']
base_enade_2019['RACA'] = df_2019['QE_I02']
base_enade_2019['ESCOLARIDADE_PAI'] = df_2019['QE_I04']
base_enade_2019['ESCOLARIDADE_MAE'] = df_2019['QE_I05']
base_enade_2019['RENDA_FAMILIAR'] = df_2019['QE_I08']
base_enade_2019['BOLSA_ESTUDANTIL'] = df_2019['QE_I12']
#base_enade_2019['INTERCAMBIO'] = df_2019['QE_I14']
base_enade_2019['TRABALHO_DURANTE_GRAD'] = df_2019['QE_I10']
base_enade_2019['COTAS'] = df_2019['QE_I15']
base_enade_2019['DURACAO_PERMANENCIA'] = base_enade_2019['ANO_PROVA'] - base_enade_2019['ANO_ENTRADA'] 
base_enade_2019['ALVO'] =  np.where(base_enade_2019['DURACAO_PERMANENCIA'] >= 6, 'PROLONGADA','PADRÃO')
#base_enade_2019 = base_enade_2019.dropna()
#print(base_enade_2019.describe())

base_enade_2017 = pd.DataFrame()

df_2017 = df_2017.loc[df_2017['CO_ORGACAD'] == 10028]
df_2017 = df_2017.loc[df_2017['CO_GRUPO'] == 4003]

base_enade_2017['ANO_PROVA'] = df_2017['NU_ANO']
base_enade_2017['ANO_ENTRADA'] = df_2017['ANO_IN_GRAD']
base_enade_2017['SEXO'] = df_2017['TP_SEXO']
base_enade_2017['IDADE'] = df_2017['NU_IDADE']
base_enade_2017['RACA'] = df_2017['QE_I02']
base_enade_2017['ESCOLARIDADE_PAI'] = df_2017['QE_I04']
base_enade_2017['ESCOLARIDADE_MAE'] = df_2017['QE_I05']
base_enade_2017['RENDA_FAMILIAR'] = df_2017['QE_I08']
base_enade_2017['BOLSA_ESTUDANTIL'] = df_2017['QE_I12']
#base_enade_2017['INTERCAMBIO'] = df_2017['QE_I14']
base_enade_2017['TRABALHO_DURANTE_GRAD'] = df_2017['QE_I10']
base_enade_2017['COTAS'] = df_2017['QE_I15']
base_enade_2017['DURACAO_PERMANENCIA'] = base_enade_2017['ANO_PROVA'] - base_enade_2017['ANO_ENTRADA'] 
base_enade_2017['ALVO'] =  np.where(base_enade_2017['DURACAO_PERMANENCIA'] >= 6, 'PROLONGADA','PADRÃO')


base_enade_2014 = pd.DataFrame()

df_2014 = df_2014.loc[df_2014['CO_ORGACAD'] == 10028]
df_2014 = df_2014.loc[df_2014['CO_GRUPO'] == 5809]

base_enade_2014['ANO_PROVA'] = df_2014['NU_ANO']
base_enade_2014['ANO_ENTRADA'] = df_2014['ANO_IN_GRAD']
base_enade_2014['SEXO'] = df_2014['TP_SEXO']
base_enade_2014['IDADE'] = df_2014['NU_IDADE']
base_enade_2014['RACA'] = df_2014['QE_I02']
base_enade_2014['ESCOLARIDADE_PAI'] = df_2014['QE_I04']
base_enade_2014['ESCOLARIDADE_MAE'] = df_2014['QE_I05']
base_enade_2014['RENDA_FAMILIAR'] = df_2014['QE_I08']
base_enade_2014['BOLSA_ESTUDANTIL'] = df_2014['QE_I12']
#base_enade_2014['INTERCAMBIO'] = df_2014['QE_I14']
base_enade_2014['TRABALHO_DURANTE_GRAD'] = df_2014['QE_I10']
base_enade_2014['COTAS'] = df_2014['QE_I15']
base_enade_2014['DURACAO_PERMANENCIA'] = base_enade_2014['ANO_PROVA'] - base_enade_2014['ANO_ENTRADA'] 
base_enade_2014['ALVO'] =  np.where(base_enade_2014['DURACAO_PERMANENCIA'] >= 6, 'PROLONGADA','PADRÃO')

base_enade_2011 = pd.DataFrame()

df_2011 = df_2011.loc[df_2011['CO_ORGACAD'] == 10028]
df_2011 = df_2011.loc[df_2011['CO_GRUPO'] == 5809]

base_enade_2011['ANO_PROVA'] = df_2011['NU_ANO']
base_enade_2011['ANO_ENTRADA'] = df_2011['ANO_IN_GRAD']
base_enade_2011['SEXO'] = df_2011['TP_SEXO']
base_enade_2011['IDADE'] = df_2011['NU_IDADE']
base_enade_2011['RACA'] = df_2011['QE_I02']
base_enade_2011['ESCOLARIDADE_PAI'] = df_2011['QE_I13']
base_enade_2011['ESCOLARIDADE_MAE'] = df_2011['QE_I14']
base_enade_2011['RENDA_FAMILIAR'] = df_2011['QE_I05']
base_enade_2011['BOLSA_ESTUDANTIL'] = df_2011['QE_I11']
#base_enade_2011['INTERCAMBIO'] = df_2011['QE_I14']
base_enade_2011['TRABALHO_DURANTE_GRAD'] = df_2011['QE_I07']
base_enade_2011['COTAS'] = df_2011['QE_I12']
base_enade_2011['DURACAO_PERMANENCIA'] = base_enade_2011['ANO_PROVA'] - base_enade_2011['ANO_ENTRADA'] 
base_enade_2011['PERMANENCIA_PROLONGADA'] =  np.where(base_enade_2011['DURACAO_PERMANENCIA'] >= 6, 1, 0)


#juntar dataframe
base_enade = pd.DataFrame()
base_enade = base_enade.append(base_enade_2019)
base_enade = base_enade.append(base_enade_2017)
base_enade = base_enade.append(base_enade_2014)
base_enade = base_enade.append(base_enade_2011)
base_enade = base_enade.dropna()

print(base_enade.describe())
print(base_enade.isnull().sum())
base_enade.to_csv('result.csv', index=False)
