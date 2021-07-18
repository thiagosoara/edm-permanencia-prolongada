import pandas as pd
import numpy as np
from pandas.core.indexes import base
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle


def mount_dataset():

    df_2019 = pd.read_csv(
        'data/microdados_enade_2019.txt', sep=';', na_values=[' ', '']
    )
    df_2017 = pd.read_csv(
        'data/MICRODADOS_ENADE_2017.txt', sep=';', na_values=[' ', '']
    )
    df_2014 = pd.read_csv(
        'data/MICRODADOS_ENADE_2014.txt', sep=';', na_values=[' ', '']
    )

    df_2014 = df_2014[df_2014.TP_PRES==555]
    df_2017 = df_2017[df_2017.TP_PRES==555]
    df_2019 = df_2019[df_2019.TP_PRES==555]

    # correção da base de 2017
    df_2017 = df_2017[df_2017.TP_PR_GER==555]

    # correção da base de 2014 e 2019
    df_2014 = df_2014[df_2014.TP_PR_OB_FG==555]
    df_2019 = df_2019[df_2019.TP_PR_OB_FG==555]

    # correção da base de 2014 , 2017 e 2019
    df_2014 = df_2014[df_2014.TP_PR_DI_FG==555]
    df_2017 = df_2017[df_2017.TP_PR_DI_FG==555]
    df_2019 = df_2019[df_2019.TP_PR_DI_FG==555]

    # correção da base de 2014 , 2017 e 2019
    df_2014 = df_2014[df_2014.TP_PR_DI_CE==555]
    df_2017 = df_2017[df_2017.TP_PR_DI_CE==555]
    df_2019 = df_2019[df_2019.TP_PR_DI_CE==555]

    base_enade_2019 = pd.DataFrame()

    # APLICAR FILTRO

    df_2019 = df_2019.query(
        '(CO_ORGACAD in [10028, 10019, 10020, 10026]) and '
        '(CO_GRUPO==4003) and '
        '(CO_CATEGAD in [10001, 10002, 10003, 115, 116, 895, 94])'
    )

    base_enade_2019['ANO_PROVA'] = df_2019['NU_ANO']
    base_enade_2019['ANO_ENTRADA'] = df_2019['ANO_IN_GRAD']
    base_enade_2019['SEXO'] = df_2019['TP_SEXO']
    base_enade_2019['IDADE'] = df_2019['NU_IDADE']
    base_enade_2019['ESTADO_CIVIL'] = df_2019['QE_I01']
    base_enade_2019['RACA'] = df_2019['QE_I02']
    base_enade_2019['ESCOLARIDADE_PAI'] = df_2019['QE_I04']
    base_enade_2019['ESCOLARIDADE_MAE'] = df_2019['QE_I05']
    base_enade_2019['RENDA_FAMILIAR'] = df_2019['QE_I08']
    base_enade_2019['AUXILIO_ESTUDANTIL'] = df_2019['QE_I12']
    base_enade_2019['BOLSA_ESTUDANTIL'] = df_2019['QE_I13']
    base_enade_2019['INTERCAMBIO'] = df_2019['QE_I14']
    base_enade_2019['TRABALHO_DURANTE_GRAD'] = df_2019['QE_I10']
    base_enade_2019['COTAS'] = df_2019['QE_I15']
    base_enade_2019['ENSINO_MEDIO'] = df_2019['QE_I17']
    base_enade_2019['PRINCIPAL_MOTIVACAO'] = df_2019['QE_I25']
    base_enade_2019['DURACAO_PERMANENCIA'] = (
        base_enade_2019['ANO_PROVA'] - base_enade_2019['ANO_ENTRADA']
    )
    base_enade_2019['PERMANENCIA_PROLONGADA'] = np.where(
        base_enade_2019['DURACAO_PERMANENCIA'] >= 6, 1, 0
    )

    base_enade_2017 = pd.DataFrame()

    df_2017 = df_2017.query(
        '(CO_ORGACAD in [10028, 10019, 10020, 10026]) and '
        '(CO_GRUPO==4003) and '
        '(CO_CATEGAD in [1, 2, 3])'
    )

    base_enade_2017['ANO_PROVA'] = df_2017['NU_ANO']
    base_enade_2017['ANO_ENTRADA'] = df_2017['ANO_IN_GRAD']
    base_enade_2017['SEXO'] = df_2017['TP_SEXO']
    base_enade_2017['IDADE'] = df_2017['NU_IDADE']
    base_enade_2017['ESTADO_CIVIL'] = df_2017['QE_I01']
    base_enade_2017['RACA'] = df_2017['QE_I02']
    base_enade_2017['ESCOLARIDADE_PAI'] = df_2017['QE_I04']
    base_enade_2017['ESCOLARIDADE_MAE'] = df_2017['QE_I05']
    base_enade_2017['RENDA_FAMILIAR'] = df_2017['QE_I08']
    base_enade_2017['AUXILIO_ESTUDANTIL'] = df_2017['QE_I12']
    base_enade_2017['BOLSA_ESTUDANTIL'] = df_2017['QE_I13']
    base_enade_2017['INTERCAMBIO'] = df_2017['QE_I14']
    base_enade_2017['TRABALHO_DURANTE_GRAD'] = df_2017['QE_I10']
    base_enade_2017['COTAS'] = df_2017['QE_I15']
    base_enade_2017['ENSINO_MEDIO'] = df_2017['QE_I17']
    base_enade_2017['PRINCIPAL_MOTIVACAO'] = df_2017['QE_I25']
    base_enade_2017['DURACAO_PERMANENCIA'] = (
        base_enade_2017['ANO_PROVA'] - base_enade_2017['ANO_ENTRADA']
    )
    base_enade_2017['PERMANENCIA_PROLONGADA'] = np.where(
        base_enade_2017['DURACAO_PERMANENCIA'] >= 6, 1, 0
    )

    base_enade_2014 = pd.DataFrame()

    df_2014 = df_2014.query(
        '(CO_ORGACAD in [10028, 10019, 10020, 10026]) and '
        '(CO_GRUPO==5809) and '
        '(CO_CATEGAD in [93, 116, 10001, 10002, 10003])'
    )

    base_enade_2014['ANO_PROVA'] = df_2014['NU_ANO']
    base_enade_2014['ANO_ENTRADA'] = df_2014['ANO_IN_GRAD']
    base_enade_2014['SEXO'] = df_2014['TP_SEXO']
    base_enade_2014['IDADE'] = df_2014['NU_IDADE']
    base_enade_2014['ESTADO_CIVIL'] = df_2014['QE_I01']

    QE_I02 = []
    col_QE_I02 = df_2014.QE_I02.to_list()
    for o in col_QE_I02:
        if o == 'C':
            QE_I02.append('D')
        elif o == 'D':
            QE_I02.append('C')
        else:
            QE_I02.append(o)

    base_enade_2014['RACA'] = QE_I02
    base_enade_2014['ESCOLARIDADE_PAI'] = df_2014['QE_I04']
    base_enade_2014['ESCOLARIDADE_MAE'] = df_2014['QE_I05']
    base_enade_2014['RENDA_FAMILIAR'] = df_2014['QE_I08']
    base_enade_2014['AUXILIO_ESTUDANTIL'] = df_2014['QE_I12']
    base_enade_2014['BOLSA_ESTUDANTIL'] = df_2014['QE_I13']
    base_enade_2014['INTERCAMBIO'] = df_2014['QE_I14']
    base_enade_2014['TRABALHO_DURANTE_GRAD'] = df_2014['QE_I10']
    base_enade_2014['COTAS'] = df_2014['QE_I15']
    base_enade_2014['ENSINO_MEDIO'] = df_2014['QE_I17']
    base_enade_2014['PRINCIPAL_MOTIVACAO'] = df_2014['QE_I25']
    base_enade_2014['DURACAO_PERMANENCIA'] = (
        base_enade_2014['ANO_PROVA'] - base_enade_2014['ANO_ENTRADA']
    )
    base_enade_2014['PERMANENCIA_PROLONGADA'] = np.where(
        base_enade_2014['DURACAO_PERMANENCIA'] >= 6, 1, 0
    )

    base_enade = pd.DataFrame()

    base_enade = base_enade.append(base_enade_2014)
    base_enade = base_enade.append(base_enade_2017)
    base_enade = base_enade.append(base_enade_2019)
    base_enade = base_enade.dropna().reset_index(drop=True)

    base_enade['IDADE'] = (
        base_enade['IDADE'] - base_enade['DURACAO_PERMANENCIA']
    )

    base_enade = base_enade[base_enade['DURACAO_PERMANENCIA'] > 3].reset_index(drop=True)

    return base_enade


def create_sequencing(base_enade):
    label_encoder_escolaridade = LabelEncoder()

    base_enade['ESCOLARIDADE_PAI'] = label_encoder_escolaridade.fit_transform(
        base_enade['ESCOLARIDADE_PAI']
    )
    base_enade['ESCOLARIDADE_MAE'] = label_encoder_escolaridade.transform(
        base_enade['ESCOLARIDADE_MAE']
    )

    encoder_renda_familiar = LabelEncoder()
    base_enade['RENDA_FAMILIAR'] = label_encoder_escolaridade.fit_transform(
        base_enade['RENDA_FAMILIAR']
    )

    return base_enade


def function1(base_enade):

    # INICIA TRATAMENTO DOS DADOS PARA ALGORITMOS
    base_enade = base_enade.drop(
        columns=['DURACAO_PERMANENCIA', 'ANO_PROVA', 'ANO_ENTRADA'],
        axis=1,
    )

    # DIVISÃO ENTRE FEATURES E CLASSE ALVO
    X_base_enade = base_enade.iloc[:, 0:10].values
    Y_base_enade = base_enade.iloc[:, 10].values

    # LABEL ENCODER
    ##CATEGORIAS ORDINAIS
    label_encoder_escolaridade_pai = LabelEncoder()
    label_encoder_escolaridade_mae = LabelEncoder()
    X_base_enade[:, 3] = label_encoder_escolaridade_pai.fit_transform(
        X_base_enade[:, 3]
    )
    X_base_enade[:, 4] = label_encoder_escolaridade_mae.fit_transform(
        X_base_enade[:, 4]
    )

    ##CATEGORIAS
    label_encoder_sexo = LabelEncoder()
    label_encoder_raca = LabelEncoder()
    label_encoder_renda_familiar = LabelEncoder()
    label_encoder_bolsa_estudantil = LabelEncoder()
    label_encoder_intercambio = LabelEncoder()
    label_encoder_trabalho_durante_grad = LabelEncoder()
    label_encoder_cotas = LabelEncoder()
    X_base_enade[:, 0] = label_encoder_sexo.fit_transform(X_base_enade[:, 0])
    X_base_enade[:, 2] = label_encoder_raca.fit_transform(X_base_enade[:, 2])
    X_base_enade[:, 5] = label_encoder_renda_familiar.fit_transform(
        X_base_enade[:, 5]
    )
    X_base_enade[:, 6] = label_encoder_bolsa_estudantil.fit_transform(
        X_base_enade[:, 6]
    )
    X_base_enade[:, 7] = label_encoder_intercambio.fit_transform(
        X_base_enade[:, 7]
    )
    X_base_enade[:, 8] = label_encoder_trabalho_durante_grad.fit_transform(
        X_base_enade[:, 8]
    )
    X_base_enade[:, 9] = label_encoder_cotas.fit_transform(X_base_enade[:, 9])

    # ONE HOT ENCODER
    onehotencoder_base_enade = ColumnTransformer(
        transformers=[('OneHot', OneHotEncoder(), [0, 2, 5, 6, 7, 8, 9])],
        remainder='passthrough',
    )
    X_base_enade = onehotencoder_base_enade.fit_transform(
        X_base_enade
    ).toarray()

    # Padronização (Standardization)
    scaler_base_enade = StandardScaler()
    X_base_enade = scaler_base_enade.fit_transform(X_base_enade)

    # Separação base TREINAMENTO e TESTE
    (
        X_base_enade_treinamento,
        X_base_enade_teste,
        Y_base_enade_treinamento,
        Y_base_enade_teste,
    ) = train_test_split(
        X_base_enade, Y_base_enade, test_size=0.20, random_state=0
    )

    print(X_base_enade.shape)
    print(X_base_enade_treinamento.shape, Y_base_enade_treinamento.shape)
    print(X_base_enade_teste.shape, Y_base_enade_teste.shape)

    with open('base_enade.pkl', mode='wb') as f:
        pickle.dump(
            [
                X_base_enade_treinamento,
                Y_base_enade_treinamento,
                X_base_enade_teste,
                Y_base_enade_teste,
            ],
            f,
        )
    # np.savetxt("result_features_algoritmo.csv", X_base_enade, delimiter=",")
    # np.savetxt("result_class_algoritmo.csv", Y_base_enade, delimiter=",")


def map_age(age):
    if age < 19:
        return '<19'
    if age < 22:
        return '<22'
    return '>21'


def categorize(dataset):

    dataset1 = dataset.drop(columns=['ANO_PROVA', 'ANO_ENTRADA', 'DURACAO_PERMANENCIA'])

    dataset1['PERMANENCIA_PROLONGADA'] = dataset1['PERMANENCIA_PROLONGADA'] == 1
    dataset1['COTAS'] = dataset1['COTAS'] != 'A'
    dataset1['TRABALHO_DURANTE_GRAD'] = dataset1['TRABALHO_DURANTE_GRAD'] != 'A'
    dataset1['INTERCAMBIO'] = dataset1['INTERCAMBIO'] != 'A'
    dataset1['BOLSA_ESTUDANTIL'] = dataset1['BOLSA_ESTUDANTIL'] != 'A'
    dataset1['AUXILIO_ESTUDANTIL'] = dataset1['AUXILIO_ESTUDANTIL'] != 'A'
    dataset1['SOLTEIRO'] = dataset1['ESTADO_CIVIL'] == 'A'
    dataset1['PRINCIPAL_MOTIVACAO'] = dataset1['PRINCIPAL_MOTIVACAO'] == 'E'
    dataset1['ENSINO_MEDIO'] = dataset1['ENSINO_MEDIO'].map({
        'A': 'PUB',
        'B': 'PRI',
        'C': 'EXT',
        'D': 'PUB',
        'E': 'PRI',
        'F': 'EXT',
    })

    dataset1['ESCOLARIDADE'] = (dataset1['ESCOLARIDADE_PAI'] > 3) | (dataset1['ESCOLARIDADE_MAE'] > 3)

    dataset1['IDADE'] = dataset1['IDADE'].apply(map_age)

    return dataset1.drop(columns=['ESCOLARIDADE_PAI', 'ESCOLARIDADE_MAE', 'ESTADO_CIVIL'])


def main():
    dataset = mount_dataset()
    dataset = create_sequencing(dataset)

    dataset.to_csv('storage/dataset_bi.csv', index=False)
    print(dataset)

    dataset_apriori = categorize(dataset)
    dataset_apriori.to_csv('storage/dataset_apriori.csv', index=False)
    print(dataset_apriori)

    # function2(dataset).to_csv('result_knime.csv', index=False)


if __name__ == '__main__':
    main()
