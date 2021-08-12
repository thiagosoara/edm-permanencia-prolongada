#import missingno as msno
#import pandas_profiling

from utils import *

import numpy as np
import pylab as plt
import seaborn as sns
import glob
import pandas as pd
pd.set_option('display.max_columns', 500)
import gc
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

warnings.filterwarnings("ignore")

def ks(data=None,target=None, prob=None):
    data['target0'] = 1 - data[target]
    data['bucket'] = pd.qcut(data[prob], 10)
    grouped = data.groupby('bucket', as_index = False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()[prob]
    kstable['max_prob'] = grouped.max()[prob]
    kstable['events']   = grouped.sum()[target]
    kstable['nonevents'] = grouped.sum()['target0']
    kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
    kstable['event_rate'] = (kstable.events / data[target].sum()).apply('{0:.2%}'.format)
    kstable['nonevent_rate'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
    kstable['cum_eventrate']=(kstable.events / data[target].sum()).cumsum()
    kstable['cum_noneventrate']=(kstable.nonevents / data['target0'].sum()).cumsum()
    kstable['KS'] = np.round(kstable['cum_eventrate']-kstable['cum_noneventrate'], 3) #* 100

    #Formating
    kstable['cum_eventrate']= kstable['cum_eventrate']#.apply('{0:.2%}'.format)
    kstable['cum_noneventrate']= kstable['cum_noneventrate']#.apply('{0:.2%}'.format)
    kstable.index = range(1,11)
    kstable.index.rename('Decile', inplace=True)
    #pd.set_option('display.max_columns', 9)
    #print(kstable)
    
    #Display KS
    from colorama import Fore
    print(Fore.RED + "KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))
    return(kstable)

def adjust_colum(df,cols_to_move,new_index):
        """
        This method re-arranges the columns in a dataframe to place the desired columns at the desired index.
        ex Usage: df = move_columns(df, ['Rev'], 2)   
        :param df:
        :param cols_to_move: The names of the columns to move. They must be a list
        :param new_index: The 0-based location to place the columns.
        :return: Return a dataframe with the columns re-arranged
        """    
        other = [c for c in df if c not in cols_to_move]
        start = other[0:new_index]
        end = other[new_index:]
        return df[start + cols_to_move + end].copy()


#input_dir = '../raw_data_enade/input/'

data_list = np.sort(np.array(glob.glob('storage/dataset_bi.csv')))#
#data_list

data = pd.read_csv(data_list[0],sep=',')

print(data.head())

print('Não alvo', round(data['PERMANENCIA_PROLONGADA'].value_counts()[0]/len(data) * 100,2), '% of the dataset')
print('Alvo', round(data['PERMANENCIA_PROLONGADA'].value_counts()[1]/len(data) * 100,2), '% of the dataset')

dict_estado_civil = {'A':'SOlTEIRO',
                     'B':'CASADO',
                     'C':'SEPARADO',
                     'D':'VIUVO',
                     'E':'OUTRO'}

data['ESTADO_CIVIL'] = data.ESTADO_CIVIL.map(dict_estado_civil)
#create categorical 
data=pd.get_dummies(data, columns=["ESTADO_CIVIL"])
data.drop('ESTADO_CIVIL_CASADO', axis=1, inplace=True)
data.drop('ESTADO_CIVIL_SEPARADO', axis=1, inplace=True)
#data.drop('ESTADO_CIVIL_VIUVO', axis=1, inplace=True)
data.drop('ESTADO_CIVIL_OUTRO', axis=1, inplace=True)

dict_raca = {'A':'BRANCO',
                     'B':'NEGRO',
                     'C':'PARDO',
                     'D':'AMARELO',
                     'E':'INDIGENA',
                     'F':'N_D'}

data['RACA'] = data.RACA.map(dict_raca)
#create categorical 
data=pd.get_dummies(data, columns=["RACA"])

dict_auxilio_estudantil = {'A':'SEM_AUXILIO',
                     'B':'ALGUM',
                     'C':'ALGUM',
                     'D':'ALGUM',
                     'E':'ALGUM',
                     'F':'ALGUM'}
data['AUXILIO_ESTUDANTIL'] = data.AUXILIO_ESTUDANTIL.map(dict_auxilio_estudantil)
#create categorical 
data=pd.get_dummies(data, columns=["AUXILIO_ESTUDANTIL"])
data.drop('AUXILIO_ESTUDANTIL_ALGUM', axis=1, inplace=True)

dict_intercambio = {'A':'SEM_INTERCAMBIO',
                     'B':'ALGUM',
                     'C':'ALGUM',
                     'D':'ALGUM',
                     'E':'ALGUM',
                     'F':'ALGUM'}
data['INTERCAMBIO'] = data.INTERCAMBIO.map(dict_intercambio)
#create categorical 
data=pd.get_dummies(data, columns=["INTERCAMBIO"])
data.drop('INTERCAMBIO_ALGUM', axis=1, inplace=True)

dict_trabalho_durante_grad = {'A':'ZERO_HORAS',
                     'B':'ATE_20_HORAS',
                     'C':'ATE_20_HORAS',
                     'D':'ENTRE_21_39_HORAS',
                     'E':'ACIMA_39_HORAS'
                     }
data['TRABALHO_DURANTE_GRAD'] = data.TRABALHO_DURANTE_GRAD.map(dict_trabalho_durante_grad)
#create categorical 
data=pd.get_dummies(data, columns=["TRABALHO_DURANTE_GRAD"])

dict_cotas = {'A':'NAO_COTISTA',
                     'B':'COTISTA',
                     'C':'COTISTA',
                     'D':'COTISTA',
                     'E':'COTISTA',
                     'F':'COTISTA'
                     }
data['COTAS'] = data.COTAS.map(dict_cotas)
#create categorical 
data=pd.get_dummies(data, columns=["COTAS"])
data.drop('COTAS_COTISTA', axis=1, inplace=True)

dict_ensino_medio = {'A':'PUBLICO',
                     'B':'PRIVADA',
                     'C':'PRIVADA',
                     'D':'PUBLICO',
                     'E':'PRIVADA',
                     'F':'PRIVADA'
                     }
data['ENSINO_MEDIO'] = data.ENSINO_MEDIO.map(dict_ensino_medio)
#create categorical 
data=pd.get_dummies(data, columns=["ENSINO_MEDIO"])
data.drop('ENSINO_MEDIO_PRIVADA', axis=1, inplace=True)

dict_bolsa_estudantil = {'A':'SEM_BOLSA',
                     'B':'BOLSA',
                     'C':'BOLSA',
                     'D':'BOLSA',
                     'E':'BOLSA',
                     'F':'BOLSA'
                     }
data['BOLSA_ESTUDANTIL'] = data.BOLSA_ESTUDANTIL.map(dict_bolsa_estudantil)
#create categorical 
data=pd.get_dummies(data, columns=["BOLSA_ESTUDANTIL"])
data.drop('BOLSA_ESTUDANTIL_BOLSA', axis=1, inplace=True)

dict_principal_motivacao = {'A':'OUTRO',
                     'B':'OUTRO',
                     'C':'OUTRO',
                     'D':'OUTRO',
                     'E':'VOCACAO',
                     'F':'OUTRO',
                     'G':'OUTRO',
                     'H':'OUTRO'       
                     }
data['PRINCIPAL_MOTIVACAO'] = data.PRINCIPAL_MOTIVACAO.map(dict_principal_motivacao)
#create categorical 
data=pd.get_dummies(data, columns=["PRINCIPAL_MOTIVACAO"])
data.drop('PRINCIPAL_MOTIVACAO_OUTRO', axis=1, inplace=True)

select_col = [c for c in data.columns.to_list() if c not in ['ANO_PROVA','ANO_ENTRADA','DURACAO_PERMANENCIA']]
final_data = data[select_col].copy()
print(final_data.head())

rename_dict = {
    'PERMANENCIA_PROLONGADA': 'target'
    }
final_data.rename(columns=rename_dict, inplace=True)
final_data = adjust_colum(final_data,['target'],final_data.shape[1]-1)

cols = ['IDADE','ESCOLARIDADE_PAI','ESCOLARIDADE_MAE','RENDA_FAMILIAR','SEXO_M','ESTADO_CIVIL_SOlTEIRO',
       'RACA_AMARELO','RACA_BRANCO','RACA_INDIGENA','RACA_NEGRO','RACA_N_D','RACA_PARDO',
        'AUXILIO_ESTUDANTIL_SEM_AUXILIO', 'INTERCAMBIO_SEM_INTERCAMBIO', 'TRABALHO_DURANTE_GRAD_ACIMA_39_HORAS',
        'TRABALHO_DURANTE_GRAD_ATE_20_HORAS','TRABALHO_DURANTE_GRAD_ENTRE_21_39_HORAS',
        'TRABALHO_DURANTE_GRAD_ZERO_HORAS', 'COTAS_NAO_COTISTA','ENSINO_MEDIO_PUBLICO','BOLSA_ESTUDANTIL_SEM_BOLSA',
        'PRINCIPAL_MOTIVACAO_VOCACAO']

X = final_data[cols]
y = final_data['target']
# Build a logreg and compute the feature importances
model = LogisticRegression()
# create the RFE model and select 8 attributes
rfe = RFE(model, 21)
rfe = rfe.fit(X, y)
# summarize the selection of the attributes
print('Selected features: %s' % list(X.columns[rfe.support_]))

from sklearn.feature_selection import RFECV
# Create the RFE object and compute a cross-validated score.
# The "accuracy" scoring is proportional to the number of correct classifications
rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=5, scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X.columns[rfecv.support_]))

# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.grid()
plt.show()

X = final_data[cols]
y = final_data['target']
# Build a logreg and compute the feature importances
model = LogisticRegression()
# create the RFE model and select 10 attributes
rfe = RFE(model, 10)
rfe = rfe.fit(X, y)
# summarize the selection of the attributes
print('Selected features: %s' % list(X.columns[rfe.support_]))

Selected_features = list(X.columns[rfe.support_])

X = final_data[Selected_features]

plt.subplots(figsize=(8, 5))
sns.heatmap(X.corr(), annot=True, cmap="RdYlGn")
plt.show()


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss

# create X (features) and y (response)
X = final_data[Selected_features]
y = final_data['target']

# use train/test split with different random_state values
# we can change the random_state values that changes the accuracy scores
# the scores change a lot, this is why testing scores is a high-variance estimate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# check classification scores of logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg. q(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
print('Train/Test split results:')
print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))

idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95

plt.figure()
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  
      "and a specificity of %.3f" % (1-fpr[idx]) + 
      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))