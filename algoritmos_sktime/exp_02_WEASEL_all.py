import time
import datetime

start = time.time()

quarter = 'local'
pasta = 'Local'
book_name = 'weasel_' + quarter
pos_label = 1
#average = 'weighted'

from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(10,random_state=1,shuffle=True)

data_path = '/home/temp/wzalewski/Patricia/DadosKepler/shallue_all_'+ quarter + '.csv'
path_result = '/home/temp/wzalewski/Patricia/Resultados_Todos/' + pasta + '/' + book_name + '.xlsx'
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sktime.utils.data_processing import from_2d_array_to_nested

from sktime.classification.dictionary_based import IndividualTDE
from sktime.classification.dictionary_based import MUSE
from sktime.classification.dictionary_based import IndividualBOSS
from sktime.classification.dictionary_based import WEASEL

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,precision_score, average_precision_score
from sklearn.metrics import f1_score,recall_score,roc_auc_score,balanced_accuracy_score

data = pd.read_csv(data_path, sep = ",") 

#definição input e label no formato tabular exigido pelo scikit-learn
data_input = data.copy()
label = data_input.pop(data_input.columns[len(data_input.columns)-1])

X = data_input.values
y = label.values

#normalização
norm_data = data_input.copy()
norm_data = norm_data.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1)
X_norm = norm_data.values

#label binário
lb = LabelBinarizer()
y = lb.fit_transform(label)
y = y.reshape(-1)[:]

#será necessário converter os dados de tabular para nested para aplicar algoritmos da sktime
X_nested = from_2d_array_to_nested(X_norm)[:]

#definição dos modelos e parametros
model_params = {
    'WEASEL' : {
        'model': WEASEL(),
        'params': {
            'window_inc': [2,3,4,5,6],
            'random_state': [1]
        }
    }
}

#definição das métricas e parametros
scoring = {'acc': 'accuracy',
           'prec': make_scorer(precision_score,pos_label=pos_label),
           'avg_prec': make_scorer(average_precision_score,pos_label=pos_label),
           'recall': make_scorer(recall_score,pos_label=pos_label),
           'f1': make_scorer(f1_score,pos_label=pos_label),
           'bal_acc': 'balanced_accuracy'
            }

#execução dos modelos com randomizedsearchcv
scores = []
for model_name, mp in model_params.items():
    clf =  RandomizedSearchCV(mp['model'], mp['params'], cv=cv, scoring=scoring, return_train_score=True, refit=False, n_iter=3)
    clf.fit(X_nested, y)

    model_dic = {'model': model_name}
    score = clf.cv_results_
    metrics = {**model_dic, **score}
    mtrc = pd.DataFrame(metrics)
    scores.append(mtrc)

#resultados em dataframe
lista = [scores[0]] 
resultados_completos = pd.concat(lista, ignore_index=True)

#dataframe com seleção de médias e desvio padrões
resultados = pd.DataFrame()
resultados[['model','mean_test_acc','std_test_acc','mean_test_prec','std_test_prec',
            'mean_test_avg_prec','std_test_avg_prec','mean_test_recall','std_test_recall',
            'mean_test_f1','std_test_f1','mean_test_bal_acc',
            'std_test_bal_acc']] = resultados_completos[['model','mean_test_acc',
                                        'std_test_acc','mean_test_prec','std_test_prec','mean_test_avg_prec',
                                        'std_test_avg_prec','mean_test_recall','std_test_recall','mean_test_f1',
                                        'std_test_f1','mean_test_bal_acc','std_test_bal_acc']]

#salva dataframes no excel
resultados.to_excel(path_result, sheet_name=quarter)  

book = load_workbook(path_result)
writer = pd.ExcelWriter(path_result, engine = 'openpyxl')
writer.book = book

resultados_completos.to_excel(writer, sheet_name=quarter + '_completo')
writer.save()
writer.close()

end = time.time()
print(str(datetime.timedelta(seconds=end - start)))