# -*- coding: utf-8 -*-
no_of_iterations=10
no_of_splits=10
no_of_trials=50


count=0
fet_lst=[]
node_lst=[]
node_dic={}
symb="`"
fs=open(r"hgt_embeddings_BERT.txt","r")
for line in fs:
    count+=1
    x=line.split(symb)
    node_lst.append(x[0])
    fet_lst.append([float(l) for l in x[1:-1]])
    node_dic[x[0]]=[float(l) for l in x[1:-1]]
    if count<=10:
       print(line)
fs.close()
print(count,len(node_lst),len(fet_lst),len(fet_lst[0]),len(node_dic.keys()))

count=0
symb="`"
edge_lst=[]
fs=open(r"total_interaction_data.txt","r")
for line in fs:
    count+=1
    x=line.split(symb)
    edge_lst.append([x[0],x[1]])
    if count<=10:
       print(line)
fs.close()


print(count)
print(len(edge_lst))

count=0
edge_label=[]
fs=open(r"labels_total_interaction_data.txt","r")


#fs=open(r"label-total-ID-a.txt","r")
for line in fs:
    count+=1
    edge_label.append(int(line.strip()))
    if count<=10:
       print(line)
fs.close()


print(count,len(edge_label))



import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np

def objective(trial):
        
        X=[]
        y=[]
        for line in edge_lst:
          [a,b]=line
          X.append(node_dic[a]+node_dic[b])
        y=edge_label
        X=np.array(X)
        y=np.array(y)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        
        param = {
            'objective': 'multi:softmax',
            'num_class': len(set(y)),
            'eval_metric': 'mlogloss',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
             'min_child_weight': trial.suggest_int('min_child_weight', 2, 20),
        }

        
        model = xgb.XGBClassifier(**param, random_state=42)
        model.fit(X_train, y_train)

        
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        return accuracy 


        

study = optuna.create_study(direction='maximize') 
study.optimize(objective, n_trials=no_of_trials) 
trial = study.best_trial


df = study.trials_dataframe()
df.to_csv("optuna_trials_log_XGBoost.csv", index=False)

param=trial.params
print(param)






import xgboost as xgb
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    make_scorer
)
import numpy as np

X=[]
y=[]
for line in edge_lst:
        [a,b]=line
        X.append(node_dic[a]+node_dic[b])
y=edge_label
X=np.array(X)
y=np.array(y)


xgb_model= xgb.XGBClassifier(**param, random_state=42)





n_splits = no_of_splits
n_repeats = no_of_iterations
repeated_kfold = RepeatedStratifiedKFold(
    n_splits=n_splits,
    n_repeats=n_repeats,
    random_state=42
)


accuracy=[]
precision=[]
recall=[]
f1_tot=[]
specificity=[]
mcc_tot=[]
auroc_tot=[]
auprc_tot=[]
counter = 0

for train_index, test_index in repeated_kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("X_train",len(X_train[0]),len(X_train))
    print("X_test",len(X_test[0]),len(X_test))
    print("iteration: ",int(counter/n_splits)+1,"fold: ",(counter%n_splits)+1)
    counter+=1
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]


    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred_prob)
    auprc = average_precision_score(y_test, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spec = tn / (tn + fp)

    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    f1_tot.append(f1)
    specificity.append(spec)
    mcc_tot.append(mcc)
    auroc_tot.append(auroc)
    auprc_tot.append(auprc)
    print("accuracy",round(acc,4),"precision",round(prec,4),"recall",round(rec,4),"f1-score",round(f1,4),"specificity",round(spec,4),"MCC",round(mcc,4), "AUROC", round(auroc,4), "AUPRC",round(auprc,4))


print("overall accuracy",round(np.mean(accuracy),4),round(np.std(accuracy),4))
print("overall precision",round(np.mean(precision),4),round(np.std(precision),4))
print("overall recall",round(np.mean(recall),4),round(np.std(recall),4))
print("overall f1-score",round(np.mean(f1_tot),4), round(np.std(f1_tot),4))
print("overall specificity",round(np.mean(specificity),4),round(np.std(specificity),4))
print("overall MCC",round(np.mean(mcc_tot),4),round(np.std(mcc_tot),4))
print("overall AUROC", round(np.mean(auroc_tot),4),round(np.std(auroc_tot),4))
print("overall AUPRC",round(np.mean(auprc_tot),4),round(np.std(auprc_tot),4))


fs=open(r"average_performance_metrics_XGBoost.txt","w")

s="overall accuracy`"+str(round(np.mean(accuracy),4))+"`"+str(round(np.std(accuracy),4))+"`"
print(s)
fs.write(s)
fs.write("\n")
s="overall precision`"+str(round(np.mean(precision),4))+"`"+str(round(np.std(precision),4))+"`"
print(s)
fs.write(s)
fs.write("\n")
s="overall recall`"+str(round(np.mean(recall),4))+"`"+str(round(np.std(recall),4))+"`"
print(s)
fs.write(s)
fs.write("\n")
s="overall f1-score`"+str(round(np.mean(f1_tot),4))+"`"+str(round(np.std(f1_tot),4))+"`"
print(s)
fs.write(s)
fs.write("\n")
s="overall specificity`"+str(round(np.mean(specificity),4))+"`"+str(round(np.std(specificity),4))+"`"
print(s)
fs.write(s)
fs.write("\n")
s="overall MCC`"+str(round(np.mean(mcc_tot),4))+"`"+str(round(np.std(mcc_tot),4))+"`"
print(s)
fs.write(s)
fs.write("\n")
s="overall AUROC`"+str(round(np.mean(auroc_tot),4))+"`"+str(round(np.std(auroc_tot),4))+"`"
print(s)
fs.write(s)
fs.write("\n")
s="overall AUPRC`"+str(round(np.mean(auprc_tot),4))+"`"+str(round(np.std(auprc_tot),4))
print(s)
fs.write(s)
fs.write("\n")
fs.close()





X_test=[]
y_test=[]
symb="`"
count=0
ft=open(r"protein_pathway_prediction.txt","r")
#ft=open(r"test_protein_go_1.txt","r")
pred_lst=[]
for line in ft:
    x=line.split(symb)
    X_test.append(node_dic[x[0]]+node_dic[x[1]])
    y_test.append(1)
    pred_lst.append(line.strip())
ft.close()
import numpy as np
X_test=np.array(X_test)
y_test=np.array(y_test)
print(len(X_test[0]))
print(len(pred_lst))



from sklearn.metrics import classification_report
predictions = xgb_model.predict(X_test)
probabilities = xgb_model.predict_proba(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy: ", round(accuracy,4))
print("Classification Report:\n", classification_report(y_test, predictions))




symb="`"
count=0
c=0
fs=open(r"result_protein_pathway_prediction_data_XGBoost.txt","w")
#fs=open(r"result_test_protein_go_1.txt","w")
for i in range(len(pred_lst)):
    count+=1
    s=""
    s+=pred_lst[i]
    s+=str(predictions[i])+symb
    s+=str(probabilities[i][0])+symb+str(probabilities[i][1])+symb
    s+=str(y_test[i])+symb
    s=s.strip()
    fs.write(s)
    fs.write("\n")
    if float(probabilities[i][1])>=0.9:
       c+=1
fs.close()
print(count,c,c/count)


X_test=[]
y_test=[]
symb="`"
count=0
ft=open(r"test_data.txt","r")
#ft=open(r"test_protein_go_1.txt","r")
pred_lst=[]
for line in ft:
    count+=1
    #if count>100:
       #break
    x=line.split(symb)
    X_test.append(node_dic[x[0]]+node_dic[x[1]])
    y_test.append(1)
    pred_lst.append(line.strip())
ft.close()
import numpy as np
X_test=np.array(X_test)
y_test=np.array(y_test)
print(len(X_test[0]))
print(len(pred_lst))



from sklearn.metrics import classification_report
predictions = xgb_model.predict(X_test)
probabilities = xgb_model.predict_proba(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy: ", round(accuracy,4))
print("Classification Report:\n", classification_report(y_test, predictions))




symb="`"
count=0
c=0
fs=open(r"result_test_data_XGBoost.txt","w")
#fs=open(r"result_test_protein_go_1.txt","w")
for i in range(len(pred_lst)):
    count+=1
    s=""
    s+=pred_lst[i]
    s+=str(predictions[i])+symb
    s+=str(probabilities[i][0])+symb+str(probabilities[i][1])+symb
    s+=str(y_test[i])+symb
    s=s.strip()
    fs.write(s)
    fs.write("\n")
    if float(probabilities[i][1])>=0.9:
       c+=1
fs.close()
print(count,c,c/count)



X_test=[]
y_test=[]
symb="`"
count=0
ft=open(r"test_data.txt","r")
pred_lst=[]
for line in ft:
    count+=1
    x=line.split(symb)
    X_test.append(node_dic[x[0]]+node_dic[x[1]])
    y_test.append(1)
    pred_lst.append(line.strip())
ft.close()
import numpy as np
X_test=np.array(X_test)
y_test=np.array(y_test)
print(len(X_test[0]))
print(len(pred_lst))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


booster = xgb_model.get_booster()
shap_values= booster.predict(xgb.DMatrix(X_test), pred_contribs=True)
contrib = shap_values[:, :-1]  
expected = shap_values[:, -1]      
mean_abs = np.abs(contrib).mean(axis=0)
percent = (mean_abs / mean_abs.sum()) * 100
fet_names = [str(i) for i in range(1,contrib.shape[1]+1)]
imp = pd.DataFrame({'feature': fet_names,'importance': mean_abs,'percentage': percent}).sort_values(by='percentage', ascending=False)
print("\nGlobal SHAP for protein_pathway")
print(imp.to_string(index=False), "\n")
with open('feature_importance_protein_pathway_XGBoost.txt', 'w') as f:
    f.write(imp.to_string(index=False))

plt.figure()
shap.summary_plot(contrib, X_test, feature_names=fet_names, show=False)
plt.title("Global SHAP summary for protein_pathway")
plt.savefig('protein_pathway_beeswarm_plot_XGBoost.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


