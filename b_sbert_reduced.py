# -*- coding: utf-8 -*-

no_of_trials=50



import numpy as np

import pandas as pd
import pickle
from sklearn.preprocessing import Normalizer

df = pd.read_csv(r"consolidated_embeddings_SBERT.txt", sep="`", header=None)
#df = pd.read_csv(r"gdrive/MyDrive/sandbox/consolidated_fet.txt", sep="`", header=None)
x = df.iloc[:, 1:-1].values


print(df.head().to_string())


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x= sc.fit_transform(x)


from sklearn.decomposition import PCA
pca = PCA(0.99)
#pca=PCA(n_components=64)


unpickled_df = pca.fit_transform(x)
explained_variance = pca.explained_variance_ratio_


print(explained_variance)

print(unpickled_df)
print(type(unpickled_df))
print(unpickled_df.shape)


cumulative_variance = np.sum(explained_variance)
print("Cumulative explained variance (features retained):", cumulative_variance)
# Example output: Cumulative explained variance (features retained): 0.58

mother=[]
x=df.values.tolist()
y=unpickled_df.tolist()
print(x[0])
print (len(x),len(x[0]))
print(y[0])
print(len(y),len(y[0]))
for i,j in zip(x,y):
    son=[]
    son.append(i[0])

    for k in j:
        son.append(k)
    mother.append(son)
print(mother[0])
print(len(mother),len(mother[0]))
dh=pd.DataFrame(mother)
print(dh)


symb="`"
count=0
#fp=open(r"gdrive/MyDrive/sandbox/reduced_consolidated_fet_PCA_1.txt","w")
fp=open(r"reduced_consolidated_embeddings_SBERT_PCA.txt","w")



for i in mother:
    count+=1
    st=""
    for j in i:
        st=st+str(j)+symb
    if count<=10:
       print(st.strip())
    fp.write(st.strip())
    fp.write("\n")
    #count+=1
fp.close()
print(count)



count=0
ids=[]
feat_lst=[]
symb="`"
#fs=open(r"gdrive/MyDrive/sandbox/reduced_consolidated_fet_PCA_1.txt","r")

fs=open(r"reduced_consolidated_embeddings_SBERT_PCA.txt","r")
for line in fs:
    count+=1
    if count<=10:
       print(line)
    x=line.split(symb)
    #if count<=10:
    ids.append(x[0])
    feat_lst.append(x[1:-1])
fs.close()
print(len(ids),len(feat_lst))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import optuna
from optuna.pruners import MedianPruner, PatientPruner
import numpy as np


input=len(feat_lst[0])
data=np.array(feat_lst).astype(np.float32)


scale = MinMaxScaler()
data = scale.fit_transform(data)
X_train, X_val = train_test_split(data, test_size=0.1, random_state=42)


train_data= TensorDataset(torch.from_numpy(X_train))
val_data= TensorDataset(torch.from_numpy(X_val))


class AE(nn.Module):
    def __init__(self, orig, inter, latent):
        super(AE, self).__init__()


        self.encoder = nn.Sequential(
            nn.Linear(orig, inter),
            nn.ReLU(),
            nn.Linear(inter, inter // 2),
            nn.ReLU(),
            nn.Linear(inter // 2, latent)
        )


        self.decoder = nn.Sequential(
            nn.Linear(latent, inter// 2),
            nn.ReLU(),
            nn.Linear(inter // 2, inter),
            nn.ReLU(),
            nn.Linear(inter, orig),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon= self.decoder(z)
        return recon, z

    def encode(self, x):
        return self.encoder(x)


def train_func(model, train_load, opt):
    model.train()
    tot_loss= 0
    for bx, in train_load:
        opt.zero_grad()
        recon, _ = model(bx)
        loss = F.mse_loss(recon, bx, reduction='mean')
        loss.backward()
        opt.step()
        tot_loss += loss.item()
    return tot_loss/ len(train_load)

def val_func(model, val_load):
    model.eval()
    tot_loss = 0
    with torch.no_grad():
        for bx, in val_load:
            recon, _ = model(bx)
            loss= F.mse_loss(recon, bx, reduction='mean')
            tot_loss += loss.item()
    return tot_loss / len(val_load)

def objective(trial):
    latent= trial.suggest_categorical('latent_dim', [32])
    inter= trial.suggest_int('intermediate_dim', 128, 512)
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch= trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    epoch = trial.suggest_int('epochs', 30, 100)

    train_load = DataLoader(train_data, batch_size=batch, shuffle=True)
    val_load= DataLoader(val_data, batch_size=batch)

    model = AE(orig=input, inter=inter, latent=latent)
    opt = optim.Adam(model.parameters(), lr=lr)

    for ep in range(epoch):
        trainl_loss = train_func(model, train_load, opt)
        val_loss = val_func(model, val_load)

        trial.report(val_loss, ep)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

if __name__ == "__main__":
    pr= PatientPruner(wrapped_pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5), patience=5)
    study = optuna.create_study(direction='minimize', pruner=pr)
    study.optimize(objective, n_trials=no_of_trials, show_progress_bar=True)

    best_par = study.best_params
    print("\nBest hyp", best_par)


    fmodel= AE(input, best_par['intermediate_dim'], best_par['latent_dim'])
    fopt = optim.Adam(fmodel.parameters(), lr=best_par['learning_rate'])

    full_load = DataLoader(TensorDataset(torch.from_numpy(data)),
                             batch_size=best_par['batch_size'], shuffle=True)

    for epoch in range(best_par['epochs']):
        train_func(fmodel, full_load, fopt)


    fmodel.eval()
    with torch.no_grad():
        full_data = torch.from_numpy(data)
        compr = fmodel.encode(full_data).numpy()
        print("\nFinal Compressed Shape:", compr.shape)

count=0
symb="`"
fet_lst=[]
#fs=open(r"gdrive/MyDrive/sandbox/reduced_consolidated_fet_AE_1.txt","w")
fs=open(r"reduced_consolidated_embeddings_SBERT_AE.txt","w")
for id,line in zip(ids,compr):
    count+=1
    #if count<=10:
       #print(len(line),line[:5],line[2])
    s=""
    s+=str(id)+symb
    t=""
    for i in range(len(line)):
        s+=str(line[i])+symb
        t+=str(line[i])+symb
    s=s.strip()
    t=t.strip()
    fet_lst.append(t)
    fs.write(s)
    fs.write("\n")
    if count<=10:
       print(s)
fet_lst=list(set(fet_lst))
print(count,len(fet_lst))
fs.close()

df = study.trials_dataframe()
df.to_csv("optuna_trials_log_SBERT_AE.csv", index=False)



fet_lst=[]
id_lst=[]
count=0
l=[]

fs=open(r"reduced_consolidated_embeddings_SBERT_AE.txt","r")
for line in fs:
    count+=1
    x=line.split("`")
    l.append(len(x))
    #if count<=10:
    id_lst.append(x[0])
    fet_lst.append(x[1:-1])
    if count<=10:
       print(line)
fs.close()
print(count)
import numpy as np
fet_lst=np.array(fet_lst)
fet_lst=fet_lst.astype(np.float32)
fet_lst=np.expand_dims(fet_lst, axis=0)
l=list(set(l))
emb_len=l[0]-2
print(len(id_lst),len(fet_lst[0]),l,emb_len)



import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna


input = torch.from_numpy(fet_lst).float()


class MHSA(nn.Module):
    def __init__(self, emb_len, heads=16, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.h_len = emb_len // heads
        self.q_proj, self.k_proj, self.v_proj = [nn.Linear(emb_len, emb_len) for _ in range(3)]
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(emb_len, emb_len)

    def forward(self, x):
        b, s, e = x.shape
        q = self.q_proj(x).view(b, s, self.heads, self.h_len).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.heads, self.h_len).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.heads, self.h_len).transpose(1, 2)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            att = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        return self.out(att.transpose(1, 2).reshape(b, s, e))

class Trans(nn.Module):
    def __init__(self, emb_len, heads, dropout=0.1):
        super().__init__()
        self.att = MHSA(emb_len, heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(emb_len, 4*emb_len), nn.GELU(), nn.Linear(4*emb_len, emb_len), nn.Dropout(dropout))
        self.norm_1, self.norm_2 = nn.LayerNorm(emb_len), nn.LayerNorm(emb_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.att(self.norm_1(x)))
        return x + self.dropout(self.ffn(self.norm_2(x)))

class MHSA_2(nn.Module):
    def __init__(self, emb_len, heads, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([Trans(emb_len, heads, dropout) for _ in range(2)])
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


def objective(trial):
    heads = trial.suggest_categorical("num_heads", [4, 8, 16])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    model = MHSA_2(emb_len, heads, dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    return train_func(model,opt)

def train_func(model,opt):
    model.train()
    opt.zero_grad()
    output = model(input)
    loss = F.mse_loss(output, input)
    loss.backward()
    opt.step()
    return loss.item()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=no_of_trials)
best_par = study.best_params
print(f"Best Params: {best_par}")
fmodel = MHSA_2(emb_len, best_par['num_heads'], best_par['dropout'])
fopt = torch.optim.AdamW(fmodel.parameters(), lr=best_par['lr'])
train_func(fmodel,fopt)
fmodel.eval()
with torch.no_grad():
    femb = fmodel(input).squeeze(0)
print("Final Embeddings: ",femb.shape)

l=[]
#final_embeddings=final_embeddings.cpu()

ft=open(r"reduced_consolidated_embeddings_SBERT_MHSA.txt","w")
for i in range(len(femb)):
#for i in range(10):
    temp=femb[i]
    #temp=temp.to(device).numpy()
    #if device=="cpu":
    temp=temp.numpy()
    temp=[str(float(str(t))) for t in temp]
    s=id_lst[i]+"`"+"`".join(temp)+"`"
    if i<=10:
       print(s)
    ft.write(s)
    ft.write("\n")
    fgh=s.split("`")
    l.append(len(fgh))
ft.close()
l=list(set(l))
print(l)




df = study.trials_dataframe()
df.to_csv("optuna_trials_log_SBERT_MHSA.csv", index=False)

import numpy as np

import pandas as pd
import pickle
from sklearn.preprocessing import Normalizer

df = pd.read_csv(r"reduced_consolidated_embeddings_SBERT_MHSA.txt", sep="`", header=None)
#df = pd.read_csv(r"gdrive/MyDrive/sandbox/consolidated_fet.txt", sep="`", header=None)
x = df.iloc[:, 1:-1].values


print(df.head().to_string())


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x= sc.fit_transform(x)


from sklearn.decomposition import PCA
pca = PCA(0.99)
#pca=PCA(n_components=64)


unpickled_df = pca.fit_transform(x)
explained_variance = pca.explained_variance_ratio_


print(explained_variance)

print(unpickled_df)
print(type(unpickled_df))
print(unpickled_df.shape)


cumulative_variance = np.sum(explained_variance)
print("Cumulative explained variance (features retained):", cumulative_variance)
# Example output: Cumulative explained variance (features retained): 0.58

mother=[]
x=df.values.tolist()
y=unpickled_df.tolist()
print(x[0])
print (len(x),len(x[0]))
print(y[0])
print(len(y),len(y[0]))
for i,j in zip(x,y):
    son=[]
    son.append(i[0])

    for k in j:
        son.append(k)
    mother.append(son)
print(mother[0])
print(len(mother),len(mother[0]))
dh=pd.DataFrame(mother)
print(dh)


symb="`"
count=0
#fp=open(r"gdrive/MyDrive/sandbox/reduced_consolidated_fet_PCA_1.txt","w")
fp=open(r"reduced_consolidated_embeddings_SBERT_MHSA_PCA.txt","w")



for i in mother:
    count+=1
    st=""
    for j in i:
        st=st+str(j)+symb
    if count<=10:
       print(st.strip())
    fp.write(st.strip())
    fp.write("\n")
    #count+=1
fp.close()
print(count)

count=0
ids=[]
feat_lst=[]
symb="`"
#fs=open(r"gdrive/MyDrive/sandbox/reduced_consolidated_fet_PCA_1.txt","r")

fs=open(r"reduced_consolidated_embeddings_SBERT_MHSA_PCA.txt","r")
for line in fs:
    count+=1
    if count<=10:
       print(line)
    x=line.split(symb)
    #if count<=10:
    ids.append(x[0])
    feat_lst.append(x[1:-1])
fs.close()
print(len(ids),len(feat_lst))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import optuna
from optuna.pruners import MedianPruner, PatientPruner
import numpy as np


input=len(feat_lst[0])
data=np.array(feat_lst).astype(np.float32)


scale = MinMaxScaler()
data = scale.fit_transform(data)
X_train, X_val = train_test_split(data, test_size=0.1, random_state=42)


train_data= TensorDataset(torch.from_numpy(X_train))
val_data= TensorDataset(torch.from_numpy(X_val))


class AE(nn.Module):
    def __init__(self, orig, inter, latent):
        super(AE, self).__init__()


        self.encoder = nn.Sequential(
            nn.Linear(orig, inter),
            nn.ReLU(),
            nn.Linear(inter, inter // 2),
            nn.ReLU(),
            nn.Linear(inter // 2, latent)
        )


        self.decoder = nn.Sequential(
            nn.Linear(latent, inter// 2),
            nn.ReLU(),
            nn.Linear(inter // 2, inter),
            nn.ReLU(),
            nn.Linear(inter, orig),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon= self.decoder(z)
        return recon, z

    def encode(self, x):
        return self.encoder(x)


def train_func(model, train_load, opt):
    model.train()
    tot_loss= 0
    for bx, in train_load:
        opt.zero_grad()
        recon, _ = model(bx)
        loss = F.mse_loss(recon, bx, reduction='mean')
        loss.backward()
        opt.step()
        tot_loss += loss.item()
    return tot_loss/ len(train_load)

def val_func(model, val_load):
    model.eval()
    tot_loss = 0
    with torch.no_grad():
        for bx, in val_load:
            recon, _ = model(bx)
            loss= F.mse_loss(recon, bx, reduction='mean')
            tot_loss += loss.item()
    return tot_loss / len(val_load)

def objective(trial):
    latent= trial.suggest_categorical('latent_dim', [32])
    inter= trial.suggest_int('intermediate_dim', 128, 512)
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch= trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    epoch = trial.suggest_int('epochs', 30, 100)

    train_load = DataLoader(train_data, batch_size=batch, shuffle=True)
    val_load= DataLoader(val_data, batch_size=batch)

    model = AE(orig=input, inter=inter, latent=latent)
    opt = optim.Adam(model.parameters(), lr=lr)

    for ep in range(epoch):
        trainl_loss = train_func(model, train_load, opt)
        val_loss = val_func(model, val_load)

        trial.report(val_loss, ep)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

if __name__ == "__main__":
    pr= PatientPruner(wrapped_pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5), patience=5)
    study = optuna.create_study(direction='minimize', pruner=pr)
    study.optimize(objective, n_trials=no_of_trials, show_progress_bar=True)

    best_par = study.best_params
    print("\nBest hyp", best_par)


    fmodel= AE(input, best_par['intermediate_dim'], best_par['latent_dim'])
    fopt = optim.Adam(fmodel.parameters(), lr=best_par['learning_rate'])

    full_load = DataLoader(TensorDataset(torch.from_numpy(data)),
                             batch_size=best_par['batch_size'], shuffle=True)

    for epoch in range(best_par['epochs']):
        train_func(fmodel, full_load, fopt)


    fmodel.eval()
    with torch.no_grad():
        full_data = torch.from_numpy(data)
        compr = fmodel.encode(full_data).numpy()
        print("\nFinal Compressed Shape:", compr.shape)

count=0
symb="`"
fet_lst=[]
#fs=open(r"gdrive/MyDrive/sandbox/reduced_consolidated_fet_AE_1.txt","w")
fs=open(r"reduced_consolidated_embeddings_SBERT_MHSA_AE.txt","w")
for id,line in zip(ids,compr):
    count+=1
    #if count<=10:
       #print(len(line),line[:5],line[2])
    s=""
    s+=str(id)+symb
    t=""
    for i in range(len(line)):
        s+=str(line[i])+symb
        t+=str(line[i])+symb
    s=s.strip()
    t=t.strip()
    fet_lst.append(t)
    fs.write(s)
    fs.write("\n")
    if count<=10:
       print(s)
fet_lst=list(set(fet_lst))
print(count,len(fet_lst))
fs.close()




df = study.trials_dataframe()
df.to_csv("optuna_trials_log_SBERT_MHSA_AE.csv", index=False)
