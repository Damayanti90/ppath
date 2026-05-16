

count=0
import numpy as np
import torch
no_of_trials=50
ids_protein=[]
feat_lst_protein=[]
ids_path=[]
feat_lst_path=[]
symb="`"
#fs=open(r"gdrive/MyDrive/pp_sandbox/merged_embeddings_3.txt","r")
fs=open(r"BERT_embeddings.txt","r")
for line in fs:
    count+=1
    if count<=10:
       print(line)
    x=line.split(symb)
    if x[0].startswith("R-HSA"):
       ids_path.append(x[0])
       feat_lst_path.append(x[1:-1])
    else:
      ids_protein.append(x[0])
      feat_lst_protein.append(x[1:-1])
fs.close()

data=np.array(feat_lst_protein).astype(np.float32)
feat_tensor_protein=torch.from_numpy(data)

data=np.array(feat_lst_path).astype(np.float32)
feat_tensor_path=torch.from_numpy(data)
#print(len(feat_tensor_drug),len(feat_tensor_drug[0]))
print(len(feat_tensor_protein),len(feat_tensor_protein[0]))
#print(len(feat_tensor_go),len(feat_tensor_go[0]))
print(len(feat_tensor_path),len(feat_tensor_path[0]))

protein_protein=[]
protein_pathway=[]
symb="`"
fs=open(r"positive_interaction_data.txt","r")
for line in fs:
    x=line.split(symb)
    t=x[1]+symb+x[0]+symb
    t=t.strip()

    if x[0] in ids_protein and x[1] in ids_protein:
       protein_protein.append(line.strip())

    elif x[0] in ids_protein and x[1] in ids_path:
       protein_pathway.append(line.strip())


    elif x[1] in ids_protein and x[0] in ids_path:
       protein_pathway.append(t.strip())
fs.close()

print(len(protein_protein),len(protein_pathway))


coord_a=[]
coord_b=[]
for line in protein_protein:
    x=line.split(symb)
    coord_a.append(ids_protein.index(x[0]))
    coord_b.append(ids_protein.index(x[1]))
protein_protein_tensor=[coord_a,coord_b]
protein_protein_tensor=np.array(protein_protein_tensor)
protein_protein_tensor=torch.from_numpy(protein_protein_tensor)
print(len(protein_protein_tensor),len(protein_protein_tensor[0]))


coord_a=[]
coord_b=[]
for line in protein_pathway:
    x=line.split(symb)
    coord_a.append(ids_protein.index(x[0]))
    coord_b.append(ids_path.index(x[1]))
protein_pathway_tensor=[coord_a,coord_b]
protein_pathway_tensor=np.array(protein_pathway_tensor)
protein_pathway_tensor=torch.from_numpy(protein_pathway_tensor)
print(len(protein_pathway_tensor),len(protein_pathway_tensor[0]))

neg_protein_protein=[]
neg_protein_pathway=[]
symb="`"
fs=open(r"hard_negative_samples.txt","r")
for line in fs:
    x=line.split(symb)
    t=x[1]+symb+x[0]+symb
    t=t.strip()

    if x[0] in ids_protein and x[1] in ids_protein:
       neg_protein_protein.append(line.strip())

    elif x[0] in ids_protein and x[1] in ids_path:
       neg_protein_pathway.append(line.strip())


    elif x[1] in ids_protein and x[0] in ids_path:
       neg_protein_pathway.append(t.strip())
fs.close()
print(len(neg_protein_protein),len(neg_protein_pathway))


coord_a=[]
coord_b=[]
for line in neg_protein_protein:
    x=line.split(symb)
    coord_a.append(ids_protein.index(x[0]))
    coord_b.append(ids_protein.index(x[1]))
neg_protein_protein_tensor=[coord_a,coord_b]
neg_protein_protein_tensor=np.array(neg_protein_protein_tensor)
neg_protein_protein_tensor=torch.from_numpy(neg_protein_protein_tensor)
print(len(neg_protein_protein_tensor),len(neg_protein_protein_tensor[0]))


coord_a=[]
coord_b=[]
for line in neg_protein_pathway:
    x=line.split(symb)
    coord_a.append(ids_protein.index(x[0]))
    coord_b.append(ids_path.index(x[1]))
neg_protein_pathway_tensor=[coord_a,coord_b]
neg_protein_pathway_tensor=np.array(neg_protein_pathway_tensor)
neg_protein_pathway_tensor=torch.from_numpy(neg_protein_pathway_tensor)
print(len(neg_protein_pathway_tensor),len(neg_protein_pathway_tensor[0]))

from torch_geometric.data import HeteroData
data = HeteroData()

data['protein'].x = feat_tensor_protein
data['pathway'].x = feat_tensor_path
data['protein', 'targets', 'protein'].edge_index = protein_protein_tensor
data['protein', 'impacts', 'pathway'].edge_index = protein_pathway_tensor
data['protein', 'no_interaction', 'protein'].edge_index = neg_protein_protein_tensor
data['protein', 'no_interaction', 'pathway'].edge_index = neg_protein_pathway_tensor

import torch_geometric.transforms as T
transform = T.ToUndirected()
data_undirected = transform(data)
data=data_undirected

print(data)
print(f"Number of nodes: ",data.num_nodes)
print(f"Number of edges: ",data.num_edges)
print(f"Has isolated nodes: ",data.has_isolated_nodes())
print(f"Has self loops: ",data.has_self_loops())
print(f"Is undirected: ",data.is_undirected())

import torch
import torch.nn.functional as F
import optuna
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.transforms import RandomLinkSplit
from optuna.pruners import MedianPruner, PatientPruner
latent_dim=len(feat_tensor_path[0])
class HGT(torch.nn.Module):
    def __init__(self, metadata, hidden_channel, out_channel, tot_heads, tot_layers):
        super().__init__()
        self.pre= torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.pre[node_type] = Linear(-1, hidden_channel)

        self.block = torch.nn.ModuleList()
        self.norm=torch.nn.ModuleList()
        for _ in range(tot_layers):
            self.block.append(HGTConv(hidden_channel, hidden_channel, metadata, heads=tot_heads))
            self.norm.append(torch.nn.LayerNorm(hidden_channel))
        self.post = Linear(hidden_channel, out_channel)

    def forward(self, x_dict, edge_index_dict):
        temp = {}
        for node_type, x in x_dict.items():
             a= self.pre[node_type]
             b = a(x)
             temp[node_type] = b.relu()
        x_dict=temp
        for i, bl in enumerate(self.block):
            x_dict = bl(x_dict, edge_index_dict)
            temp_1={}
            for node_type,x in x_dict.items():
                tt=self.norm[i]
                temp_1[node_type]=tt(x).relu()
            x_dict=temp_1
        temp_2={}
        for node_type,x in x_dict.items():
            temp_2[node_type]=self.post(x)
        return temp_2


def get_link_loss(out, data_split):
        loss = 0
        for edge_type in [("protein", "targets", "protein"), ("protein", "impacts", "pathway")]:
            if edge_type not in data_split.edge_label_index_dict: continue
            src, rel, dst = edge_type
            z_u, z_v = out[src], out[dst]
            edge_index = data_split[edge_type].edge_label_index
            label = data_split[edge_type].edge_label
            scores = (z_u[edge_index[0]] * z_v[edge_index[1]]).sum(dim=-1)
            loss += F.binary_cross_entropy_with_logits(scores, label)
        return loss

def objective(trial, train_data,val_data):

    h_channels = trial.suggest_categorical("hidden_channels", [16, 32, 64, 128])
    n_heads = trial.suggest_categorical("num_heads", [4, 8])
    n_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    #latent_dim = trial.suggest_categorical("latent_dim", [128, 256])
    #epochs = trial.suggest_int("epochs", 20, 100)
    epochs=100
    latent_dim=len(feat_tensor_path[0])
    model = HGT(data.metadata(), h_channels, latent_dim, n_heads, n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return train_func(model,optimizer,epochs)

def train_func(model, optimizer,epochs):
    for epoch in range(epochs):

        model.train()
        optimizer.zero_grad()
        out = model(train_data.x_dict, train_data.edge_index_dict)
        train_loss = get_link_loss(out, train_data)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(val_data.x_dict, val_data.edge_index_dict)
            val_loss = get_link_loss(val_out, val_data)

        

    return val_loss.item()


transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.2,
    neg_sampling_ratio=1.0,
    add_negative_train_samples=True,
    edge_types=[("protein", "targets", "protein"), ("protein", "impacts", "pathway")],
    rev_edge_types=[None, ("pathway", "rev_impacts", "protein")]
)

train_data, val_data, _ = transform(data)


pruner = PatientPruner(MedianPruner(n_startup_trials=5), patience=10)
study = optuna.create_study(direction="minimize", pruner=pruner)
study.optimize(lambda trial: objective(trial, train_data, val_data), n_trials=no_of_trials)


print(f"Best Params: {study.best_params}")
best_par = study.best_params
fmodel=HGT(data.metadata(), best_par['hidden_channels'], latent_dim, best_par['num_heads'], best_par['num_layers'])
fmodel.eval()
with torch.no_grad():
    femb = fmodel(data.x_dict, data.edge_index_dict)
print(f"\nOptimization Complete. Best Hyperparameters: {best_par}")





df = study.trials_dataframe()
df.to_csv("optuna_trials_log_HGT+BERT.csv", index=False)

count=0
count_pro=0
symb="`"
fet_lst=[]
fs=open(r"hgt_embeddings_BERT.txt","w")
for id,line in zip(ids_protein,femb['protein']):
    count+=1
    count_pro+=1
    #if count<=10:
       #print(len(line),line[:5],line[2])
    s=""
    s+=str(id)+symb
    t=""
    for i in range(len(line)):
        s+=str(float(line[i]))+symb
        t+=str(float(line[i]))+symb
    s=s.strip()
    t=t.strip()
    fet_lst.append(t)
    fs.write(s)
    fs.write("\n")
    if count_pro<=10:
       print(s)
fet_lst=list(set(fet_lst))





count_path=0
for id,line in zip(ids_path,femb['pathway']):
    count+=1
    count_path+=1
    #if count<=10:
       #print(len(line),line[:5],line[2])
    s=""
    s+=str(id)+symb
    t=""
    for i in range(len(line)):
        s+=str(float(line[i]))+symb
        t+=str(float(line[i]))+symb
    s=s.strip()
    t=t.strip()
    fet_lst.append(t)
    fs.write(s)
    fs.write("\n")
    if count_path<=10:
       print(s)
fet_lst=list(set(fet_lst))
print(count,len(fet_lst),len(fet_lst[0]))
fs.close()
