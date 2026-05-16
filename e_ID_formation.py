# -*- coding: utf-8 -*-

no_of_iterations=10
no_of_splits=10
no_of_trials=50

count=0
symb="`"
drug=[]
protein=[]
go=[]
pathway=[]
fp=open(r"BERT_embeddings.txt","r")
for line in fp:
    count+=1
    x=line.split(symb)
    if x[0].startswith("R-HSA"):
       pathway.append(x[0])
    else:
       protein.append(x[0])
fp.close()
print(count,len(protein),len(pathway))

node=protein+pathway


ppi=[]
fs=open(r"total_ppi.txt","r")
for line in fs:
    x=line.split(symb)
    if x[0] not in node or x[1] not in node:
       continue
    ppi.append(line.strip())
fs.close()
ppi=list(set(ppi))
print(len(ppi))



prot_path=[]
fs=open(r"total_protein_pathway_associations.txt","r")
for line in fs:
    x=line.split(symb)
    if x[0] not in node or x[1] not in node:
       continue
    prot_path.append(line.strip())
fs.close()
prot_path=list(set(prot_path))
print(len(prot_path))








import random


all_edges =prot_path
num_edges_to_select = int(0.8*len(prot_path))
selected_prot_path = random.sample(all_edges, num_edges_to_select)
print(f"Original number of edges: {len(all_edges)}")
print(f"Selected number of edges: {len(selected_prot_path)}")
print(f"Randomly selected edges: {selected_prot_path[0:5]}")
test_prot_path=set(all_edges).difference(set(selected_prot_path))
test_prot_path=list(test_prot_path)
print(len(test_prot_path))





count=0
fs=open(r"positive_interaction_data.txt","w")

for line in selected_prot_path:
    count+=1
    fs.write(line.strip())
    fs.write("\n")

for line in ppi:
    count+=1
    fs.write(line.strip())
    fs.write("\n")
fs.close()
print(count)

count=0

fs=open(r"test_data.txt","w")
for line in test_prot_path:
    count+=1
    fs.write(line.strip())
    fs.write("\n")
fs.close()
print(count)

already=[]
fs=open(r"positive_interaction_data.txt","r")
for line in fs:
    already.append(line.strip())
fs.close()
print(len(already),already[0])




lines=[]
fs=open(r"BERT_embeddings.txt","r")
#fs=open(r"reduced-reduced-consolidated-embeddings.txt","r")
for line in fs:
    lines.append(line.strip())
fs.close()
lines=list(set(lines))

count=0
fet_lst=[]
node_lst=[]
node_dic={}
count_node={}
index_node={}
symb="`"
#fs=open(r"drug-protein-go-pathway-consolidated-embeddings-bDVAE.txt","r")
for line in lines:
    count+=1
    x=line.split(symb)
    node_lst.append(x[0])
    count_node[count]=x[0]
    index_node[count-1]=x[0]
    fet_lst.append([float(l) for l in x[1:-1]])
    node_dic[x[0]]=[float(l) for l in x[1:-1]]
    if count<=10:
       print(line)
#fs.close()
print(count,len(node_lst),len(fet_lst),len(fet_lst[0]),len(node_dic.keys()),len(count_node.keys()))

count=0
symb="`"
edge_lst=[]
pos_edge=[]
fs=open(r"positive_interaction_data.txt","r")
#fs=open(r"positive_interaction_data.txt","r")
#fs=open(r"interaction_data/tot-pos-ID-b.txt","r")
for line in fs:
    count+=1
    if count%10000==0:
       print(count)
    x=line.split(symb)
    if x[0] not in node_lst or x[1] not in node_lst:
       continue
    if len(x[0].strip())==0 or len(x[1].strip())==0:
       continue
    edge_lst.append((node_lst.index(x[0].strip())+1,node_lst.index(x[1].strip())+1))
    pos_edge.append((node_lst.index(x[0].strip()),node_lst.index(x[1].strip())))
    if count<=10:
       print(line)
fs.close()


print(count)
print(len(edge_lst),len(pos_edge))



import torch
import torch.nn as nn
import torch.optim as optim
import random
import optuna

nodes = len(node_lst)
emb_len = len(fet_lst[0])
node_emb = torch.tensor(fet_lst)


class gen(nn.Module):
    def __init__(self, emb_len, nodes, hid_dim):
        super(gen, self).__init__()
        self.nodes = nodes
        self.net = nn.Sequential(
            nn.Linear(emb_len * 2, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 2)
        )

    def forward(self, z):
        out = self.net(z)
        temp=torch.sigmoid(out)
        temp=temp*self.nodes-1
        temp=torch.round(temp).long()
        ind=torch.clamp(temp,0,self.nodes-1)
        return ind

class discr(nn.Module):
    def __init__(self, emb_len, hid_dim):
        super(discr, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_len* 2, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


def gp(ds, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1)
    interp = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    d_interp = ds(interp)
    fake = torch.ones(d_interp.size())
    grad = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_a = grad.view(grad.size(0), -1)
    return ((grad_a.norm(2, dim=1) - 1) ** 2).mean()



def discr_loss(par,ds):
    pos_ind = random.sample(pos_edge, par['batch_size'])
    edge_ind= torch.tensor(pos_ind)
    u_ind = edge_ind[:, 0]
    v_ind = edge_ind[:, 1]
    pos_emb = torch.cat([node_emb[u_ind], node_emb[v_ind]], dim=1)
    d_out = ds(pos_emb)
    return torch.mean(d_out),pos_emb

def gen_loss(par,gn,ds):
    z = torch.randn(par['batch_size'], emb_len * 2)
    gen_ind = gn(z)
    u_ind = gen_ind[:, 0]
    v_ind = gen_ind[:, 1]
    gen_emb = torch.cat([node_emb[u_ind], node_emb[v_ind]], dim=1)
    d_fake = ds(gen_emb)
    return torch.mean(d_fake),gen_emb



def train_func(par):
    beta1=0.5
    lambda_gp=10
    gn = gen(emb_len, nodes, hid_dim=par['g_hidden_dim'])
    ds = discr(emb_len, hid_dim=par['d_hidden_dim'])
    g_opt = optim.Adam(gn.parameters(), lr=par['g_lr'], betas=(0.0, 0.9))
    d_opt = optim.Adam(ds.parameters(), lr=par['d_lr'], betas=(0.0, 0.9))
    for epoch in range(100):
        for i in range(5):
            d_opt.zero_grad()
            d_real,pos_emb=discr_loss(par, ds)
            d_fake,gen_neg=gen_loss(par,gn,ds)
            gp_a = gp(ds, pos_emb.data, gen_neg.data)
            d_loss = -d_real + d_fake + lambda_gp * gp_a
            d_loss.backward()
            d_opt.step()
        g_opt.zero_grad()
        temp,_=gen_loss(par,gn,ds)
        g_loss=-temp
        g_loss.backward()
        g_opt.step()

        #if (epoch + 1) % 10 == 0:
            #print(f"Epoch {epoch + 1}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    return gn, ds,g_loss.item()


def objective(trial):
    par= {
        'g_lr': trial.suggest_float('g_lr', 1e-5, 5e-4, log=True),
        'd_lr': trial.suggest_float('d_lr', 1e-5, 5e-4, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128]),
        'g_hidden_dim': trial.suggest_int('g_hidden_dim', 128, 256),
        'd_hidden_dim': trial.suggest_int('d_hidden_dim', 128, 256)
    }


    _,_,loss=train_func(par)
    return loss

def hard_neg(model, hard,best_par):
    model.eval()
    hard_neg = []
    batch=best_par['batch_size']
    limit=int(hard/batch)
    if hard%batch>0:
       limit+=1
    with torch.no_grad():
        for i in range(limit):
            if i*batch>hard:
               cur_batch = hard - (i-1)*batch
            else:
               cur_batch=batch
            temp = torch.randn(cur_batch, node_emb.shape[1] * 2)
            ind= model(temp)
            for a, b in ind:
                hard_neg.append([a.item(), b.item()])
    return hard_neg



if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=no_of_trials)
    best_par= study.best_params
    print("\nBest hyperparameters: ", study.best_params)
    fmodel, _,_ = train_func(best_par)
    hard = len(pos_edge)
    neg_samp = hard_neg(fmodel, hard,best_par)
    for i, edge in enumerate(neg_samp):
        if i >= 5:
            break
        print(edge)
        
        
df = study.trials_dataframe()
df.to_csv("optuna_trials_log_WGAN_GP.csv", index=False)        
        

protein=[]
pathway=[]

symb="`"
fs=open(r"BERT_embeddings.txt","r")
#fs=open(r"reduced-consolidated-drug-protein-go-pathway-ßDVAE-embeddings.txt","r")
#fs=open(r"reduced-consolidated-embeddings.txt","r")
for line in fs:
    x=line.split(symb)
    if x[0].startswith("R-HSA"):
       pathway.append(x[0])
    else:
       protein.append(x[0])
fs.close()
print(len(protein),len(pathway))

interactions=[]
symb="`"
fs=open(r"positive_interaction_data.txt","r")
for line in fs:
    interactions.append(line.strip())
    x=line.split(symb)
    s=x[1]+symb+x[0]+symb
    s=s.strip()
    interactions.append(s)
fs.close()
print(len(interactions))

combinations=[]
e=0
dgi=0
dpi=0
dpath=0
ddi=0
dpath=0
pgi=0
ppi=0
ppath=0
dispath=0
disprot=0
disdrug=0
for comb in neg_samp:
    e+=1
    if e%1000==0:
       print(e)
    [a,b]=comb
    s=index_node[a]+symb+index_node[b]+symb
    s=s.strip()
    y=s.split(symb)
    t=index_node[b]+symb+index_node[a]+symb
    t=t.strip()
    
    if s in interactions:
       continue
    if y[0] in pathway and y[1] in pathway:
       continue
    
    
    combinations.append(s)
    if e<=10:
       print(s)
print(e,"drug-drug",ddi,"drug-GO",dgi,"drug-pathway",dpath,"total",disdrug,ddi+dpi+dgi+dpath+pgi+ppath+ppi+disprot+dispath+disdrug)




symb="`"
e=0
#ft=open(r"hard_negative_samples_e.txt","w")
#ft=open(r"hard_negative_samples_a.txt","w")
ft=open(r"hard_negative_samples.txt","w")
for comb in combinations:
    e+=1
    #(a,b)=comb
    #s=index_node[a]+symb+index_node[b]+symb
    #s=s.strip()
    if e<=10:
       print(comb)
    ft.write(comb)
    ft.write("\n")
ft.close()
print(e)



count=0
removed=0
pos=0
neg=0
id_training=[]
fv=open(r"labels_total_interaction_data.txt","w")
ft=open(r"total_interaction_data.txt","w")
fs=open(r"positive_interaction_data.txt","r")
for line in fs:
    count+=1
    if count%10000==0:
       print(count,pos,neg)
    pos+=1
    ft.write(line.strip())
    ft.write("\n")
    fv.write("1")
    fv.write("\n")
    id_training.append(line.strip())
fs.close()



fs=open(r"hard_negative_samples.txt","r")
for line in fs:
    count+=1
    if count%10000==0:
       print(count,pos,neg)
    x=line.split("`")
    t=x[1]+"`"+x[0]+"`"
    t=t.strip()
    neg+=1
    ft.write(line.strip())
    ft.write("\n")
    fv.write("0")
    fv.write("\n")
    id_training.append(line.strip())
fs.close()
ft.close()
fv.close()
print(count,pos,neg)

