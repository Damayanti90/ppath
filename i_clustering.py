

count=0
pathway=[]
precomputed_embeddings=[]
symb="`"
ft=open(r"pathway_embeddings_hgt_BERT.txt","w")
fs=open(r"hgt_embeddings_BERT.txt","r")
for line in fs:
    count+=1
    x=line.split(symb)
    if x[0].startswith("R-HSA"):
       pathway.append(x[0])
       precomputed_embeddings.append([float(i) for i in x[1:-1]])
       ft.write(line.strip())
       ft.write("\n")
fs.close()
ft.close()
import numpy as np
precomputed_embeddings=np.array(precomputed_embeddings)
print(len(precomputed_embeddings))

count=0
c=0
import re
pathway_dic={}
fs=open(r"ReactomePathways.gmt",'r')
for line in fs:
    count+=1
    x=re.split("\t",line)
    if x[1].strip() in pathway:
       c+=1
       genes=[a.strip() for a in x[2:]]
       pathway_dic[x[1]]=genes
       if c<=10:
          print(count,len(x),len(genes))
          print(line)
fs.close()
print(count,c,len(pathway_dic.keys()))



def jaccard(p1_genes, p2_genes): 
    inter = set(p1_genes).intersection(set(p2_genes)) 
    union = set(p1_genes).union(set(p2_genes))  
    if len(union) == 0:        
        return 0.0
    else:
        jacc = len(inter) / len(union)
        return jacc



import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
import networkx as nx

def clustering(go_terms, similarity_matrix, thresh= 0.5):   
    dist = 1 - similarity_matrix
    np.fill_diagonal(dist, 0)
    if np.any(dist < 0):
        distance_matrix[dist < 0] = 0   
    cut = 1 - thresh   
    clus = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average', distance_threshold=cut)   
    clus.fit(dist) 
    return clus.labels_
    


import pandas as pd

def sim_mat(path_lst):
    sim_matrix = pd.DataFrame(index=path_lst, columns=path_lst, dtype=float)   
    for term1 in path_lst:
        for term2 in path_lst:            
            s = jaccard(pathway_dic[term1], pathway_dic[term2])
            sim_matrix.loc[term1, term2] = s    
    return sim_matrix

import numpy as np

import pandas as pd
import pickle
from sklearn.preprocessing import Normalizer

df = pd.read_csv(r"pathway_embeddings_hgt_BERT.txt", sep="`", header=None)
#df = pd.read_csv(r"consolidated_fet.txt", sep="`", header=None)
x = df.iloc[:, 1:-1].values


print(df.head().to_string())


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x= sc.fit_transform(x)


from sklearn.decomposition import PCA
#pca = PCA(0.99)
pca=PCA(n_components=2)


unpickled_df = pca.fit_transform(x)
explained_variance = pca.explained_variance_ratio_


print(explained_variance)

print(unpickled_df)
print(type(unpickled_df))
print(unpickled_df.shape)


cumulative_variance = np.sum(explained_variance)
print("cumulative explained variance:", cumulative_variance)


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
#fp=open(r"reduced_consolidated_fet_PCA_1.txt","w")
fp=open(r"2D_pathway_embeddings.txt","w")



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
print("entries in 2D_pathway_embeddings file",count)

go_coords={}
fp=open(r"2D_pathway_embeddings.txt","r")
for line in fp:
    x=line.split("`")
    go_coords[x[0]]=[round(float(x[1]),4),round(float(x[2]),4)]
fp.close()
print("keys in go_coords",len(go_coords.keys()))





def inertia(X, labels):
    ulabels = np.unique(labels)
    inert = 0
    for label in ulabels:
        if label == -1:
            continue
        clus_pt = X[labels == label]
        if clus_pt.size > 0:
            centroid = np.mean(clus_pt, axis=0)
            inert += np.sum((clus_pt- centroid)**2)
    return inert



go_terms=[aa for aa in pathway_dic.keys()]
print("starting similarity calculation")
data=sim_mat(go_terms)
#print(data)
a=data.to_numpy()

import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator

ft=open(r"threshold_stats.txt","w")
thresholds = np.linspace(0.1, 0.5, 50)
inertias = []
num_clusters_list = []

t=""
for similarity_threshold in thresholds:
        s=""
        print("checking for threshold=",similarity_threshold)
        labels = clustering(go_terms, a, thresh=similarity_threshold)
        n_clusters = len(np.unique(labels))
        num_clusters_list.append(n_clusters)
        #go_terms=np.array(go_terms)
        temp=[]
        for g in go_terms:
            temp.append(go_coords[g])
        temp=np.array(temp)
        inert= inertia(temp, labels)
        s+="threshold="+str(similarity_threshold)+",WCSS="+str(inertia)+", #clusters="+str(n_clusters)+symb
        inertias.append(inert)
        s=s.strip()
        ft.write(s)
        ft.write("\n")
        print(s)
kn = KneeLocator(thresholds, inertias, curve='convex', direction='decreasing', S=1.0)
opt = kn.elbow


if opt is not None:
       
       idx = list(thresholds).index(opt)
       opt_clus = num_clusters_list[idx]

       print("opt dist:", opt)
       print("opt clus:", opt_clus)
       t+="threshold="+str(opt)+", WCSS="+str(inertias[idx])+",#clusters="+str(opt_clus)+symb
else:
       print("elbow not found")
       t+="threshold=NA,WCSS=NA,#clusters=NA`"
t=t.strip()
print(t)
ft.close()

ft=open(r"pathway_cluster_labels.txt","w")
labels = clustering(go_terms, a, thresh=opt)
for i in range(len(go_terms)):
    print(go_terms[i],labels[i])
    s=str(go_terms[i])+"`"+str(labels[i])+"`"
    ft.write(s)
    ft.write("\n")
    print(s)
ft.close()







train=[]
fs=open("total_interaction_data.txt","r")
for line in fs:
    train.append(line.strip())
fs.close()

test=[]
fs=open("test_data.txt","r")
for line in fs:
    test.append(line.strip())
fs.close()




existent=train+test
print(len(existent))









drug_go_dic={}
drugs_covered=[]
threshold=0.7
count=0
c=0
fs=open(r"result_protein_pathway_prediction_data_lightGBM.txt","r")
#fs=open(r"result_drug-go-prediction-data_modified.txt","r")
for line in fs:
    count+=1
    x=line.split("`")
    s=x[0]+"`"+x[1]+"`"
    s=s.strip()
    drug_go_dic[s]=[float(x[4].strip())]
    if round(float(x[4].strip()),4)>=threshold:
       c+=1



fs.close()
#print(c,c/len(drug_go_dic.keys()))
#print(len(drug_go_dic.keys()))



count=0
c=0
fs=open(r"result_protein_pathway_prediction_data_XGBoost.txt","r")
for line in fs:
    count+=1
    x=line.split("`")
    s=x[0]+"`"+x[1]+"`"
    s=s.strip()
    drugs_covered.append(x[0])
    #if s in drug_go:
    temp=drug_go_dic[s]
    temp.append(float(x[4].strip()))
    drug_go_dic[s]=temp
    if round(float(x[4].strip()),4)>=threshold:
       c+=1
fs.close()
#print(c,c/len(drug_go_dic.keys()))
#print(len(drug_go_dic.keys()))





print(len(drug_go_dic.keys()))
blind=set(drug_go_dic.keys()).difference(set(existent))
blind=list(blind)
print(len(blind))










count=0
lit=0
fa=open(r"result_prediction_data_combined.txt","w")
count=0
c=0
d=0
both=0
#for line in test_data_result.keys():
for line in blind:
    count+=1
    [a,b]=drug_go_dic[line]
    if round(a,4)>=threshold:
       c+=1
    if round(b,4)>=threshold:
       d+=1
    if round(a,4)>=threshold and round(b,4)>=threshold:
    #if round((a+b+c+d)/4,4)>=threshold:
       both+=1
       s=line.strip()+str(a)+"`"+str(b)+"`"
       s=s.strip()
       x=line.split("`")
       fa.write(s)
       fa.write("\n")
       #if c<=10:
          #print(s)

print(count,"XGBoost",d,d/count,"lightGBM", c, c/count, "both", both, both/count)
fa.close()


count=0
symb="`"
pred_data=[]
fs=open(r"result_prediction_data_combined.txt","r")
for line in fs:
    count+=1
    x=line.split(symb)
    s=x[0]+symb+x[1]+symb
    s=s.strip()
    pred_data.append(s)
    if count<=10:
       print(line)
fs.close()
print(len(pred_data))

count=0
symb="`"
existing_data=[]
fs=open(r"positive_interaction_data.txt","r")
for line in fs:
    count+=1
    x=line.split(symb)
    if x[1].startswith("R-HSA")==0:
       continue
    s=x[0]+symb+x[1]+symb
    s=s.strip()
    existing_data.append(s)
    if count<=10:
       print(line)
fs.close()
print(len(existing_data))

pred_data=set(pred_data). difference (set(existing_data))
print(len(pred_data))

dark_kinase=[]

count=0
fs=open(r"result_protein_pathway_prediction_data_XGBoost.txt","r")
for line in fs:
    count+=1
    x=line.split(symb)
    dark_kinase.append(x[0])
fs.close()
print(count)
dark_kinase=list(set(dark_kinase))
print(len(dark_kinase))

pred_data=list(pred_data)

tot_data=existing_data+pred_data
tot_data=list(set(tot_data))
print(len(tot_data))

prot_dic={}
for line in tot_data:
    x=line.split(symb)
    if x[0] not in prot_dic.keys():
       prot_dic[x[0]]=[x[1]]
    else:
       temp=prot_dic[x[0]]
       temp.append(x[1])
       prot_dic[x[0]]=temp
print(len(prot_dic.keys()))

pathway_cluster={}
ft=open(r"pathway_cluster_labels.txt","r")
for line in ft:
    x=line.split(symb)
    pathway_cluster[x[0]]=int(x[1])
ft.close()
print(len(pathway_cluster.keys()))

prot_clus={}
for prot in prot_dic.keys():
    pathway_lst=prot_dic[prot]
    clus=[]
    for path in pathway_lst:
        if path in pathway_cluster.keys():
           clus.append(pathway_cluster[path])
    clus=list(set(clus))
    prot_clus[prot]=clus
print(len(prot_clus.keys()))

dark_sim={}
for prot in dark_kinase:
    dark_sim[prot]=[prot]
print(len(dark_sim.keys()))

count=0
func_sim_dark={}
fs=open(r"functionally_similar_clusters.txt","w")
for prot_a in dark_sim.keys():
    count+=1
    a=prot_clus[prot_a]
    for prot_b in prot_clus.keys():
        if prot_a==prot_b:
           continue
        b=prot_clus[prot_b]
        if len(set(a).difference(set(b)))==0 and len(set(a).intersection(set(b)))==len(set(a)):
           temp=dark_sim[prot_a]
           temp.append(prot_b)
           dark_sim[prot_a]=temp
    #if count<=10:
    print(prot_a,len(dark_sim[prot_a]),dark_sim[prot_a])
    if len(dark_sim[prot_a])>1:
       func_sim_dark[prot_a]=dark_sim[prot_a]
       s=""
       for prot in dark_sim[prot_a]:
           s+=prot+symb
       s=s.strip()
       print(s)
       fs.write(s)
       fs.write("\n")
print(count)
fs.close()
print(len(func_sim_dark.keys()))

symb="`"
ft=open(r"hdkp_functional_similarity.txt","w")
s="HDKP`functionally similar proteins`"
ft.write(s)
ft.write("\n")
fs=open(r"functionally_similar_clusters.txt","r")
for line in fs:
    x=line.split(symb)
    t=""
    for pro in x[1:-1]:
        t+=pro+","
    t=t.strip()
    t=t[:-1]
    t+="`"
    wh=x[0]+"`"+t
    print(wh)
    ft.write(wh)
    ft.write("\n")
ft.close()


