# -*- coding: utf-8 -*-




count=0
ft=open(r"BERT_embeddings.txt","w")
fs=open(r"reduced_consolidated_embeddings_esm2_MHSA_AE.txt","r")

for line in fs:
    count+=1
    ft.write(line.strip())
    ft.write("\n")
fs.close()


fs=open(r"reduced_consolidated_embeddings_SBERT_MHSA_AE.txt","r")
for line in fs:
    count+=1
    ft.write(line.strip())
    ft.write("\n")
fs.close()
ft.close()
print(count)


