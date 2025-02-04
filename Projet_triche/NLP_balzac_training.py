"""
@author = asyl_scy (Kara)

date = 04/02/2025

topic = "étude de cas d'une triche"

Ici, on va entrainer le transformer avec des phrases issu de "father goriot". 
"""
from transformers import  AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split


import re
import numpy as np
import tensorflow as tf

def list_sentences_txt (txt, ind, uni_transfo):
    """le but ici est de créer une liste comportant les phrases du texte
    txt= le texte
    ind= les indices qui vont permettre de couper le texte en sentences
    uni_tranfo = Booleen: si True, applique une transformation des sentences en unicodes
    """
    #liste qui contiendra les phrases
    list_stock_sentence=[]
    for i in range(len(ind)-1):
        #les +1 et -1 servent à avoir des phrases classiques avec des points à la fin.
        stock_transi=txt[ind[i]+1:ind[i+1]+1]
        stock_transi=stock_transi.replace("\r\n"," ")
        stock_transi.replace("  ", " ")
        stock_transi=stock_transi.lower()
        
        if uni_transfo:
            stock_transi=transform(stock_transi)

        
        list_stock_sentence.append(stock_transi)
    return list_stock_sentence


def transform(txt):#retourne un array ayant les équivalants unicode du txt en entrée
  return np.asarray([ord(c) for c in txt if ord(c) < 255], dtype=np.int32)#ord() permet de mettre l'unicode correspond à un élément



balzac_adress= "father_goriot_english.txt"
encodage_uni=False

with tf.io.gfile.GFile(balzac_adress, 'r') as f:
    #importation
    txt = f.read()

#j'ai repéré manuellement les endroits où il y a le txt de balzac qui m'intéresse
pre=(re.search('Mme. Vauquer ', txt)).start()
post=(re.search('ADDENDUM', txt)).end()
#en fait un sclicing juste pour récup ce qui est intéressant
balzac_txt=txt[pre:post]

#list compréhension pour récupérer les indices qui me permettront de faire des phrases (.,!,?,etc.)
when_point=[x for x in range (len(balzac_txt)) if x==len(balzac_txt)-1 or balzac_txt[x] in ["?", "!"]  or  balzac_txt[x] == "." and balzac_txt[x+1] not in [".", '"'] and balzac_txt[x-1] not in  [".","M"] or  balzac_txt[x] == '"']

#création sentences
list_stock_s=list_sentences_txt(balzac_txt, when_point,encodage_uni)
#supprime les phrases trop petites (>50) et trop grandes (<600)
list_stock_s_2=[x for x in list_stock_s if len(x) >50 and len(x) <600]

#longueur maximale
max_len=max([len(x) for x in list_stock_s_2])

#récupère les tokens du transformer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

#on récupère la max_len des phrases tokenisés
max_len_padded=max([len([y for y in tokenizer(x).attention_mask if y ==1]) for x in list_stock_s_2])#131
max_len=max([len(x) for x in list_stock_s_2])#597 (juste pour vérifier, ne sera jamais utilisé)
#attention mask= si 1, a un mot; si =0, pas de mot (pad)
train_ds, test_ds = train_test_split(list_stock_s_2, test_size=0.2, random_state=42)
#tokenize les phrases de dataset train et test
tokenized_train=[tokenizer(sentence, max_length=max_len_padded, padding="max_length") for sentence in train_ds]
tokenized_test=[tokenizer(sentence, max_length=max_len_padded, padding="max_length") for sentence in test_ds]


#creation dataset
#on crée la rubrique labels (on en a pas besoin mais il est obligatoire pour l'entrainement)

#train
for ts in tokenized_train:
    ts["labels"] = ts["input_ids"]
dataset_train = Dataset.from_dict({
    "input_ids": [ts["input_ids"] for ts in tokenized_train],
    "attention_mask": [ts["attention_mask"] for ts in tokenized_train],
     "labels": [ts["labels"] for ts in tokenized_train],
})

#test

for ts in tokenized_test:
    ts["labels"] = ts["input_ids"]
dataset_test = Dataset.from_dict({
    "input_ids": [ts["input_ids"] for ts in tokenized_test],
    "attention_mask": [ts["attention_mask"] for ts in tokenized_test],
    "labels": [ts["labels"] for ts in tokenized_test],
})



#charger modèle
model = AutoModelForCausalLM.from_pretrained("gpt2")

training_args = TrainingArguments(
    run_name="balzac_model_en",
    output_dir="./results",          # répertoire pour sauvegarder le modèle
    overwrite_output_dir=True,       # écraser les anciennes sauvegardes
    num_train_epochs=5,              # nombre d’époques
    per_device_train_batch_size=64,   # taille batch par GPU
    per_device_eval_batch_size=64,    # taille batch pour validation
    save_steps=10,                   # sauvegarde tous les 10 steps
    save_total_limit=2,              # limite de sauvegardes
    logging_dir="./logs",            # répertoire pour les logs
    learning_rate=5e-5,              # taux d’apprentissage
    warmup_steps=50,                 # étapes de "warmup" pour le scheduler
)

#compile le trainer
trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=dataset_train,
      eval_dataset=dataset_test,
  )

trainer.train()#entrainement
#sauvegarde model + tokenizer
model.save_pretrained("./balzac_model")
tokenizer.save_pretrained("./balzac_model_token")
