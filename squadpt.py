# Algum modelo para falar o contexto da imagem e ser passado como contexto do Q&A.

###################### FROM SCRATCH ######################
# Database obtained from https://forum.ailab.unb.br/t/datasets-em-portugues/251
import json
import pandas as pd

# Abrindo o json #
file_path = "/home/toni/Documents/QuestionAnswering/QuestionAnswering/squad-pt/squad-train-v1.1.json"
with open(file_path, "r") as json_file:
    squad_pt = json.load(json_file)

# Transformando o json em dataframe #

# Inicializando listas #
lista_titulo = []
lista_contexto = []
lista_questao = []
lista_resposta = []

# Obtendo os dados #
for item in squad_pt['data']:
    titulo = item['title']
    for paragraph in item['paragraphs']:
        contexto = paragraph['context']
        for qa in paragraph['qas']:
            questao = qa['question']
            resposta = qa['answers'][0]['text']
            
            # Append data to lists
            lista_titulo.append(titulo)
            lista_contexto.append(contexto)
            lista_questao.append(questao)
            lista_resposta.append(resposta)

# Criando o dataframe #
squad_pt = pd.DataFrame({
    'Titulo': lista_titulo,
    'Contexto': lista_contexto,
    'Questao': lista_questao,
    'Resposta': lista_resposta
})
print(squad_pt.head)