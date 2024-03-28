from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Inicializacao do modelo #
#model_name = "deepset/roberta-base-squad2"
model_name = "timpal0l/mdeberta-v3-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Predicoes #
respostas = pipeline('question-answering', model=model, tokenizer=tokenizer)
QA_input = {
    'question': 'Where is the banana?',
    'context': 'Tem algo amarelo no lado da cama. A banana é amarela.'
}
res = respostas(QA_input)
print(res)

# Tentativa de classificar Q&A #
respostas = pipeline('question-answering', model=model, tokenizer=tokenizer)
QA_input = {
    'question': 'Por favor, classifique a questão: Eu acho esse filme meio lixo.',
    'context': 'Esse banco de dados é sobre reviews de filme, no qual existem dois tipos de frase, as positivas, que exibem algo a favor ou as negativas, que exibem algo contra.'
}
res = respostas(QA_input)
print(res)

# Tentativa de classificar Q&A #
respostas = pipeline('question-answering', model=model, tokenizer=tokenizer)
QA_input = {
    'question': 'Please, classify the sentiment of this following phrase: I hate this movie.',
    'context': 'This database is about movie reviews, in which there are two types of reviews, the positive, that are in favor of the movie and the negative, that is contrary to the movie review.'
}
res = respostas(QA_input)
print(res)

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
    'Title': lista_titulo,
    'Context': lista_contexto,
    'Question': lista_questao,
    'Answer': lista_resposta
})
print(squad_pt.head)
