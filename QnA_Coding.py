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

