############################ From the tutorial ############################


from sentence_transformers import InputExample, datasets, models, SentenceTransformer, losses
from torch.utils.data import DataLoader

treino = [
    InputExample(texts=[squad_pt['Questao']]),
    InputExample(texts=[squad_pt['Contexto']]),
]
#train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)


batch_size = 24

loader = datasets.NoDuplicatesDataLoader(treino, batch_size=batch_size)

bert = models.Transformer('microsoft/mpnet-base')
pooler = models.Pooling(
    bert.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)

modelo = SentenceTransformer(modules=[bert,pooler])

loss = losses.MultipleNegativesRankingLoss(modelo)

# treinando #

epocas = 1
passos = int(len(loader) * epocas * 0.1)

modelo.fit(
    train_objectives=[(loader, loss)],
    epochs=epocas,
    warmup_steps=passos,
    output_path='mpnet-mnr-squad2',
    show_progress_bar=True
)

