import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from langchain_community.embeddings import OllamaEmbeddings
import ollama

# Fornecer um texto simples
texto = ["Ollama é uma poderosa ferramenta de embedding"]

# Define o modelo de embeddings

valor = OllamaEmbeddings(model="mxbai-embed-large")
vetor = valor.embed_documents(texto)

# Mostra o tamanho dos embeddings
embedding_tamanho = len(vetor[0])
print(f'O tamanho do embedding é: {embedding_tamanho}')

#Exibir o vetor de embeddings:
print(f'Valores iniciais: {vetor}')

