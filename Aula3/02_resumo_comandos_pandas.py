import panda as pd

df.head()  # Primeiras linhas
df.tail()  # Ultimas linhas
df.shape  # Número de linhas e colunas
df.columns  # Nomes das colunas
df.describe()  # Estatísticas descritivas
df.info()  # Informações sobre o DataFrame

# Selecionando colunas
df['nome_da_coluna']  # Selecionando uma coluna
df[['coluna1', 'coluna2']]  # Selecionando mais de uma coluna

# Filtrando dados
df[df['coluna'] > valor]  # Filtrando linhas com base em uma condição

# Ordenação de dados

# Ordenando por uma coluna em ordem crescente
df.sort_values('coluna', ascending=True)
# Ordenando por uma coluna em ordem decrescente
df.sort_values('coluna', ascending=False)

# Valores ausentes
df.isnull()  # Verificando valores ausentes
df.isnull().sum()  # Contando valores ausentes por coluna


