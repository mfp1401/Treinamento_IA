import pandas as pd
Aluno = {
    "Nome": ["João", "Maria", "Pedro","Marcelo", "Ana","Lucas", "Carla", "Rafael", "Fernanda", "Bruno"],
    "Idade": [20, 22, 19, 21, 23, 20, 22, 21, 19, 23],
    "Curso": ["Engenharia", "Medicina", "Direito", "Arquitetura", "Administração", "Engenharia", "Medicina", "Direito", "Arquitetura", "Administração"],
    "Nota": [8.5, 9.0, 7.5, 8.0, 9.5, 7.0, 8.5, 9.0, 7.5, 8.0]
}

df = pd.DataFrame(Aluno)
print(df['Nome'])

df.head() # Exibe as primeiras 5 linhas do DataFrame
df.tail(5) # Exibe as últimas 5 linhas do DataFrame
df.shape # Exibe o número de linhas e colunas do DataFrame
df.columns # Exibe os nomes das colunas do DataFrame
df.describe() # Exibe estatísticas descritivas do DataFrame
df.info() # Exibe informações sobre o DataFrame
df['Nota'] # Seleciona a coluna "Nota" do DataFrame
df[['Nome', 'Nota']] # Seleciona as colunas "Nome" e "Nota"
df[df['Nota'] >= 7.0] # Filtra as linhas do DataFrame onde a nota é maior ou igual a 7
df.sort_values('Nota', ascending=False) # Ordena o DataFrame pela coluna "Nota" em ordem decrescente
df.sort_values('Nome') # Ordena o DataFrame pela coluna "Nome" em ordem crescente
df.isnull() # Verifica se há valores ausentes no DataFrame

print(df.head(5))          # Primeiras 5 linhas
print(df.tail(5))         # Últimas 5 linhas
print(df.shape)           # Quantidade de linhas e colunas
print(df.columns)         # Nomes das colunas
print(df.describe())      # Estatísticas
df.info()                 # Informações do DataFrame

print(df['Nota'])         # Coluna Nota
print(df[['Nome', 'Nota']])  # Nome e Nota

print(df[df['Nota'] >= 7.0]) # Notas >= 7

print(df.sort_values('Nota', ascending=False)) # Ordena por Nota
print(df.sort_values('Nome'))                  # Ordena por Nome

print(df.isnull())         # Verifica valores nulos