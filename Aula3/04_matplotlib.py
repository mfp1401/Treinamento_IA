import matplotlib.pyplot as plt
import numpy as np

nomes = ["Evelyn", "Lucas", "Gabriel", "Ana", "Rafael","Marcelo"]
faltas = [2, 5, 1, 0, 3, 4]

plt.bar(nomes, faltas)
plt.xlabel("Alunos")
plt.ylabel("Faltas")
plt.title("Faltas por Aluno")
plt.show()

plt.hist(faltas)
plt.xlabel("Número de Faltas")
plt.ylabel("Frequência")
plt.title("Histograma de Faltas")
plt.show()
