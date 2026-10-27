import matplotlib.pyplot as plt

Escolinha_de_ia = {
    "Alunos": ["Evelyn", "Lucas", "Gabriel", "Ana", "Rafael","Marcelo"],
    "Periodo": ["P4", "P2", "P1", "P3", "P2", "P5"]
}

plt.bar(Escolinha_de_ia["Alunos"], Escolinha_de_ia["Periodo"])
plt.xlabel("Alunos")
plt.ylabel("Período")
plt.title("Período por Aluno")

plt.show()

plt.hist(Escolinha_de_ia["Periodo"])
plt.xlabel("Período")
plt.ylabel("Número de períodos")
plt.title("Histograma de Período")

plt.show()

Escolinha_de_ia2 ={
    
    "Alunos": ["Evelyn", "Lucas", "Gabriel", "Ana", "Rafael","Marcelo"],
    "Faltas": [2, 5, 1, 0, 3, 4]
}

plt.scatter(Escolinha_de_ia2["Alunos"], Escolinha_de_ia2["Faltas"])
plt.xlabel("Alunos")
plt.ylabel("Faltas")
plt.title("Faltas por Aluno")

plt.show()

