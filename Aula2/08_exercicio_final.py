nome = ["Evelyn", "Pedro", "Gustavo", "Mateus", "Felipe","Marcelo"]
nota = [8,6,9,5,7,10]

aprovados =0
reprovados =0

for nome in nome:
    print(nome)

for nota in nota:
    if nota >= 7:
        aprovados += 1
    else:
        reprovados += 1

print(len(nome))
print("Aprovados: " + str(aprovados))
print("Reprovados: " + str(reprovados))

      




