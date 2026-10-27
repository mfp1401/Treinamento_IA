# Aula 5.4 — Feature engineering

Criar **novas colunas** a partir das existentes pode ensinar ao modelo coisas que ele não enxergaria sozinho. Cada feature nova é testada com validação cruzada e só fica no modelo se trouxer ganho real.

## Conteúdos

- Criar features dentro do Pipeline com um transformador próprio
- Medir o ganho de cada feature contra um baseline (validação cruzada estratificada)
- Manter só as features acima de um ganho mínimo (0,002)
- Usar o teste uma única vez, no final

## Resultados (Titanic)

| Etapa | Acurácia (validação cruzada) |
|---|---|
| Baseline | 80,2% |
| + `Title` (título extraído do nome) | 81,9% (ganho +0,017) |
| + `Deck` (letra da cabine) | 82,6% (ganho +0,007) |

`FamilySize` e `FarePerPerson` não trouxeram ganho, e `IsAlone` piorou o resultado. A acurácia no teste ficou em **83,8%**.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `exercicio_guiado.py` | os 12 passos da aula, com as conclusões comentadas no final |

---

**← Anterior:** [Aula 5.3](../Aula5.3/README.md) · **Voltar:** [Aula 5](../README.md) · **Próxima →** [Aula 5.5](../Aula5.5/README.md)
