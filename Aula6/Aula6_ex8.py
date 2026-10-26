 

import re
import numpy as np


# =============================================================
# Funções do exercício guiado
# =============================================================
def tokenizar(texto):
    return re.findall(r"[a-záàâãéêíóôõúç]+", texto.lower())


def construir_vocabulario(textos_treino):
    palavras = set()
    for txt in textos_treino:
        palavras.update(tokenizar(txt))
    return {p: j for j, p in enumerate(sorted(palavras))}


def vetorizar(textos, vocab):
    X = np.zeros((len(textos), len(vocab)), dtype=int)
    for i, txt in enumerate(textos):
        for t in tokenizar(txt):
            if t in vocab:              
                X[i, vocab[t]] += 1
    return X


# =============================================================
# Modelo (etapas 5 a 11)
# =============================================================
class NaiveBayesDoZero:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):                                   # 9 · fit()
        self.classes_ = np.unique(y)

        # 5 · prioris P(C): fração de mensagens de cada classe
        priori = np.array([np.mean(y == c) for c in self.classes_])

        # 6 · frequências: quantas vezes cada palavra aparece em cada classe
        self.contagens_ = np.array([X[y == c].sum(axis=0) for c in self.classes_])

        # 7 · suavização de Laplace: soma alpha em todas as contagens
        totais = self.contagens_.sum(axis=1, keepdims=True)
        prob = (self.contagens_ + self.alpha) / (totais + self.alpha * X.shape[1])

        # 8 · log-probabilidades (evita multiplicar muitos números pequenos)
        self.log_priori_ = np.log(priori)
        self.log_prob_ = np.log(prob)
        return self

    def log_pontuacoes(self, X):
        return X @ self.log_prob_.T + self.log_priori_

    def predict(self, X):                                  # 10 · predict()
        S = self.log_pontuacoes(X)
        return self.classes_[np.argmax(S, axis=1)]

    def predict_proba(self, X):                            # 11 · predict_proba()
        S = self.log_pontuacoes(X)
        S = S - S.max(axis=1, keepdims=True)               # estabilidade numérica
        E = np.exp(S)
        return E / E.sum(axis=1, keepdims=True)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# =============================================================
# Teste rápido do núcleo 
# =============================================================
mini = ["promoção grátis grátis", "ganhe promoção", "reunião hoje", "agenda reunião"]
y_mini = np.array([1, 1, 0, 0])
vocab_mini = construir_vocabulario(mini)
nb_mini = NaiveBayesDoZero(alpha=1.0).fit(vetorizar(mini, vocab_mini), y_mini)
p = nb_mini.predict_proba(vetorizar(["grátis promoção"], vocab_mini))[0, 1]
print(f"Teste do núcleo: P(spam | 'grátis promoção') = {p:.4f}")
assert np.isclose(p, 0.8815, atol=1e-4)


# =============================================================
# 1 · Dataset: 48 mensagens (24 spam, 24 normais)
# =============================================================
spam = [
    "Parabéns você foi sorteado e ganhou um prêmio clique agora",
    "Oferta imperdível desconto de cinquenta por cento só hoje",
    "Ganhe dinheiro rápido trabalhando em casa sem experiência",
    "Seu cartão foi aprovado clique no link para liberar o crédito",
    "Promoção relâmpago frete grátis em toda a loja",
    "Você ganhou um iphone novo responda para receber",
    "Empréstimo aprovado sem consulta ao serasa ligue já",
    "Clique aqui e ganhe cupom de desconto exclusivo",
    "Última chance de garantir sua vaga no sorteio milionário",
    "Renda extra garantida invista hoje e lucre amanhã",
    "Seu cpf foi selecionado para receber um bônus em dinheiro",
    "Compre agora e ganhe outro grátis oferta limitada",
    "Atualize seus dados bancários para não perder o prêmio",
    "Liquidação total produtos com desconto de até setenta por cento",
    "Você foi escolhido para um teste grátis clique no link",
    "Ganhe pontos em dobro e troque por prêmios incríveis",
    "Crédito pré aprovado sem burocracia peça já o seu",
    "Promoção exclusiva para você cupom grátis só hoje",
    "Receba dinheiro no pix agora mesmo cadastre seu cpf",
    "Seu prêmio está esperando confirme seus dados no link",
    "Emagreça dez quilos em uma semana oferta especial",
    "Oferta de black friday antecipada descontos imperdíveis",
    "Ganhe um vale compras clicando no link abaixo",
    "Aproveite a promoção e ganhe frete grátis hoje",
]
normais = [
    "Oi tudo bem vamos almoçar amanhã",
    "A reunião de equipe foi remarcada para quinta feira",
    "Pode me enviar o relatório até sexta",
    "Chego em casa por volta das sete",
    "Não esquece de comprar pão e leite",
    "O professor publicou as notas da prova",
    "Vamos ao cinema no sábado à noite",
    "Segue em anexo a ata da reunião de ontem",
    "Minha mãe pediu para você ligar mais tarde",
    "O trabalho em grupo precisa ser entregue na segunda",
    "Obrigado pela ajuda com o projeto",
    "Vou me atrasar uns dez minutos para a aula",
    "A conta de luz chegou vou pagar amanhã",
    "Feliz aniversário muita saúde e alegria",
    "Você viu o jogo de ontem foi muito bom",
    "Precisamos marcar a revisão do contrato com o cliente",
    "O médico confirmou a consulta para terça",
    "Esqueci meu guarda chuva no seu carro",
    "Vamos estudar juntos para a prova de cálculo",
    "A agenda da semana já está no calendário",
    "Me avisa quando sair do trabalho",
    "O almoço de domingo vai ser na casa da vó",
    "Consegue revisar o código antes do deploy",
    "Bom dia a apresentação ficou ótima",
]
textos = spam + normais
y = np.array([1] * len(spam) + [0] * len(normais))   
print("\n1 · mensagens:", len(textos), "| spam:", y.sum(), "| normais:", (y == 0).sum())


# =============================================================
# 2 · Separar treino e teste (
# =============================================================
teste = [i for i in range(len(textos)) if i % 4 == 3]
treino = [i for i in range(len(textos)) if i % 4 != 3]
textos_treino = [textos[i] for i in treino]
textos_teste = [textos[i] for i in teste]
y_treino, y_teste = y[treino], y[teste]
print("2 · treino:", len(treino), "| teste:", len(teste),
      "| spam no teste:", y_teste.sum())

# 3 · Vocabulário só com o treino (sem vazamento)
vocab = construir_vocabulario(textos_treino)
print("3 · tamanho do vocabulário:", len(vocab))

# 4 · Textos em vetores de contagem
X_treino = vetorizar(textos_treino, vocab)
X_teste = vetorizar(textos_teste, vocab)
print("4 · X_treino:", X_treino.shape, "| X_teste:", X_teste.shape)


# =============================================================
# 5 a 11 · Treinar o modelo
# =============================================================
modelo = NaiveBayesDoZero(alpha=1.0).fit(X_treino, y_treino)
print("\n5 · prioris P(normal), P(spam):", np.exp(modelo.log_priori_))
print("7 · cada linha de P(palavra | classe) soma 1:", np.allclose(np.exp(modelo.log_prob_).sum(axis=1), 1))

proba = modelo.predict_proba(X_teste)
y_pred = modelo.predict(X_teste)
print("11 · predict_proba soma 1 por linha:", np.allclose(proba.sum(axis=1), 1))
print("10 · previsão = maior pontuação:",
      np.array_equal(y_pred, modelo.classes_[np.argmax(modelo.log_pontuacoes(X_teste), axis=1)]))


# =============================================================
# 12 · Acurácia e matriz de confusão
# =============================================================
tp = np.sum((y_teste == 1) & (y_pred == 1))
tn = np.sum((y_teste == 0) & (y_pred == 0))
fp = np.sum((y_teste == 0) & (y_pred == 1))
fn = np.sum((y_teste == 1) & (y_pred == 0))
M = np.array([[tn, fp], [fn, tp]])

print("\n12 · acurácia:", modelo.score(X_teste, y_teste))
print("12 · matriz de confusão (linhas = real, colunas = previsto):\n", M)
for txt, real, prev, pr in zip(textos_teste, y_teste, y_pred, proba[:, 1]):
    marca = "" if real == prev else "  <-- ERRO"
    print(f"   real {real} | prev {prev} | P(spam)={pr:.3f} | {txt}{marca}")


# =============================================================
# 13 · Testar mensagens novas
# =============================================================
novas = [
    "Parabéns você ganhou um cupom grátis clique no link",
    "Amanhã tem reunião com o cliente às nove",
    "Promoção de pizza hoje no restaurante da esquina",
]
X_novas = vetorizar(novas, vocab)
print("\n13 · mensagens novas:")
for txt, prev, pr in zip(novas, modelo.predict(X_novas), modelo.predict_proba(X_novas)[:, 1]):
    print(f"   {'SPAM  ' if prev == 1 else 'normal'} | P(spam)={pr:.3f} | {txt}")


# =============================================================
# 14 · Explicar duas previsões (peso de cada palavra)
# peso > 0 empurra para spam; peso < 0 empurra para normal
# =============================================================
inv_vocab = {j: p for p, j in vocab.items()}
peso = modelo.log_prob_[1] - modelo.log_prob_[0]     # log P(w|spam) - log P(w|normal)

def explicar(texto):
    x = vetorizar([texto], vocab)[0]
    print(f"\n14 · '{texto}'")
    print(f"   priori: {modelo.log_priori_[1] - modelo.log_priori_[0]:+.3f}")
    for j in np.flatnonzero(x):
        print(f"   {inv_vocab[j]:12s} x{x[j]}  peso {peso[j]:+.3f}")
    fora = [t for t in tokenizar(texto) if t not in vocab]
    print("   fora do vocabulário (ignoradas):", fora)
    print(f"   P(spam) = {modelo.predict_proba(x[None, :])[0, 1]:.3f}")

explicar(novas[0])
explicar(novas[2])


# =============================================================
# Extra · conferência com o MultinomialNB do sklearn
# =============================================================
from sklearn.naive_bayes import MultinomialNB
sk = MultinomialNB(alpha=1.0).fit(X_treino, y_treino)
print("\nExtra · probabilidades iguais ao MultinomialNB:",
      np.allclose(proba, sk.predict_proba(X_teste)))