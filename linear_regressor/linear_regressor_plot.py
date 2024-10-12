import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def make_dataset(n_samples:int):
    x = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * x + np.random.randn(n_samples, 1)

    return x, y

n_samples = 100
x, y = make_dataset(n_samples)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=16)

model = LinearRegression()

fig, ax = plt.subplots()
ax.scatter(x, y, color="blue", label="Dados reais")
line, = ax.plot([], [], 'r-', label='Reta de Regressão')

# Definir limites do gráfico
ax.set_xlim(0, 2)
ax.set_ylim(0, 15)
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.legend()

# Lista para armazenar as previsões intermediárias
linhas = []

# Simulando o treinamento em etapas
for i in range(2, 101, 5):  # Começando a ajustar a partir de 2 pontos
    model.fit(X_train[:i], y_train[:i])  # Ajusta o modelo com i exemplos
    linhas.append(model.predict(np.array([[0], [2]])))  # Previsão da reta

# Função de atualização da animação
def update(num):
    line.set_data([0, 2], linhas[num])
    return line,

# Criando a animação
ani = FuncAnimation(fig, update, frames=len(linhas), interval=300, repeat=False)

# Exibindo a animação
plt.show()
