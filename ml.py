# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ml.py                                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: yohan <yohan@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/12 13:41:04 by yohan             #+#    #+#              #
#    Updated: 2025/02/12 13:41:06 by yohan            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from SGDmodel import AdalineSGD
from io import StringIO
from mlxtend.plotting import plot_decision_regions
import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np

s = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

response = requests.get(s)
response.raise_for_status()
df = pd.read_csv(StringIO(response.text), header=None,  encoding='utf-8')

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
X = (X - X.mean(axis=0)) / X.std(axis=0)


ada = AdalineSGD(learning_rate=0.01, n_iter=15, random_state=1)
ada.fit(X, y)
plt.figure(figsize=(10, 5))
plot_decision_regions(X, y, clf=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Plot Cost Function Convergence
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o', linestyle='--', color='b')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.title('Cost Function Convergence')
plt.grid(True)
plt.show()

def loop():
    print("Please write 'exit' to leave and 'examine' to enter AI Iris predictor")
    while (1):
        user_input = input()
        if user_input == "exit":
            break
        elif user_input == "examine":
            try:
                sepalLength = float(input("Enter sepal length of Iris: "))  # Convert input to float
                petalLength = float(input("Enter petal length of Iris: "))  # Convert input to float
            except ValueError:
                print("Please enter valid numerical values.")
                continue
            iris = "Iris Setosa" if ada.predict([sepalLength, petalLength]) == -1 else "Iris Versicolor"
            print(iris)
loop()