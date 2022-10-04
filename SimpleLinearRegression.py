# importando
from sklearn.metrics import r2_score
from sklearn import linear_model
import pandas as pd
import numpy as np

# Preparando dados

df = pd.read_csv("./FuelConsumption.csv")

cdf = df[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]]

# Criando um dataset de treino e test

msk = np.random.rand(len(df)) < 0.8 # Usando 80% para treino
train = cdf[msk]
test = cdf[~msk] # resto para teste

# Modelando nossos dados com sklearn

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[["ENGINESIZE"]])
train_y = np.asanyarray(train[["CO2EMISSIONS"]])
regr.fit(train_x, train_y)
print("Coeficiente: ", regr.coef_)
print("Interceptador: ", regr.intercept_)

# Mostando o MAE, RMSE e R-squared da previsão:
print("")
print("======================")
print("    Com Emginesize    ")
print("======================")
print("")
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )

# Redefinindo e testando com outros dados
print("")
print("======================")
print("    Com FUELC_COMB    ")
print("======================")
print("")

train_x = train[["FUELCONSUMPTION_COMB"]]
test_x = test[["FUELCONSUMPTION_COMB"]]

regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)
predictions = regr.predict(test_x)
print("Mean Absolute Error: %.2f" % np.mean(np.absolute(predictions - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )