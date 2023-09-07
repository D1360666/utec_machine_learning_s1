import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import pandas as pd
from ydata_profiling import ProfileReport
from ydata_profiling.utils.cache import cache_file
file_name="./Clientes.csv"

column_name = ["Email","Address","Avatar","Avg._Session_Length","Time_on_App","Time_on_Website","Length_of_Membership","Yearly_Amount_Spent"]

df = pd.read_csv(file_name, sep=",", names=column_name)


profile = ProfileReport(df, title="Profiling Report", explorative=True,
                        html={"style":{"full_width": True}}, sort=None, progress_bar=False)

df.to_csv("./Clientes2.csv")

new_file_name = "./Clientes2.csv"
df2 = pd.read_csv(new_file_name, sep=",", names=column_name)
df2.drop(index=df.index[0], axis=6, inplace=True)
rofile = ProfileReport(df2, title="Profiling Report", explorative=True,
                        html={"style":{"full_width": True}}, sort=None, progress_bar=False)

profile.to_notebook_iframe()
profile.to_file("archivo.html")
rofile.to_notebook_iframe()
rofile.to_file("archivo2.html")
print(df)
print('df 2 --------------------------------')
print(df2)

df_train = df2.sample(frac=0.8, random_state=None)
df_test = df2.sample(frac=0.2, random_state=None)

print('TRAIN------------------------------')
print(df_train)
print(df_train.count)
print('TEST------------------------------')
print(df_test)
print(df_test.count)
# Datos de ejemplo (variables independientes x y variables dependientes y)
x = np.array([float(df_train['Time_on_Website'].values)])
y = np.array([float(df_test['Time_on_Website'].values)])

#Cómo encontrar la precisión con Python y Sklearn
#precision_score(df_train['Time_on_App'], df_test['Time_on_App'])
#recall_score(df_train['Time_on_App'].values, df_test['Time_on_App'].values)
# Crear un objeto de regresión lineal
modelo = LinearRegression()

# Ajustar el modelo a los datos
modelo.fit(x.reshape(-1, 1), y)

# Realizar predicciones con el modelo entrenado
y_pred = modelo.predict(x.reshape(-1, 1))

# Visualizar los resultados
plt.scatter(x, y, label='Datos reales')
plt.plot(x, y_pred, color='red', label='Regresión lineal')
plt.xlabel('Variable independiente (x)')
plt.ylabel('Variable dependiente (y)')
plt.legend()
plt.show()

# Imprimir los coeficientes de la línea recta (pendiente e intercepto)
print("Pendiente (coeficiente 'a'): ", modelo.coef_[0])
print("Intercepto (coeficiente 'b'): ", modelo.intercept_)
