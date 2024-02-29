# Importamos las bibliotecas generales
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
# Importamos los módulos específicos
from sklearn.decomposition import KernelPCA
 
 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
if __name__ == "__main__":
 # Cargamos los datos del dataframe de pandas
 dt_heart = pd.read_csv('data/heart.csv')
 # Imprimimos un encabezado con los primeros 5 registros
 print(dt_heart.head(5))
 # Guardamos nuestro dataset sin la columna de target
 dt_features = dt_heart.drop(['target'], axis=1)
 # Este será nuestro dataset, pero sin la columna
 dt_target = dt_heart['target']
 # Normalizamos los datos
 dt_features = StandardScaler().fit_transform(dt_features)
  # Partimos el conjunto de entrenamiento y para añadir replicabilidad usamos el random state
 X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)
 
 #Aplicamos la función de kernel de tipo polinomial
 kpca = KernelPCA(n_components=4, kernel='poly' )
 #Vamos a ajustar los datos
 kpca.fit(X_train)
 
 #Aplicamos el algoritmo a nuestros datos de prueba y de entrenamiento
 dt_train = kpca.transform(X_train)
 dt_test = kpca.transform(X_test)
 
 #Aplicamos la regresión logística un vez que reducimos su dimensionalidad
 logistic = LogisticRegression(solver='lbfgs')
 
 #Entrenamos los datos
 logistic.fit(dt_train, y_train)
 
 #Imprimimos los resultados
 print("SCORE KPCA: ", logistic.score(dt_test, y_test))