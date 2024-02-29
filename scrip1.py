# Importamos las bibliotecas generales
import pandas as pd
import sklearn
import matplotlib.pyplot as plt 

# Importamos los módulos específicos
 
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
 
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
  
   # Partimos el conjunto de entrenamiento. Para añadir replicabilidad usamos el random state
   X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)
   # Consultamos la fórmula para nuestra tabla
   print(X_train.shape)
   print(y_train.shape)
   
# Llamamos y configuramos nuestro algoritmo pca
   '''EL número de componentes es opcional, ya que por defecto si no le pasamos el número de componentes lo asignará de esta forma:
   a: n_components = min(n_muestras, n_features)'''
   pca = PCA(n_components=3)
   # Esto para que nuestro PCA se ajuste a los datos de entrenamiento que tenemos como tal
   pca.fit(X_train)
   #Como haremos una comparación con incremental PCA, haremos lo mismo para el IPCA.
   '''EL parámetro batch se usa para crear pequeños bloques, de esta forma podemos ir entrenandolos
   poco a poco y combinarlos en el resultado final'''
   ipca = IncrementalPCA(n_components=3, batch_size=10)
   #Esto para que nuestro PCA se ajuste a los datos de entrenamiento que tenemos como tal
   ipca.fit(X_train)
   ''' Aquí graficamos los números de 0 hasta la longitud de los componentes que me sugirió el PCA o que
   me generó automáticamente el pca en el eje x, contra en el eje y, el valor de la importancia
   en cada uno de estos componentes, así podremos identificar cuáles son realmente importantes
   para nuestro modelo '''
   plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
   plt.show()
   #Ahora vamos a configurar nuestra regresión logística
   logistic = LogisticRegression(solver='lbfgs')
   # Configuramos los datos de entrenamiento
   dt_train = pca.transform(X_train)
   dt_test = pca.transform(X_test)
   # Mandamos los data frames la la regresión logística
   logistic.fit(dt_train, y_train)
   #Calculamos nuestra exactitud de nuestra predicción
   print("SCORE PCA: ", logistic.score(dt_test, y_test))
   #Configuramos los datos de entrenamiento
   dt_train = ipca.transform(X_train)
   dt_test = ipca.transform(X_test)
   # Mandamos los data frames la la regresión logística
   logistic.fit(dt_train, y_train)
   #Calculamos nuestra exactitud de nuestra predicción
   print("SCORE IPCA: ", logistic.score(dt_test, y_test))
   