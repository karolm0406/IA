# Manipulación de datos
import pandas as pd

# Operaciones numéricas
import numpy as np

# Para medir el tiempo que toma ejecutar los procesos
from time import time

# Para separar datos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

# Librería para SVM (Support Vector Machine)
from sklearn.svm import SVC

# Medición de precisión
from sklearn.metrics import accuracy_score, confusion_matrix

# Generar gráficos
import matplotlib.pyplot as plt

# Leemos el set de datos y lo cargamos en la variable diabetes_df, que es un DataFrame
diabetes_df = pd.read_csv('diabetes.csv')   

# Mostrar información sobre el conjunto de datos
diabetes_df.info()

# Mostrar las primeras 50 filas del DataFrame
diabetes_df.head(50)

# Contar los valores únicos de la columna 'Outcome' en el DataFrame
diabetes_df['Outcome'].value_counts()

# Creamos un nuevo DataFrame llamado X con las columnas de características
# Se obtiene generando una lista de columnas del DataFrame original
lista_caract = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'    
]

# Luego, tomamos esas columnas del DataFrame original
X = diabetes_df[lista_caract]

# Mostramos los primeros cinco registros para conocer cómo se compone
X.head()

#Utilizaremos el mismo procedimiento para generar y
lista_etiq=['Outcome']
y=diabetes_df[lista_etiq]
y.head()

#Separar en datos de entrenamiento y datos de prueba
X_train, X_test, y_train, y_test= train_test_split(
    X,
    y
)  

#Mostraremos la cantidad a utilizar
X_train.shape
y_train.shape

#Luego, la cantidad de datos a utilozar para
X_test.shape
y_test.shape

#Maquina de soporte
clf=SVC(kernel='linear')

#Guardamos el registro del momento en el que empezamos
hora_inicio=time()

#Iniciamos el entrenamiento ejecutando el metodo fit
#Los valores que enviamos son los valores de X y y
#
# El .ravel() que final de y.values es un pequeño truco patra cambiar su forma 
#esto permite convertir una matriz de dos dimensiones en una sola dimnsion
#con ello,  cada elemento de esta nueva matriz corresponde a un registro cd X
clf.fit(X_train.values, y_train.values.ravel())

#Imprimimos el tiempo tomado para el entrenamiento
print("Entrenamineto terminado en {} segundos".format(time()- hora_inicio))

#Otra vez guardaremos registro del tiempo que nos toma crear esta predicción
hora_inicio= time()
#Iniciamos la prediccion con nuestra X de prueba
y_pred = clf.predict(X_test)
#Mostramos el tiempo tomado para la predicción
print("Prediccion terminada en {} segundos" .format(time()-hora_inicio))

#Evaluamos la precision
accuracy_score(y_test,y_pred)

#Matriz de cofusion: Nos ayuda a tener una mejor idea del rendimiento de nuestro rendimiento de datos de prueba(y_test en este caso) y nuestros datos calculados(y_pred en este caso)
#la funcion confusion_matrix recibe las "respuestas correctas" y nuestras predicciones
#genera una matriz que indica, para cada clase, la canbtidad de predicciones correctas
conf_diabetes= confusion_matrix(y_test, y_pred)
conf_diabetes

def plot_cm(cm, classes):
    """Esta funcion s envcarga de generar un gráfico con nuestra matriz de confusion, cm es l matriz generada por confusion_matrix
    classes es una lista que contiene las posibles clases que puede predecir nuestro"""
    
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Matriz de confusión')
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh=cm.max()/2.
    for indice_fila, fila in enumerate(cm):
        for indice_columna, columna in enumerate(fila):
            if cm[indice_fila, indice_columna] > thresh:
                color="white"
            else:
                color="black"
            plt.text(
                indice_columna,
                indice_fila,
                cm[indice_fila, indice_columna],
                color=color,
                horizontalaligment="center"
            )
    plt.ylabel("valores reales")
    plt.xlabel("valores calculados")
    plt.show()
    
    #Generamos el grafico llamadp la funcion que creamos y envindo los parametros
    #cm  muestra matriz de confusion (conf_diabetes)
    #clases = las clases a predecir (si tienen diabetes o no)
    plot_cm(conf_diabetes, ['No diabetes', 'Si diabetes'])