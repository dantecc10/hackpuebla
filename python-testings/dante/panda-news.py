import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# Ejemplo de datos
data = {
    'title': [
        'NASA confirma la existencia de vida en Marte',
        'La Tierra es plana según un nuevo estudio',
        'Nuevo avance en la cura del cáncer',
        'Los OVNIs atacaron una ciudad en Rusia'
    ],
    'source': [
        'space.com',
        'flatearthsociety.org',
        'medicalnewstoday.com',
        'ufo-blog.com'
    ],
    'label': [
        'true',
        'false',
        'true',
        'false'
    ]
}

df = pd.DataFrame(data)

# Listas de fuentes confiables y no confiables
trusted_sources = ['space.com', 'medicalnewstoday.com']
untrusted_sources = ['flatearthsociety.org', 'ufo-blog.com']

# Función para asignar fiabilidad a las fuentes
def source_reliability(source):
    if source in trusted_sources:
        return 1  # Fuente confiable
    elif source in untrusted_sources:
        return 0  # Fuente no confiable
    else:
        return 0.5  # Fuente desconocida

# Añadir columna de fiabilidad de la fuente
df['source_reliability'] = df['source'].apply(source_reliability)

# Convertir las etiquetas de texto a numéricas
df['label_num'] = df['label'].map({'true': 1, 'false': 0})

# Vectorizar los títulos de las noticias
vectorizer = CountVectorizer()

# Transformar los títulos en vectores de características
X_title = vectorizer.fit_transform(df['title']).toarray()
X_reliability = df['source_reliability'].values.reshape(-1, 1)

# Concatenar las características del título y la fiabilidad de la fuente
X = np.hstack((X_title, X_reliability))

y = df['label_num'].values

# Separar los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')

# Función para predecir la veracidad de una noticia
def predict_news(title, source):
    reliability = source_reliability(source)
    title_vector = vectorizer.transform([title]).toarray()
    title_with_reliability = np.hstack((title_vector, [[reliability]]))
    prediction = model.predict(title_with_reliability)
    return 'true' if prediction >= 0.5 else 'false'

# Ejemplo de verificación de una nueva noticia
new_title = 'NASA descubre agua en Marte'
new_source = 'space.com'
print(predict_news(new_title, new_source))  # Debería devolver 'true' ya que 'space.com' es una fuente confiable

# Otro ejemplo de verificación de una noticia
new_title = 'Los OVNIs atacaron una ciudad en Rusia'
new_source = 'ufo-blog.com'
print(predict_news(new_title, new_source))  # Debería devolver 'false' ya que 'ufo-blog.com' es una fuente no confiable

# Graficar los resultados de accuracy y val_accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy and Validation Accuracy Over Epochs')
plt.show()
