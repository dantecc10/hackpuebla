import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Cargar el conjunto de datos
train_data = pd.read_csv('fake-news/train.csv')
test_data = pd.read_csv('fake-news/test.csv')

# Eliminar filas con valores nulos
train_data = train_data.dropna()

# Rellenar valores nulos en la columna 'text' con una cadena vacía
test_data['text'] = test_data['text'].fillna('')


# Preprocesar el texto
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data['text'])

X = tokenizer.texts_to_sequences(train_data['text'])
X = pad_sequences(X, maxlen=500)

y = train_data['label'].values

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocesar los datos de prueba (sin etiquetas)
X_test = tokenizer.texts_to_sequences(test_data['text'])
X_test = pad_sequences(X_test, maxlen=500)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D

# Construir el modelo
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=500))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val), verbose=2)

# Evaluar el modelo
loss, accuracy = model.evaluate(X_val, y_val, verbose=2)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Predecir en el conjunto de datos de prueba
predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int).flatten()

# Crear un DataFrame con las predicciones
submission = pd.DataFrame({'id': test_data['id'], 'label': predictions})

# Guardar el archivo de envío
submission.to_csv('fake-news/submission.csv', index=False)
