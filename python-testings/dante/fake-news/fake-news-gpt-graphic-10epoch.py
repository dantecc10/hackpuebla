import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Cargar el conjunto de datos
data = pd.read_csv('fake-news/train.csv')

# Manejo de valores nulos
data['text'].fillna('', inplace=True)

# Mezclar los datos
data = data.sample(frac=1).reset_index(drop=True)

# Preprocesar el texto
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(data['text'])

X = tokenizer.texts_to_sequences(data['text'])
X = pad_sequences(X, maxlen=500)

y = data['label'].values

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construir el modelo
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=500))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Definir callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_fake_news_model.keras', save_best_only=True, monitor='val_loss')

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), 
                    callbacks=[early_stopping, model_checkpoint], verbose=2)

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Guardar el modelo final
model.save('final_fake_news_model.keras')

# Función para predecir noticias nuevas
def predict_news(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=500)
    pred = model.predict(padded)
    return 'Fake' if pred < 0.5 else 'Real'

# Ejemplo de uso
news = "Noam Chomsky dies"
print(predict_news(news))

# Función para graficar la historia del entrenamiento
def plot_training_history(history):
    # Resumen de la precisión del modelo
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión del modelo')
    plt.ylabel('Precisión')
    plt.xlabel('Épocas')
    plt.legend(loc='upper left')

    # Resumen de la pérdida del modelo
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida del modelo')
    plt.ylabel('Pérdida')
    plt.xlabel('Épocas')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

# Graficar el historial de entrenamiento
plot_training_history(history)
