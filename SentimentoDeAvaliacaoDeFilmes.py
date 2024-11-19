import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt

# Load and preprocess the data
max_features = 20000  # Number of words to consider as features
maxlen = 100  # Cut texts after this number of words (among top max_features most common words)
batch_size = 32

# Load the data and split it into training and testing sets
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)

# Build the model
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model for 10 epochs and monitor the validation metrics
history = model.fit(input_train, y_train,
                    batch_size=batch_size,
                    epochs=10,
                    validation_data=(input_test, y_test))

# Evaluate the model
score, acc = model.evaluate(input_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# Display some entries of the word index
word_index = imdb.get_word_index()
for word, index in list(word_index.items())[:10]:
    print(f"Word: {word} - Index: {index}")

# Example real review with movie name
movie_name = "The Godfather"
real_review = "The Godfather is an amazing movie! The acting and direction are extraordinary."

# Convert the review to a sequence of numbers and pad it
real_review_sequence = [word_index[word] if word in word_index else 0 for word in real_review.lower().split()]
real_review_data = sequence.pad_sequences([real_review_sequence], maxlen=maxlen)

# Predict the sentiment
prediction = model.predict(real_review_data)
print(f"Review for '{movie_name}':", real_review)
print("Predicted sentiment score:", prediction[0][0])

if prediction[0][0] > 0.5:
    print("Positive review")
else:
    print("Negative review")

# Plot the training and validation metrics
# Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
