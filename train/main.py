import tensorflow as tf

# not sure if this is the right alphabet, or if it should be the ASL alphabet
alphabet = 'abcdefghijklmnopqrstuvwxyz'

# Create the transformer model.
model = tf.keras.Sequential([
  # embedding layer will map each letter of the alphabet to a 128-dimensional vector.
  tf.keras.layers.Embedding(len(alphabet), 128),
  # LSTM layer will learn the temporal relationships between the letters
  tf.keras.layers.LSTM(256),
  # dense layer will output a probability distribution over the letters of the alphabet
  tf.keras.layers.Dense(len(alphabet))
])

# Compile the model.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# The value of x_train should be a matrix of shape (n,m), where n is the number of training examples and m is the length of each example. Each row of x
# train should represent a single training example. The values in each row should be the features of the example. The features can be anything that you think is important for the model to learn. For example, you could use the pixel values of a video frame, the spectrogram of a audio recording, or the text transcription of a sign.
# In the example code, x_train is a matrix of shape (100,128). This means that there are 100 training examples and each example has 128 features. The features are the 128-dimensional vectors that are output by the embedding layer.
# You can use any data format you want for x_train. However, it is important to make sure that the data is formatted in a way that is compatible with the model. For example, if you are using a convolutional neural network, you will need to make sure that the data is in a format that is compatible with convolutions.
x_train = [[1, 2, 3], [3, 4, 5]]
# not sure yet
y_train = [[1, 0, 0], [0, 1, 0]]
# Train the model.
model.fit(x_train, y_train, epochs=10)

# Evaluate the model.
# The loss function will be categorical crossentropy. This function measures
# the difference between the predicted probability distribution and the actual
# probability distribution. The optimizer will be Adam. This is an optimization
# algorithm that is often used for training deep learning models. The metrics
# will be accuracy and loss. Accuracy is the percentage of the test set that the
# model correctly predicts. Loss is a measure of how well the model fits the
# training data.

# The value of x_test should be a matrix of shape (n,m), where n is the number of test examples and m is the length of each example. Each row of x_test should represent a single test example. The values in each row should be the features of the example. The features can be anything that you think is important for the model to learn. For example, you could use the pixel values of a video frame, the spectrogram of a audio recording, or the text transcription of a sign.
x_test = [[1, 2, 3], [3, 4, 5]]
y_test = [[1, 0, 0], [0, 1, 0]]
loss, accuracy = model.evaluate(x_test, y_test)

# Print the results.
print('Test loss:', loss)
print('Test accuracy:', accuracy)
