
import numpy as np
import tensorflow as tf
from Confusion_mat import multi_confu_matrix
from keras.layers import Conv2D, MaxPooling2D,Flatten
from keras.models import Sequential
from keras.layers import  Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.layers import Bidirectional, LSTM
from keras.layers import SimpleRNN, Dense
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler

def CNN(X_train, y_train, X_test, y_test):
    # reshaping data
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1, 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1, 1))


    model = Sequential()
    model.add(Conv2D(64, (1, 1), padding='valid', input_shape=X_train[1].shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(6, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=2, batch_size=10, verbose=0)
    y_predict = np.argmax(model.predict(X_test), axis=1)
    return y_predict, multi_confu_matrix(y_test, y_predict)


def DNN(x_train, y_train, x_test, y_test):
    """
    Deep Neural Network (DNN) for a numerical dataset using Keras.

    Parameters:
    - x_train, y_train: Training features and labels
    - x_test, y_test: Testing features and labels
    - num_epochs: Number of training epochs
    - batch_size: Batch size for training

    Returns:
    - Accuracy on the test set
    """
    # Standardize the input features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Create a Sequential model
    model = Sequential()

    # Input layer
    model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dropout(0.5))

    # Hidden layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer

    model.add(Dense(6, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)


    # Make predictions on the test data
    y_pred = np.argmax(model.predict(x_test), axis=1)

    cm = multi_confu_matrix(y_test, y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    return y_pred, cm


def RNN(x_train, y_train, x_test, y_test):
    # Standardize the input features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Reshape data for RNN input (assuming x_train and x_test are 2D arrays)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # Create a Sequential model
    model = Sequential()

    # Simple RNN layer
    model.add(SimpleRNN(64, activation='relu', input_shape=(x_train.shape[1], 1)))


    model.add(Dense(6, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    # Make predictions on the test data
    y_pred = np.argmax(model.predict(x_test), axis=1)

    cm = multi_confu_matrix((y_test, y_pred))

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    return y_pred, cm




# Custom RBM class (for demonstration; in practice, consider using a library or well-tested implementation)
class RBM:
    def __init__(self, n_visible, n_hidden):
        self.W = np.random.normal(0, 0.1, (n_visible, n_hidden))
        self.h_bias = np.zeros(n_hidden)
        self.v_bias = np.zeros(n_visible)

    def sample_h(self, v):
        h_prob = self.sigmoid(np.dot(v, self.W) + self.h_bias)
        return h_prob > np.random.rand(*h_prob.shape)

    def sample_v(self, h):
        v_prob = self.sigmoid(np.dot(h, self.W.T) + self.v_bias)
        return v_prob > np.random.rand(*v_prob.shape)

    def fit(self, data, n_iter=1000):
        for _ in range(n_iter):
            v = data
            h = self.sample_h(v)
            v_prime = self.sample_v(h)

            # Update weights and biases
            self.W += np.dot(v.T, h) - np.dot(v_prime.T, self.sample_h(v_prime))
            self.h_bias += np.mean(h, axis=0) - np.mean(self.sample_h(v_prime), axis=0)
            self.v_bias += np.mean(v, axis=0) - np.mean(v_prime, axis=0)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


def create_rbm_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())

    # RBM feature extraction layer
    n_visible = np.prod(input_shape)
    n_hidden = 128  # Number of hidden neurons
    rbm = RBM(n_visible, n_hidden)

    return model, rbm


def create_wdlstm_model(input_shape):
    model = models.Sequential()
    model.add(layers.LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(1, activation='sigmoid'))  # Adjust output layer for binary classification
    return model


def hybrid_optimization(model):
    # Placeholder for hybrid optimization logic
    # You can implement custom optimizers or any other hybrid strategies here
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


def rbm_wdlstm(X_train, y_train, X_test, y_test):
    # Preprocess data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Create RBM Model
    input_shape_rbm = (X_train.shape[1], X_train.shape[2], X_train.shape[3])  # Example for image data
    convnet, rbm = create_rbm_model(input_shape_rbm)

    # Fit RBM (this is simplified; use your data appropriately)
    rbm.fit(X_train_scaled)

    # Create WDLSTM Model
    input_shape_wdlstm = (X_train.shape[1], X_train.shape[2], 1)  # Adjust based on your feature representation
    wdlstm_model = create_wdlstm_model(input_shape_wdlstm)

    # Combine the ConvNet-RBM with WDLSTM for final predictions
    combined_input = layers.Input(shape=input_shape_rbm)
    x = convnet(combined_input)
    x = layers.Reshape((x.shape[1], 1))(x)  # Reshape to fit WDLSTM
    wdlstm_output = wdlstm_model(x)

    final_model = models.Model(inputs=combined_input, outputs=wdlstm_output)

    # Apply hybrid optimization
    hybrid_optimization(final_model)

    # Train the final model
    final_model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the model
    loss, accuracy = final_model.evaluate(X_test_scaled, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

