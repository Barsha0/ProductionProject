import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv('C:\\Users\\user\\OneDrive\\Desktop\\Production\\Dataset\\Dataset.csv')

# Assume X contains your features and y contains your target variable
X = data.drop('Type', axis=1).values
y = data['Type'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the input data for Conv1D
X_train_reshaped = np.expand_dims(X_train, axis=-1)
X_test_reshaped = np.expand_dims(X_test, axis=-1)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped.reshape(X_train_reshaped.shape[0], -1))
X_test_scaled = scaler.transform(X_test_reshaped.reshape(X_test_reshaped.shape[0], -1))

# Reshape back to original shape
X_train_scaled = X_train_scaled.reshape(X_train_reshaped.shape)
X_test_scaled = X_test_scaled.reshape(X_test_reshaped.shape)


cnn_model = 'CNNmodel.h5'

# Check if the model file exists
if os.path.exists('CNNmodel.h5'):
    # Load the saved model
    model = tf.keras.models.load_model(cnn_model)
    print("Loaded CNN model from disk")
else:
    # Define and train the model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=X_train_reshaped.shape[1:]),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Save the model
    model.save('CNNmodel.h5')
    print("Saved model to disk")

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {accuracy}")

lstm_model = 'LSTMmodel.h5'

# Check if the LSTM model file exists
if os.path.exists('LSTMmodel.h5'):
    # Load the saved LSTM model
    lstm_model = tf.keras.models.load_model(lstm_model)
    print("Loaded LSTM model from disk")
else:
    # Define and train the LSTM model
    lstm_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=X_train_scaled.shape[1:]),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the LSTM model
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the LSTM model
    lstm_model.fit(X_train_scaled, y_train, epochs=2, batch_size=32, validation_split=0.2)

    # Save the LSTM model
    lstm_model.save('LTSMmodel.h5')
    print("Saved LSTM model to disk")

    
# Evaluate the LSTM model
lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test_scaled, y_test)
print(f"LSTM Test Accuracy: {lstm_accuracy}")


transformer_model = 'Transformermodel.h5'

# Check if the Transformer model file exists
if os.path.exists('Transformermodel.h5'):
    # Load the pre-trained Transformer model
    transformer_model = tf.keras.models.load_model(transformer_model)
    print("Loaded Transformer model from disk")
else:
    # Build the Transformer model
    query_input = Input(shape=(X_train_scaled.shape[1], 1))
    key_input = Input(shape=(X_train_scaled.shape[1], 1))
    value_input = Input(shape=(X_train_scaled.shape[1], 1))
    attention_output = MultiHeadAttention(num_heads=2, key_dim=1)(query_input, key_input, value_input)
    flatten_layer = Flatten()(attention_output)
    output = Dense(1, activation='sigmoid')(flatten_layer)
    transformer_model = Model(inputs=[query_input, key_input, value_input], outputs=output)

    # Compile the transformer model
    transformer_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the transformer model
    history_transformer = transformer_model.fit([X_train_scaled, X_train_scaled, X_train_scaled], y_train, epochs=1, batch_size=32, validation_split=0.2)
    
    # Save the trained model
    transformer_model.save('Transformermodel.h5')
    print("Saved Transformer model to disk")

    # Evaluate Model
    loss, accuracy = transformer_model.evaluate([X_test_scaled, X_test_scaled, X_test_scaled], y_test)
    print("Transformer Test Accuracy:", accuracy)



def extract_features_from_url(url):
    features = [
        len(url),                               # Length of the URL
        url.count('.'),                         # Number of dots in the URL
        sum(c.isdigit() for c in url),          # Number of digits in the URL
        sum(c.isalpha() for c in url),          # Number of alphabetic characters in the URL
        sum(not c.isalnum() for c in url),      # Number of special characters in the URL
        url.count('-'),                         # Number of hyphens in the URL
        url.count('_'),                         # Number of underscores in the URL
        url.count('/'),                         # Number of slashes in the URL
        url.count('?'),                         # Number of question marks in the URL
        url.count('='),                         # Number of equal signs in the URL
        url.count('@'),                         # Number of at symbols in the URL
        url.count('$'),                         # Number of dollar signs in the URL
        url.count('!'),                         # Number of exclamation marks in the URL
        url.count('#'),                         # Number of hashtag symbols in the URL
        url.count('%'),                         # Number of percent symbols in the URL
        len(url.split('/')[2].split('.')),      # Domain length
        url.count('-') - 2,                     # Number of hyphens in the domain
        sum(not c.isalnum() for c in url.split('/')[2]),  # Number of special characters in the domain
        sum(c.isdigit() for c in url.split('/')[2]),      # Number of digits in the domain
        sum(c.isalpha() for c in url.split('/')[2]),       # Number of alphabetic characters in the domain
        len(url.split('/')[2].split('.')),                  # Number of subdomains
        url.count('.') - 1,                                 # Number of dots in the subdomain
        sum(c == '-' for c in url.split('/')[2]),           # Whether the subdomain has a hyphen
        sum(not c.isalnum() for c in url.split('/')[2]),    # Whether the subdomain has special characters
        sum(c.isdigit() for c in url.split('/')[2]),        # Whether the subdomain has digits
        len(url.split('/')),                                # Average subdomain length
        sum(c == '.' for c in url.split('/')[2]),           # Average number of dots in the subdomain
        sum(c == '-' for c in url.split('/')[2]),           # Average number of hyphens in the subdomain
        sum(not c.isalnum() for c in url.split('/')[2]),    # Whether the subdomain has special characters
        sum(c.isdigit() for c in url.split('/')[2]),        # Whether the subdomain has digits
        len(set(url.split('/')[1:])),                       # Whether the subdomain has repeated digits
        len(url.split('?')) - 1,                            # Whether the URL has a query
        len(url.split('#')) - 1,                            # Whether the URL has a fragment
        len(url.split('=')) - 1,                            # Whether the URL has an anchor
        len(set(url)),                                      # Entropy of the URL
        len(set(url.split('/')[2])),                        # Entropy of the domain
    ]

    # If the number of features is less than 41, add placeholder values
    while len(features) < 41:
        features.append(0)  # Placeholder value

    return features[:41]  # Return only the first 41 features


def preprocess_url(url, scaler):
    # Extract features from the URL
    features = extract_features_from_url(url)
    
    # Ensure that the features list contains 41 elements
    if len(features) != 41:
        raise ValueError(f"Expected 41 features, but got {len(features)}")
    
    # Convert the features to a numpy array and reshape for scaling
    features_array = np.array(features).reshape(1, -1)
    
    # Scale the features using the provided scaler
    scaled_features = scaler.transform(features_array)
    
    # Reshape the scaled features for compatibility with the model
    reshaped_features = np.expand_dims(scaled_features, axis=-1)
    
    return reshaped_features

# Example URL for which to extract features
url = "https://www.instagram.com/"

# Preprocess a URL for prediction
preprocessed_url = preprocess_url(url, scaler)

# Make prediction
prediction = model.predict(preprocessed_url)
binary_prediction = (prediction > 0.5).astype(int)

# Display the prediction
if binary_prediction[0][0] == 1:
    print("The URL is predicted to be fake.")
else:
    print("The URL is predicted to be genuine.")
