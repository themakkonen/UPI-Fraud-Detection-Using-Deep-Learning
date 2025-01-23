import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Input, Dense, LSTM, Dropout
from flask import Flask, request, jsonify
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output

# Load the dataset
data = pd.read_csv('upi_transactions.csv')  # Ensure this file exists

# Drop unnecessary columns (modify based on your dataset)
if 'timestamp' in data.columns and 'user_id' in data.columns:
    data.drop(['timestamp', 'user_id'], axis=1, inplace=True)

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Scale the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Split the data into training and testing sets
X_train, X_test = train_test_split(scaled_data, test_size=0.2, random_state=42)

# Autoencoder for Anomaly Detection
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoder = Dense(32, activation='relu')(input_layer)
encoder = Dense(16, activation='relu')(encoder)
decoder = Dense(32, activation='relu')(encoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the Autoencoder
autoencoder.fit(X_train, X_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Anomaly Detection with Autoencoder
reconstructed_data = autoencoder.predict(X_test)
reconstruction_error = np.mean(np.square(X_test - reconstructed_data), axis=1)
threshold = np.percentile(reconstruction_error, 95)  # 95th percentile threshold
anomalies = reconstruction_error > threshold

# Prepare data for LSTM
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Labels for training the LSTM model (1 for fraud, 0 for normal)
y_train = np.array([1 if error > threshold else 0 for error in np.mean(np.square(X_train - autoencoder.predict(X_train)), axis=1)])
y_train = y_train[:X_train_lstm.shape[0]]  # Ensure label alignment

# LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the LSTM model
model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Predict on the test set
y_pred = model.predict(X_test_lstm)
y_pred = (y_pred > 0.5).astype(int)

# Evaluate the model
print(classification_report(anomalies, y_pred))

# Flask Deployment
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        input_data = request.get_json()
        # Convert input to a DataFrame and scale it
        input_df = pd.DataFrame(input_data)
        X_new = scaler.transform(input_df)
        X_new = X_new.reshape((1, 1, X_new.shape[1]))
        # Predict fraud probability
        y_pred = model.predict(X_new)
        return jsonify({'fraud_score': float(y_pred[0][0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
