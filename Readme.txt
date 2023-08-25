# Stock Price Prediction Assignment(I used Jupyter Notebook for this)

## Introduction
This assignment focuses on predicting stock prices using an LSTM-based model. The code provided demonstrates the process of loading stock data, preprocessing it, training an LSTM model, and making predictions.

## Instructions

### Prerequisites
Make sure you have the following libraries installed:
- numpy
- pandas
- sklearn
- tensorflow

### Dataset
1. Obtain the stock price dataset in CSV format. Ensure it contains columns for Date and Close price.
2. Save the dataset file in the same directory as the Python scripts.

### Usage
1. Open the Python script `200741_Purva.py` in your preferred IDE or editor.
2. Update the `file_path` variable in the `load_data` function with the path to your dataset file.
3. Run the script. It will perform the following steps:
    - Load the stock price data from the dataset.
    - Preprocess the data by scaling the Close prices.
    - Split the data into training and test sets.
    - Train an LSTM model on the training data.
    - Evaluate the model's performance using mean squared error (MSE) and directional accuracy.
    - Output the next day and next next day predictions.
4. Review the MSE and directional accuracy values printed in the console.

### Customization
- To adjust the LSTM model architecture, modify the `build_lstm_model` function in the script.
- To change the training parameters such as epochs and batch size, modify the `train_model` function.
- experiment with different hyperparameters to improve the model's performance.
- As we have to take the parameters from the trained data , we have to give the data parameters directly to the evaluate function.


### Training code
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def split_data(df, train_ratio=0.8):
    train_size = int(len(df) * train_ratio)
    train_data = df[:train_size]
    test_data = df[train_size:]
    return train_data, test_data


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(100))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


train_data, test_data = split_data(df)

X_train = train_data.iloc[:-2]['Close'].values
y_train = train_data.iloc[2:]['Close'].values

X_train = np.reshape(X_train, (X_train.shape[0], 1, 1))

model = build_lstm_model(input_shape=(1, 1))
model.fit(X_train, y_train, epochs=500, batch_size=8)

save_path = r'C:\Users\purva\Downloads\trained_model.h5'
model.save(save_path)
print(f'Trained model saved at: {save_path}')

