from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adadelta

def build_model(input_dim, hidden_layers, neurons, learning_rate, opt_name):
    opts = {
        'SGD': SGD(learning_rate),
        'Adam': Adam(learning_rate),
        'RMSprop': RMSprop(learning_rate),
        'Adadelta': Adadelta(learning_rate)
    }
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_dim=input_dim))
    for _ in range(hidden_layers-1):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(
        optimizer=opts[opt_name],
        loss='mse',
        metrics=['mae']
    )
    return model
