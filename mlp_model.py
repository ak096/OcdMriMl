hay100
akshay100
from keras.models import Sequential
from keras.layers import Dense


def make_model(input_size, layers):
    # create model
    model = Sequential()
    model.add(Dense(input_size, input_dim=input_size, kernel_initializer='normal', activation='relu'))

    for l in layers:
        model.add(Dense(l, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model
