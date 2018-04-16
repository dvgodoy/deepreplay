from keras.layers import Dense, Activation, BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD
from keras.initializers import glorot_normal, normal
from deepreplay.datasets.parabola import load_data
from deepreplay.callbacks import ReplayData

X, y = load_data()

sgd = SGD(lr=0.05)

def basic_model(activation, initializers):
    model = Sequential()
    model.add(Dense(units=2,
                    input_dim=2,
                    kernel_initializer=initializers[0],
                    activation=activation,
                    name='hidden'))

    model.add(Dense(units=1,
                    kernel_initializer=initializers[1],
                    activation='sigmoid',
                    name='output'))
    return model

def bn_model(activation, initializers):
    model = Sequential()
    model.add(Dense(units=2,
                    input_dim=2,
                    kernel_initializer=initializers[0],
                    name='hidden_linear'))
    model.add(BatchNormalization(name='hidden_bn'))
    model.add(Activation(activation, name='hidden_activation'))

    model.add(Dense(units=1,
                    kernel_initializer=initializers[1],
                    activation='sigmoid',
                    name='output'))
    return model


for seed in range(100):
    print('Using seed {}')
    replay = ReplayData(X, y, filename='part2_relu.h5', group_name='seed{:03}'.format(seed))

    glorot_initializer = glorot_normal(seed=seed)
    normal_initializer = normal(seed=42)

    model = basic_model('relu', [glorot_initializer, normal_initializer])

    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['acc'])

    model.fit(X, y, epochs=150, batch_size=16, callbacks=[replay])
