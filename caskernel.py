import config
import pickle
from utils.tools import *
import tensorflow.keras as keras

def caskernel_loss(y_true, y_pred):
    mse = keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
    return mse


with open(config.train, 'rb') as ftrain:
    train_input, train_label = pickle.load(ftrain)
with open(config.val, 'rb') as fval:
    val_input, val_label = pickle.load(fval)
with open(config.test, 'rb') as ftest:
    test_input, test_label = pickle.load(ftest)


# ****************************************
# hyper-parameters
learning_rate = 5e-4
batch_size = 64
sequence_length = config.max_sequence
embedding_dim = config.gc_emd_size
z_dim = 64
rnn_units = 128
verbose = 2
patience = 10
dropout = 0.3
# hyper-parameters
# ****************************************


caskernel_inputs = keras.layers.Input(shape=(sequence_length, embedding_dim))
masked_input = keras.layers.Masking(mask_value=0., input_shape=(sequence_length, embedding_dim))(caskernel_inputs)
bn_caskernel_inputs = keras.layers.BatchNormalization()(masked_input)

gru_1 = keras.layers.Bidirectional(keras.layers.GRU(rnn_units, return_sequences=True, dropout=dropout))(
    bn_caskernel_inputs)

multi = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=rnn_units, dropout=dropout)(gru_1, gru_1, gru_1)
pool = tf.reduce_max(multi, 1)


mlp = tf.keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(dropout),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(dropout),
    keras.layers.Dense(1)
])
outputs = mlp(pool)

caskernel = keras.Model(inputs=caskernel_inputs, outputs=outputs)

caskernel.summary()

optimizer = keras.optimizers.Adam(lr=learning_rate)
caskernel.compile(loss=caskernel_loss, optimizer=optimizer, metrics=['msle'])
train_generator = Generator(train_input, train_label, batch_size, sequence_length)
val_generator = Generator(val_input, val_label, batch_size, sequence_length)
test_generator = Generator(test_input, test_label, batch_size, sequence_length)
early_stop = keras.callbacks.EarlyStopping(monitor='val_msle', patience=patience, restore_best_weights=True)
train_history = caskernel.fit_generator(train_generator,
                                    validation_data=val_generator,
                                    epochs=1000,
                                    verbose=verbose,
                                    callbacks=[early_stop],
                                    )
print('Training end!')
caskernel.evaluate_generator(test_generator, verbose=1)


