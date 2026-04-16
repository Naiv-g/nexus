import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        return self.combine_heads(concat_attention)

def build_advanced_nifty_model(seq_len=10, n_features=46):
    inputs = layers.Input(shape=(seq_len, n_features))
    x = layers.Dense(128)(inputs)
    x = layers.LayerNormalization()(x)
    lstm_1 = layers.Bidirectional(layers.LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(x)
    lstm_1 = layers.SpatialDropout1D(0.3)(lstm_1)
    attn_out = MultiHeadSelfAttention(embed_dim=256, num_heads=4)(lstm_1)
    attn_out = layers.Add()([lstm_1, attn_out])
    attn_out = layers.LayerNormalization()(attn_out)
    lstm_2 = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(attn_out)
    x = layers.BatchNormalization()(lstm_2)
    dense = layers.Dense(128, activation='swish')(x)
    dense = layers.Dropout(0.4)(dense)
    dense = layers.Dense(64, activation='swish')(dense)
    direction_out = layers.Dense(2, activation='softmax', name='direction_output')(dense)
    volatility_out = layers.Dense(1, activation='linear', name='volatility_output')(dense)
    model = Model(inputs=inputs, outputs=[direction_out, volatility_out])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss={'direction_output': 'sparse_categorical_crossentropy', 'volatility_output': 'huber'},
        loss_weights={'direction_output': 1.0, 'volatility_output': 0.5},
        metrics={'direction_output': 'accuracy', 'volatility_output': 'mae'}
    )
    return model

def load_and_train():
    with open('data/processed.pkl', 'rb') as f:
        d = pickle.load(f)
    
    def reshape_data(x):
        return x.reshape(-1, 10, 46)

    X_train = reshape_data(d['X_tr'])
    X_val = reshape_data(d['X_val'])
    
    model = build_advanced_nifty_model()
    
    model.fit(
        X_train, 
        {'direction_output': d['y_dir_tr'], 'volatility_output': d['y_vol_tr']},
        validation_data=(X_val, {'direction_output': d['y_dir_val'], 'volatility_output': d['y_vol_val']}),
        epochs=50,
        batch_size=64,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )
    
    model.save('data/lstm_final_model.h5')

if __name__ == "__main__":
    load_and_train()

