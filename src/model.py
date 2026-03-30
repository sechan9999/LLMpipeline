import tensorflow as tf
from tensorflow.keras import layers, Model

class Attention(layers.Layer):
    """Simple Query-Key-Value Attention mechanism for Bi-LSTM context."""
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, features):
        # features shape: (batch_size, time_steps, hidden_size)
        query_with_time_axis = tf.expand_dims(features[:, -1, :], 1)
        score = tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(features))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class SequenceModel(Model):
    """
    Bi-LSTM with Attention.
    Architecture:
    1. Embedding Layer
    2. Bidirectional LSTM (captures temporal dependencies in both directions)
    3. Custom Attention (prioritizes critical parts of the sequence)
    4. Dense Hidden with Dropout
    5. Sigmoid Classifier
    """
    def __init__(self, vocab_size, embed_dim=128, rnn_units=64):
        super(SequenceModel, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embed_dim)
        self.bi_lstm = layers.Bidirectional(
            layers.LSTM(rnn_units, return_sequences=True)
        )
        self.attention = Attention(rnn_units * 2)
        self.dense_hidden = layers.Dense(64, activation='relu')
        self.dropout = layers.Dropout(0.3)
        self.classifier = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.bi_lstm(x)
        
        # Apply Attention
        context_vector, _ = self.attention(x)
        
        x = self.dense_hidden(context_vector)
        if training:
            x = self.dropout(x)
        return self.classifier(x)
