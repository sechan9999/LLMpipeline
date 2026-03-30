import tensorflow as tf
import numpy as np

class TextPipeline:
    """
    A production-grade tf.data input pipeline for NLP.
    Demonstrates:
    1. Vectorization (via Keras TextVectorization for OS compatibility)
    2. Parallel processing using tf.data.AUTOTUNE
    3. Prefetching to minimize GPU starvation
    4. Dataset shuffling and batching
    """
    def __init__(self, vocab_size=5000, batch_size=32, max_seq_len=128):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=vocab_size,
            output_mode='int',
            output_sequence_length=max_seq_len
        )

    def adapt(self, text_samples):
        """Adapt vectorization layer to text samples."""
        self.vectorize_layer.adapt(text_samples)

    def _preprocess(self, text, label):
        """Vectorizes text into integer sequences."""
        tokens = self.vectorize_layer(text)
        return tokens, label

    def create_dataset(self, texts, labels, shuffle=True):
        """
        Creates a tf.data.Dataset with optimized loading strategies.
        """
        dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)

        # Mapping with parallel execution (AUTOTUNE)
        dataset = dataset.map(
            self._preprocess, 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        dataset = dataset.batch(self.batch_size)
        
        # Crucial for preventing GPU starvation
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
