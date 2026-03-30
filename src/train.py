import tensorflow as tf

@tf.function # Compiles to static graph for production execution speed
def train_step(model, x, y, optimizer, loss_fn, metric):
    """
    Custom training step with Gradient Clipping.
    Clipping prevents the 'exploding gradient' problem in deep sequence models.
    """
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    
    grads = tape.gradient(loss_value, model.trainable_weights)
    
    # Senior-level optimization: Global Norm Clipping (stability focus)
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    metric.update_state(y, logits)
    
    return loss_value
