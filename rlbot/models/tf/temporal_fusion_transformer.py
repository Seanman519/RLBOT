"""Tf model.

Resources:
https://github.com/tsmatz/minecraft-rl-pigchase-attention/blob/master/train.py
https://keras.io/examples/structured_data/classification_with_grn_and_vsn/
https://arxiv.org/pdf/1912.09363.pdf

https://github.com/google-research/google-research/blob/master/tft/libs/tft_model.py

self attention + gated residual network
"""
from __future__ import annotations

from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
# Layer definitions.
concat = tf.keras.backend.concatenate
stack = tf.keras.backend.stack
K = tf.keras.backend
Add = tf.keras.layers.Add
LayerNorm = tf.keras.layers.LayerNormalization
Dense = tf.keras.layers.Dense
Multiply = tf.keras.layers.Multiply
Dropout = tf.keras.layers.Dropout
Activation = tf.keras.layers.Activation
Lambda = tf.keras.layers.Lambda
LeakyRelu = tf.keras.layers.LeakyReLU(alpha=0.01)


# Attention Components.
# @tf.function(jit_compile=True)
def get_decoder_mask(self_attn_inputs):
    """Returns causal mask to apply for self-attention layer.

    Args:
        self_attn_inputs (tf.keras.Layer):
            Inputs to self attention layer to determine mask shape

    Returns:
        tf.keras.Layer
            mask

    """
    len_s = tf.shape(self_attn_inputs)[1]
    bs = tf.shape(self_attn_inputs)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask


def scaled_dot_product_attention(q, k, v, mask, attn_dropout=0.0):
    """Defines scaled dot product attention layer.

    https://github.com/google-research/google-research/tree/master/tft

    Args:
        q (tf.keras.Layer):
            Queries
        k (tf.keras.Layer):
            Keys
        v (tf.keras.Layer):
            Values
        mask (tf.keras.Layer):
            Masking if required -- sets softmax to very large value
        attn_dropout (float):
            dropout rate

    Returns:
        Tuple of (layer outputs, attention weights)


    """
    activation = Activation("softmax")
    dropout = Dropout(attn_dropout)

    temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype="float32"))
    attn = K.batch_dot(q, k, axes=[2, 2]) / temper  # shape=(batch, q, k)
    if mask is not None:
        mmask = tf.float32.min * (1.0 - K.cast(mask, "float32"))  # setting to infinity
        attn = Add()([attn, mmask])
    attn = activation(attn)
    attn = dropout(attn)
    # output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
    output = K.batch_dot(attn, v)
    return output, attn


def interpretable_multi_head_attention(n_head, d_model, dropout, q, k, v, mask=None):
    """Interpretabl multi-head attention.

    https://github.com/google-research/google-research/tree/master/tft
    Defines interpretable multi-head attention layer.

    Applies interpretable multihead attention.

    Args:
        n_head (int)
            Number of heads
        d_model
            TFT state dimensionality
        dropout (float)
            Dropout rate to apply
        q (List[tf.keras.Layer])
            List of queries across heads - Query tensor of shape=(?, T, d_model)
        k (List[tf.keras.Layer])
            List of keys across heads - Key of shape=(?, T, d_model)
        v (List[tf.keras.Layer])
            List of values across heads - Values of shape=(?, T, d_model)
        mask (List[tf.keras.Layer])
            mask - Masking if required with shape=(?, T, T)

    Returns:
      Tuple of (layer outputs, attention weights)

    """
    d_k = d_v = d_k = d_v = d_model // n_head

    qs_layers = []
    ks_layers = []
    vs_layers = []

    # Use same value layer to facilitate interp
    vs_layer = Dense(d_v, use_bias=False)

    for _ in range(n_head):
        qs_layers.append(Dense(d_k, use_bias=False))
        ks_layers.append(Dense(d_k, use_bias=False))
        vs_layers.append(vs_layer)  # use same vs_layer

    w_o = Dense(d_model, use_bias=False)

    heads = []
    attns = []
    for i in range(n_head):
        qs = qs_layers[i](q)
        ks = ks_layers[i](k)
        vs = vs_layers[i](v)
        head, attn = scaled_dot_product_attention(qs, ks, vs, mask, dropout)

        head_dropout = Dropout(dropout)(head)
        heads.append(head_dropout)
        attns.append(attn)
    head = K.stack(heads) if n_head > 1 else heads[0]
    attn = K.stack(attns)

    outputs = K.mean(head, axis=0) if n_head > 1 else head
    outputs = w_o(outputs)
    outputs = Dropout(dropout)(outputs)  # output dropout

    return outputs, attn


def gated_residual_network(layer0, units, dropout):
    """Gated residual network.

    Args:
        layer0 (tf.keras.Layer)
        units (int)
        dropout (float)

    Returns:
        tf.keras.Layer

    """
    layer = Dense(units, activation="elu")(layer0)
    layer = Dense(units)(layer)
    layer = Dropout(dropout)(layer)
    gated_linear_unit = Dense(units)(layer) * Dense(
        units,
        activation="sigmoid",
    )(layer)

    if layer0.shape[-1] != units:
        layer0 = Dense(units)(layer0)
    layer = layer0 + gated_linear_unit
    layer = LayerNorm()(layer)
    return layer
