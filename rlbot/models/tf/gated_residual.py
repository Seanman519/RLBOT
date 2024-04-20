"""Tf model.

Resources:
https://keras.io/examples/structured_data/classification_with_grn_and_vsn/

self attention + gated residual network
"""
from __future__ import annotations

from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()


LayerNorm = tf.keras.layers.LayerNormalization
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Activation = tf.keras.layers.Activation


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


def select_best_features(feature_list, encoding_size, gr_units, gr_dropout):
    """Select best features.

    Args:
        feature_list (tf.keras.Layer)

    Returns:
        tf.keras.Layer

    """
    # list for storing all symbol layers
    encoded_features = []

    # summarise feature for each symbol
    for feature in feature_list:
        # for _ in range(self.feature_encoding_depth):
        #     feature = gated_residual_network(feature,self.gr_units,self.gr_dropout)
        feature = Dense(encoding_size)(feature)
        encoded_features.append(feature)

    # Variable Selection
    c = tf.keras.backend.concatenate(encoded_features)
    c = gated_residual_network(c, gr_units, gr_dropout)
    c = tf.expand_dims(
        Dense(units=len(encoded_features), activation="softmax")(c),
        axis=-1,
    )

    g = []
    for feature in encoded_features:
        g.append(gated_residual_network(feature, gr_units, gr_dropout))
    g = tf.stack(g, axis=1)
    new_layer = tf.squeeze(tf.matmul(c, g, transpose_a=True), axis=1)
    return new_layer
