"""Tf model.

Resources:
https://github.com/tsmatz/minecraft-rl-pigchase-attention/blob/master/train.py
https://keras.io/examples/structured_data/classification_with_grn_and_vsn/
https://arxiv.org/pdf/1912.09363.pdf

https://github.com/google-research/google-research/blob/master/tft/libs/tft_model.py

self attention + gated residual network
"""
from __future__ import annotations

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf

from rlbot.models.tf.gated_residual import gated_residual_network
from rlbot.models.tf.gated_residual import select_best_features

tf1, tf, tfv = try_import_tf()
Dense = tf.keras.layers.Dense
LeakyRelu = tf.keras.layers.LeakyReLU(alpha=0.01)


class AgentModel(TFModelV2):
    """Agent Model."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        """Init.

        Args:
            obs_space
            action_space
            num_outpts
            model_config
            name

        Returns:
            None

        """
        super().__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )

        for k, v in model_config["custom_model_config"].items():
            setattr(self, k, v)

        self.feat_groups = self.symbol_structure

        inputs = []
        symbol_features = []
        for k, v in self.feat_groups.items():
            name = f"{k}"
            shape = v
            in_layer = tf.keras.layers.Input(shape=shape, dtype="float32", name=name)
            inputs.append(in_layer)

            feat = in_layer

            split_dims = [1] * shape[1]

            feature_list = tf.split(feat, split_dims, axis=2)
            feature_list = [tf.squeeze(x, axis=-1) for x in feature_list]

            symbol_features.append(
                select_best_features(
                    feature_list,
                    self.encoding_size,
                    self.gr_units,
                    self.gr_dropout,
                ),
            )

        # Date array
        name = "date_arr"
        shape = self.input_shape[name]
        in_layer = tf.keras.layers.Input(shape=shape, dtype="float32", name=name)
        inputs.append(in_layer)
        in_layer = Dense(self.gr_units, activation=LeakyRelu)(in_layer)
        symbol_features.append(in_layer)

        # positions values
        name = "pos_val"
        shape = self.input_shape[name]
        in_layer = tf.keras.layers.Input(shape=shape, dtype="float32", name=name)
        inputs.append(in_layer)

        feat = in_layer
        split_dims = [1] * shape[1]
        feature_list = tf.split(feat, split_dims, axis=2)
        feature_list = [tf.squeeze(x, axis=-1) for x in feature_list]
        symbol_features.append(
            select_best_features(
                feature_list,
                self.encoding_size,
                self.gr_units,
                self.gr_dropout,
            ),
        )

        common_layer = tf.keras.layers.Concatenate(axis=-1)(symbol_features)

        # actor layers
        actor_out = common_layer
        for _ in range(self.actor_layer_depth):
            actor_out = gated_residual_network(
                actor_out,
                self.branch_layer_units,
                self.gr_dropout,
            )
        actor_out = Dense(self.final_layer_units, activation=LeakyRelu)(actor_out)
        actor_out = Dense(
            units=self.num_outputs,
            activation=None,
            dtype="float32",
        )(actor_out)

        # value function layers
        value_out = common_layer
        for _ in range(self.value_layer_depth):
            value_out = gated_residual_network(
                value_out,
                self.branch_layer_units,
                self.gr_dropout,
            )
        value_out = Dense(self.final_layer_units, activation=LeakyRelu)(value_out)
        value_out = Dense(
            units=1,
            activation=None,
            dtype="float32",
        )(value_out)

        self.base_model = tf.keras.models.Model(
            inputs=inputs,
            outputs=[actor_out, value_out],
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """Forward."""
        input_layers = []
        for k, _ in self.feat_groups.items():
            name = f"{k}"
            input_layers.append(input_dict["obs"][name])

        for name in ["date_arr", "pos_val"]:
            input_layers.append(input_dict["obs"][name])

        action_embed, self._value_out = self.base_model(input_layers)

        mask = input_dict["obs"]["mask"]
        inf_mask = tf.maximum(action_embed + tf.math.log(mask), tf.float32.min)
        return inf_mask, state

    # This is needed on "critic" process
    @override(ModelV2)
    def value_function(self):
        """Value function."""
        return tf.reshape(self._value_out, [-1])

    def import_from_h5(self, import_file):
        """Import from h5."""
        # Override this to define custom weight loading behavior from h5 files.
        self.base_model.load_weights(import_file)
