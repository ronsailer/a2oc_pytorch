import lasagne
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import torch

from collections import OrderedDict

if theano.config.device.startswith("gpu"):
	pass
import numpy as np


def get_activation(activation):
	if activation == "softmax":
		output = torch.nn.Softmax()
	elif activation is None:
		output = None
	elif activation == "tanh":
		output = torch.nn.Tanh()
	elif activation == "relu":
		output = torch.nn.ReLU()
	elif activation == "linear":
		output = None
	elif activation == "sigmoid":
		output = torch.nn.Sigmoid()
	else:
		print "activation not recognized:", activation
		raise NotImplementedError
	return output


class MLP3D():
	def __init__(self, input_size=None, num_options=None, out_size=None, activation="softmax"):
		option_out_size = out_size
		limits = (6. / np.sqrt(input_size + option_out_size)) / num_options
		self.options_W = torch.from_numpy(
				np.random.uniform(size=(num_options, input_size, option_out_size), high=limits, low=-limits).astype(
						"float32"))
		self.options_b = torch.from_numpy(np.zeros(shape=(num_options, option_out_size), dtype="float32"))
		self.activation = get_activation(activation)
		self.params = [self.options_W, self.options_b]

	def apply(self, inputs, option=None):
		W = self.options_W[option]
		b = self.options_b[option]

		out = T.sum(inputs.dimshuffle(0, 1, 'x') * W, axis=1) + b
		return out if self.activation is None else self.activation(out)

	def save_params(self):
		return [i.get_value() for i in self.params]

	def load_params(self, values):
		print "LOADING NNET..",
		for p, value in zip(self.params, values):
			p.set_value(value.astype("float32"))
		print "LOADED"


class Model():
	def __call__(self, *args, **kwargs):
		return self.apply(*args, **kwargs)

	def get_activation(self, model):
		activation = model["activation"] if "activation" in model else "linear"
		return get_activation(activation)

	def create_layer(self, inputs, model):

		if model["model_type"] == "conv":
			stride = model["stride"] if "stride" in model else 1
			# class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)[source]
			layer = torch.nn.Conv2d(inputs,
			                  model["out_size"],
			                  kernel_size=model["filter_size"],
			                  stride=stride)
		elif model["model_type"] == "mlp":
			# class torch.nn.Linear(in_features, out_features, bias=True)[source]
			layer = torch.nn.Linear(inputs, out_features=model["out_size"])
			if "W" in model:
				torch.nn.init.constant(layer.weight, model["W"])
			if "b" in model:
				torch.nn.init.constant(layer.bias, model["b"])

		elif model["model_type"] == "activation":
			layer = self.get_activation(model)
		else:
			print "UNKNOWN LAYER NAME"
			raise NotImplementedError
		return layer

	def __init__(self, model_in, input_size=None, rng=1234):
		"""
		example model:
		model = [{"model_type": "conv", "filter_size": [5,5], "pool": [1,1], "stride": [1,1], "out_size": 5},
				 {"model_type": "conv", "filter_size": [7,7], "pool": [1,1], "stride": [1,1], "out_size": 15},
				 {"model_type": "mlp", "out_size": 300, "activation": "tanh"},
				 {"model_type": "mlp", "out_size": 10, "activation": "softmax"}]
		"""
		self.theano_rng = RandomStreams(rng)
		rng = np.random.RandomState(rng)
		lasagne.random.set_rng(rng)

		new_layer = tuple(input_size) if isinstance(input_size, list) else input_size
		model = [model_in] if isinstance(model_in, dict) else model_in

		print "Building following model..."
		print model

		self.model = model
		self.input_size = input_size
		self.out_size = model_in[-1]["out_size"]

		# create neural net layers
		self.params = []
		layers_dict = OrderedDict()

		for i, m in enumerate(model):
			layer = self.create_layer(new_layer, m)

			if layer:
				self.params += list(layer.parameters())
				layers_dict[m["model_type"] + str(i)] = layer

		self.layers = torch.nn.Sequential(layers_dict)

		print "Build complete."
		print

	def apply(self, x):
		last_layer_inputs = x
		for i, m in enumerate(self.model):
			if m["model_type"] in ["mlp", "logistic", "advantage"] and last_layer_inputs.ndim > 2:
				last_layer_inputs = last_layer_inputs.flatten(2)
			last_layer_inputs = self.layers[i](last_layer_inputs)
		return last_layer_inputs

	def save_params(self):
		return [i.get_value() for i in self.params]

	def load_params(self, values):
		print "LOADING NNET..",
		for p, value in zip(self.params, values):
			p.set_value(value.astype("float32"))
		print "LOADED"


if __name__ == "__main__":
	pass
