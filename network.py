from branch import Branch
from neuron import Neuron
from layer import Layer
import random
from tqdm import tqdm

class Network:
    def __init__(self, config):
        
        self._num_inputs = config["num_inputs"][0]
        self._weight_interval = config["init_weight_interval"]
        self._eta = config["eta"][0]
        self._num_hidden_layers = config["num_hidden_layers"][0]
        self._num_neurons_layer = config["num_neurons_layer"][0]
        self._num_neurons_output = config["num_neurons_output"][0]
        self._thresholds = config["thresholds"] if "thresholds" in config else [] 
        self._outputs = []
        self._inputs = [] 
        self._layers = []
        self._max_neuron = 0
        self.build_network()
        
    @property
    def max_neuron(self):
        return self._max_neuron

    @max_neuron.setter
    def max_neuron(self, max_neuron):
        self._max_neuron = max_neuron

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        self._outputs = outputs

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        self._inputs = inputs

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, eta):
        self._eta = eta

    @property
    def layers(self):
        return self._layers

    def build_network(self):
        print("Creation des couches:")
        self.build_layers()
        print("Creation des neurones et des branches:")
        self.build_neurons()
        self.build_branches()

    def build_layers(self):
        # hidden + output layer
        total_layers = self._num_hidden_layers + 1
        for idx in tqdm(range(total_layers)):
            layer_id = idx + 1
            new_layer = Layer(layer_id)
            self._layers.append(new_layer)

    def build_neurons(self):

        # add neurons for hidden layer
        for idx in tqdm(range(self._num_hidden_layers)):
            for idx2 in range(self._num_neurons_layer):
                neuron_id = str(idx2 + 1)
                new_neuron = Neuron(neuron_id, self._layers[idx])
                self.set_thresholds(new_neuron, idx, idx2, "threshold_hidden")
                self._layers[idx].add_neuron(new_neuron)

        # add neurons for output layer
        for idx in tqdm(range(self._num_neurons_output)):
            neuron_id = str(idx + 1)
            new_neuron = Neuron(neuron_id, self._layers[-1])
            self.set_thresholds(new_neuron, 0, idx2, "threshold_output")
            self._layers[-1].add_neuron(new_neuron)

    def set_thresholds(self, neuron, layer_pos, neuron_pos, threshold_type):
        # maps threshold config with corresponding neuron
        if threshold_type in self._thresholds: 
            if layer_pos < len(self._thresholds[threshold_type]):
                if neuron_pos < len(self._thresholds[threshold_type][layer_pos]):
                    neuron.threshold =  self._thresholds[threshold_type][layer_pos][neuron_pos]
   
    def build_branches(self):
        for idx, layer in enumerate(self._layers):
            for neuron in layer.neurons:
                # ENTRANCE layer only - create and set branches that connect into the neuron from each input
                if layer.layer_id == 1:
                    for idx2 in range(self._num_inputs):
                        
                        random_weight = random.uniform(self._weight_interval[0],self._weight_interval[1])

                        new_branch_id = str(idx2 + 1) + "" + str(neuron.neuron_id) + "" + str(layer.layer_id)
                        new_branch_in = Branch(new_branch_id, random_weight)
                        neuron.add_branch(new_branch_in, "in")

                # HIDDEN layers only - create and set branches that leave the current neuron and connect into next layer's neuron
                if layer.layer_id != len(self._layers):
                    for next_layer_neuron in self._layers[idx + 1].neurons:
                        
                        random_weight = random.uniform(self._weight_interval[0],self._weight_interval[1])

                        new_branch_id = str(neuron.neuron_id) + "" + str(next_layer_neuron.neuron_id) + "" + str(next_layer_neuron.parent_layer.layer_id)
                        new_branch_out = Branch(new_branch_id, random_weight)
                        neuron.add_branch(new_branch_out, "out")
                        next_layer_neuron.add_branch(new_branch_out, "in")


       