import math

class Calculator:
    def __init__(self, network, output_ftn, logger):
        self._network = network
        self._output_ftn = output_ftn[0]
        self._logger = logger

    # activation function
    def activation(self, element):

        ftn = 0
        if self._output_ftn == "sigmoid":
            ftn = 1 / (1 + math.exp(-element))
        elif self._output_ftn == "sine":
            ftn = math.sin(element)

        return ftn 

    def derivative_activation(self, element):

        ftn = 0
        if self._output_ftn == "sigmoid":
            ftn = element * (1 - element)
        elif self._output_ftn == "sine":
             ftn = math.cos(element)

        return ftn

    # phase 1: compute i and a
    def compute_activation(self):  
        for layer in self._network.layers:
            for neuron in layer.neurons:
               # used to access past source position ( source = neurons if layer = 2 or source = inputs if layer = 1) to compute "a"
                src_pos = 0
                sum_weight = 0
                for branch in neuron.branches_in:
                    # ENTRANCE layer - inputs x branch weights
                    if neuron.parent_layer.layer_id == 1:
                        sum_weight += branch.weight * self._network.inputs[src_pos]
                    else:
                        # HIDDEN layer - needs to access the last layer's neuron's "a"
                        last_layer = neuron.parent_layer.layer_id - 2
                        last_neuron_a = self._network.layers[last_layer].neurons[src_pos].a
                        sum_weight += branch.weight * last_neuron_a
                    src_pos += 1

                # add threshold and compute our a with activation function
                neuron.i = sum_weight + neuron.threshold
                neuron.a = self.activation(neuron.i)

    def compute_max_neuron(self):
        last_layer_neurons = self._network.layers[-1].neurons
        self._network.max_neuron = max(last_layer_neurons, key=lambda neuron: neuron.a)

    # phase 2: compute error
    def compute_error(self):  
        for layer in reversed(self._network.layers):
            output_pos = 0
            for neuron in layer.neurons:
                
                # for each neuron in EXIT layer 
                # 1 - multipy derivative activation with (outputs - a)
                if neuron.parent_layer.layer_id == len(self._network.layers):
                    
                    # output_pos: goes through output array so the first output can be used  
                    # for calculations with the first neuron in the last layer, and so on. 
                    # Works since an output matches a neuron in the last layer

                    # send i or a to derivative depending on user selected function
                    element = neuron.i if self._output_ftn == "sine" else neuron.a
                    neuron.error = self.derivative_activation(element) * (self._network.outputs[output_pos] - neuron.a)
                    output_pos += 1
                else:
                    # for each neuron in HIDDEN layer
                    # 1- sum all outgoing branch weights multiplied by error of neurons in next layer 
                    # 2- multiply sum with derivative activation function 
                    sum_error_weight = 0
                    next_neuron_pos = 0
                    next_layer = self._network.layers[neuron.parent_layer.layer_id]

                    for branches_out in neuron.branches_out:
                        next_neuron = next_layer.neurons[next_neuron_pos]
                        next_neuron_error = next_neuron.error
                        sum_error_weight += (branches_out.weight * next_neuron_error)
                        next_neuron_pos += 1

                    # send i or a to derivative depending on user selected function
                    element = neuron.i if self._output_ftn == "sine" else neuron.a
                    neuron.error = self.derivative_activation(element) * sum_error_weight
       
    # phase 3: compute new weight
    def compute_new_weight(self):
        for layer in self._network.layers:
            for neuron in layer.neurons:
                # used to access past source position (source = neurons if layer = 2 or inputs if layer = 1) to compute "a"
                src_pos = 0
                for branch in neuron.branches_in:
                    # ENTRANCE layer
                    if neuron.parent_layer.layer_id == 1:
                        diff_weight = self._network.eta * neuron.error * self._network.inputs[src_pos]
                    else:
                        # HIDDEN and EXIT layers - need to access the last layer's neuron's "a"
                        last_layer = neuron.parent_layer.layer_id - 2
                        last_neuron_a = self._network.layers[last_layer].neurons[src_pos].a
                        diff_weight = self._network.eta * last_neuron_a * neuron.error
                    
                    src_pos += 1
                    # save delta weight and compute new weight
                    branch.delta_weight = diff_weight
                    branch.weight += diff_weight