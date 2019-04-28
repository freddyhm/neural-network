class Layer:
    def __init__(self, layer_id):
        self._layer_id = layer_id
        self._neurons = []

    def add_neuron(self, neuron):
        self._neurons.append(neuron)

    @property
    def neurons(self):
        return self._neurons

    @property 
    def layer_id(self):
        return self._layer_id
