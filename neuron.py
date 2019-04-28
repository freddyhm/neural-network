class Neuron:
    def __init__(self, neuron_id, parent_layer = 0, threshold = 0):
        self._neuron_id = neuron_id
        self._threshold = threshold
        self._parent_layer = parent_layer
        self._error = 0
        self._i = 0
        self._a = 0
        self._branches_in = []
        self._branches_out = []
        
    def add_branch(self, branch, type_branch):
        if type_branch == "in" :
            self._branches_in.append(branch)
        elif type_branch == "out":
            self._branches_out.append(branch)

    @property
    def branches_in(self):
        return self._branches_in
    
    @property
    def branches_out(self):
        return self._branches_out

    @property
    def neuron_id(self):
        return self._neuron_id

    @property
    def i(self):
        return self._i

    @i.setter
    def i(self, i):
        self._i = i

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        self._a = a

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        self._threshold = threshold

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, error):
        self._error = error

    @property
    def parent_layer(self):
        return self._parent_layer

    @parent_layer.setter
    def parent_layer(self, parent_layer):
        self._parent_layer = parent_layer