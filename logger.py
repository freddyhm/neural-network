import json
from tqdm import tqdm 
class Logger:

    def __init__(self, network, path):
        self._path = path
        self._network = network
        self._log_output = ""
        self._performance_output = ""

    def reset(self):
        self._log_output = ""


    def log_network(self, network): 
        self._log_output += self._performance_output
        for layer in tqdm(network.layers):
            self.log_layer(layer.layer_id)
            for neuron in tqdm(layer.neurons):
                self.log_neuron(neuron)
                for branch in tqdm(neuron.branches_in):
                    self.log_branch(branch)
            
        self.write()

    def log_layer(self, layer_id):
        self._log_output +="------------------------------------------------------\n"
        self._log_output += "Couche: " + str(layer_id) + "\n"
        self._log_output +="------------------------------------------------------\n"

    def log_neuron(self, neuron):
        self._log_output +="----------------------------------\n"
        self._log_output += "Neurone id: " + neuron.neuron_id + "\n"
        self._log_output +="----------------------------------\n"
        self._log_output += "Activation (i): " + str(neuron.i) + "\n"
        self._log_output += "Sortie d'activation (a): " + str(neuron.a) + "\n"
        self._log_output += "Seuil: " + str(neuron.threshold) + "\n"
        self._log_output +="----------------------------------\n"

    def log_branch(self, branch):
        self._log_output +="----------------\n"
        self._log_output += "Branche id: " + branch.branch_id + "\n"
        self._log_output +="----------------\n"
        self._log_output += "Delta poid: " + str(branch.delta_weight) + "\n"
        self._log_output += "Poid: " + str(branch.weight) + "\n"
        self._log_output +="-----------------\n"

    def log_performance(self, mode, num_epoch, success, num_learning_data, learning_rate):

        self._performance_output +="---------------------------------------------------\n"
        self._performance_output +="Mode: " + mode + "\n"
        self._performance_output +="---------------------------------------------------\n"
        self._performance_output +="Taux de succes: (nb de succes / nb de donnee) * 100\n"
        self._performance_output +="---------------------------------------------------\n"
        self._performance_output +="Epoque: " + str(num_epoch) + "\n"
        self._performance_output +="Nb de succes: " + str(success) + "\n" 
        self._performance_output +="Nb de donne: " + str(num_learning_data) + "\n" 
        self._performance_output +="Taux: " + str(learning_rate) + "%\n"
        self._performance_output +="---------------------------------------------------\n"
        
        return self._performance_output


    def log_learning_db(self, learning_db_path, data):
        
        log_data_output = ""
        with open(learning_db_path, "w") as out:

            log_data_output +="----------------------------------\n"
            log_data_output += "Base de donnee d'apprentissage\n"
            log_data_output +="----------------------------------\n"

            for element in tqdm(data):
                log_data_output += json.dumps(element) + "\n"

            out.write(log_data_output)

    def write(self):
        with open(self._path, "w") as out:
            out.write(self._log_output)

    def write_performance(self):
        with open("performance-output.txt", "w") as out:
            out.write(self._performance_output)