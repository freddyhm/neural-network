from network import Network
from calculator import Calculator
from utility import Utility
from logger import Logger
import random

class System:
    def __init__(self):
        self._utility = None
        self._network = None
        self._calculator = None
        self._logger = None
        self._performance_output = ""
        self._data = None
        self._data_learning = None
        self._data_vc = None
        self._data_gen = None
        self._data_test = None
        self._data_size = 0
        self._success_count = 0
        self._data_processed_count =  0

    def build(self):
        self.load_helpers()
        self._network = Network(self._config)
        self._logger = Logger(self._network, self._config["saved_network_path"][0])
        self._calculator = Calculator(self._network, self._config["activation_function"], self._logger)

    def load_helpers(self):
        self._utility = Utility("config.txt")
        self._config = self._utility.load_config()
        

    def load_data(self, data_type, mode):

        if data_type == None:
            if mode == "learn":
                data_path = "training_data_path"
            elif mode == "vc":
                data_path = "vc_data_path"
            elif mode == "generalization":
                data_path = "gen_data_path"

            data_type = self._utility.load_data(self._config[data_path][0], mode)
            
        return data_type


    def run(self, mode):

        if mode == "learn":
            self._data_learning = self.load_data(self._data_learning, mode)
            self._data = self._data_learning
        elif mode == "vc":
            self._data_vc = self.load_data(self._data_vc, mode)
            self._data = self._data_vc
        elif mode == "generalization":
            self._data_gen = self.load_data(self._data_gen, mode)
            self._data = self._data_gen
        
        self._data_size = len(self._data)

        epoch = self._config["epoch"][0] if mode == "learn" else 1

        k = 0
        while k < epoch:

            # reset so we get a new performance rate for every epoch
            self._success_count = 0
            self._data_processed_count = 0
            self._performance_output = ""

            i = 0
            while i < self._data_size:
                
                self.set_io()

                if mode == "learn":
                    self.train()
                elif mode == "vc" or mode == "generalization":
                    self.get_output()
                
                self._data_processed_count += 1
                performance_rate = self._utility.get_performance_rate(self._success_count, self._data_processed_count)
                
                print("Iteration:" + str(i))
                print(str(performance_rate))

                i += 1

            print(self._logger.log_performance(mode, str(k + 1), self._success_count, self._data_processed_count, performance_rate))
        
            k += 1
        
        print("Sauvegarde des donnees utilisee...")
        self._logger.log_learning_db(self._config["saved_learning_db_path"][0], self._data)
            
    def train(self):
        if self.get_output() == False:
            self.teach()

    def set_io(self):
        # get random input and output 
        rand_num = random.randint(0, (self._data_size - 1))
        self._network.inputs = self._data[rand_num]["input"]
        self._real_expected_output = self._data[rand_num]["output"]
        self._network.outputs = self._utility.code_output(self._real_expected_output)

    def teach(self):
        self._calculator.compute_error()
        self._calculator.compute_new_weight()

    def get_output(self):
        self._calculator.compute_activation()
        self._calculator.compute_max_neuron()

        return self.are_outputs_equal()

    def are_outputs_equal(self):
        
        # code our network's output
        max_neuron_pos = int(self._network.max_neuron.neuron_id)
        result_network_output = [0] * len(self._network.layers[-1].neurons)
        result_network_output[max_neuron_pos - 1] = 1

        # keep track of our successes so we can return a performance rate 
        if self._network.outputs == result_network_output:
            self._success_count += 1
            return True
        else:
            return False 

    def save(self):
        self._logger.log_network(self._network)

    def save_performance(self):
        self._logger.write_performance()
    
    @property
    def performance_output(self):
        return self._performance_output