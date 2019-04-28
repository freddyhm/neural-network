import re
import random
from tqdm import tqdm 

class Utility:
    def __init__(self, path):
        self._config = {}
        self._data = []
        self._path = path

    def load_data(self, data_path, mode):
        self._data = []
        
        # must tell progress bar how many lines to load
        print("Chargement des donees")
        file_size = 0
        if mode == "learn":
            file_size = 1340
        elif mode == "vc":
            file_size = 120
        elif mode == "generalization":
            file_size = 780

        with open(data_path) as filestream:
            for line in tqdm(filestream, total=file_size, unit="data"):  
                line_data = line.split(": ")        
                line_element_data = line_data[1].split(" ")
                # keep static params in new list
                line_extracted_data = [] 
                start = 0
                end = 26 if self._config["param_type"][0] == "all" else 12 
                frame_size = 26
                num_of_frames = self._config["frames"][0]
                frame_total = frame_size * num_of_frames
                # extract params for every frame
                for idx in range(0, frame_total, frame_size):
                    line_extracted_data += line_element_data[start:end]
                    start += frame_size
                    end += frame_size
       
                key_name = line_data[0]
                self.format_data(line_extracted_data)
                self._data.append({"output": key_name, "input": line_extracted_data})
        return self._data
        
    def load_config(self):
        # read file and extract data
        with open(self._path) as filestream:
            for line in filestream:
                key_name = ""
                line_data = line.split(": ")
                line_element_data = line_data[1].split(" ")
                key_name = self.get_key_name(line_data[0])

                # add selected properties to our config structure 
                if key_name != "undefined":
                    self.format_data(line_element_data)

                    # thresholds are formatted in a different way 
                    # to set them for neurons
                    if "threshold" in key_name:
                        self.format_threshold_data(line_element_data, key_name)
                    elif key_name.isdigit() or key_name == 'o':  
                        self.format_code_output_data(line_element_data, key_name)
                    else:
                        self._config[key_name] = line_element_data
        return self._config

    def format_data(self, line_element_data):
        for idx, element in enumerate(line_element_data) :
            if element != "" and element != " " and element != "\n":
                # remove all non-digits, non-alpha characters
                element = element.rstrip()
                element = element.strip()

                # convert strings into float and int when necessary
                if re.search("\d+\.\d+", element): 
                    element = float(line_element_data[idx])
                elif re.search("\d", element):
                    element = int(line_element_data[idx])    
                 
                line_element_data[idx] = element
            else:
                del line_element_data[idx]

    def format_code_output_data(self, line_element_data, key_name):
        if "code" not in self._config.keys():
            self._config["code"] = {}
                        
        self._config["code"][key_name] = line_element_data

    def format_threshold_data(self, line_element_data, key_name):
        if "thresholds" not in self._config.keys():
            self._config["thresholds"] = {"threshold_hidden":[], "threshold_output":[]}

        neurons = []
        for threshold in line_element_data:
            neurons.append(threshold)
            
        self._config["thresholds"][key_name].append(neurons)
    
    # map parameter names in config file to keys we can use
    def get_key_name(self, line_param_name):
        if line_param_name == "Number of input(s) [frames x static/all]":
            key_name = "num_inputs"
        elif line_param_name == "Eta":
            key_name = "eta"
        elif line_param_name == "Epoch":
            key_name = "epoch"
        elif line_param_name == "Number of frames to extract":
            key_name = "frames"
        elif line_param_name == "Load static parameters or all parameters [static = 12][all = 26]":
            key_name = "param_type"
        elif line_param_name == "Activation Function":
            key_name = "activation_function"
        elif line_param_name == "Training data path":
            key_name = "training_data_path"
        elif line_param_name == "Initial weight interval":
            key_name = "init_weight_interval"
        elif line_param_name == "VC data path":
            key_name = "vc_data_path"
        elif line_param_name == "Generalization data path":
            key_name = "gen_data_path"
        elif line_param_name == "Number of hidden layer(s)":
            key_name = "num_hidden_layers"
        elif line_param_name == "Number of neuron(s) per layer":
            key_name = "num_neurons_layer"
        elif line_param_name == "Number of neuron(s) in output layer":
            key_name = "num_neurons_output"    
        elif line_param_name == "Saved network snapshot file":
            key_name = "saved_network_path"
        elif line_param_name == "Saved database used for learning":
            key_name = "saved_learning_db_path"
        elif "Threshold for hidden" in line_param_name:
            key_name = "threshold_hidden"
        elif "Threshold for output" in line_param_name:
            key_name = "threshold_output"
        elif re.search("\d", line_param_name) or line_param_name == 'o':
            key_name = line_param_name 
        else:
            key_name = "undefined"
        return key_name
    
    def code_output(self, expected_output):
       return self._config["code"][expected_output]
       
    def get_performance_rate(self, success_num, total_data):
        return (success_num / total_data) * 100