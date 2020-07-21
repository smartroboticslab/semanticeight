from _config import *
from _test_case import *

import random
import os
import ruamel
from ruamel.yaml import YAML
import warnings

import sys
yaml = YAML()
yaml.preserve_quotes = True
CS = ruamel.yaml.comments.CommentedSeq  # defaults to block style
CM = ruamel.yaml.comments.CommentedMap  # defaults to block style

def valid_sequence(sequence_header):
    if len(sequence_header) not in [2, 3]:
        warnings.warn(
            "A sequence in setup YAML has wrong number of parameters ({} not in [2, 3])".format(len(sequence_header)))
        return False
    return True

def unique_sequence(sequence_header, sequence_names=[]):
    if sequence_header[0] in sequence_names:
        warnings.warn(
            "Sequence name already exists in dataset. Use unique sequence name.")
        return False
    return True

def sequence_from_header(sequence_header, sequence_names=[]):
    if not valid_sequence(sequence_header):
        return None
    if not unique_sequence(sequence_header, sequence_names):
        return None
    if len(sequence_header) is 2:
        return Sequence(sequence_header)
    elif len(sequence_header) is 3:
        return Sequence(sequence_header)

def list_sequences(sequences):
    if isinstance(sequences[0], list):
        return sequences
    else:
        return [sequences]


class Sequence:

    def __init__(self, sequence_header):
        if valid_sequence(sequence_header):
            self.name              = sequence_header[0]
            self.file_path         = sequence_header[1]
            self.ground_truth_file = sequence_header[2] if len(sequence_header) is 3 else None
            self.config            = Config()
        else:
            warnings.warn("Use of invalid sequence header.")

    def setup_from_yaml(self, sequence_config_yaml_path):
        with open(sequence_config_yaml_path) as f:
            sequence_config_yaml = yaml.load(f)
        if 'general' in sequence_config_yaml:
            if 'sequences' in sequence_config_yaml['general']:
                sequence_config_yaml['general']['sequences'] = list_sequences(sequence_config_yaml['general']['sequences'])
                if len(sequence_config_yaml['general']['sequences']) is 1:
                    SEQUENCE = sequence_from_header(sequence_config_yaml['general']['sequences'][0])
                    if SEQUENCE:
                        self = SEQUENCE
                elif len(sequence_config_yaml['general']['sequences']) is 0:
                    warnings.warn(
                        "Sequence setup YAML sequence list is empty [].\n"
                        "The following sequence header is used:\n"
                        "sequence_name      = {}\n"
                        "sequence_path = {}\n"
                        "ground_truth_file  = {}".format(self.name, self.file_path, self.ground_truth_file))
                else:
                    warnings.warn(
                        "Sequence can only be initalised by single sequence header. Use Dataset() instead.")
            else:
                warnings.warn(
                    "Sequence setup YAML misses sequence key.\n"
                    "The following sequence header is used:\n"
                    "sequence_name      = {}\n"
                    "sequence_path = {}\n"
                    "ground_truth_file  = {}".format(self.name, self.file_path, self.ground_truth_file))
        else:
            warnings.warn(
                "Sequence setup YAML misses general collection.\n"
                "The following sequence header is used:\n"
                "sequence_name      = {}\n"
                "sequence_path = {}\n"
                "ground_truth_file  = {}".format(self.name, self.file_path, self.ground_truth_file))
        self.config.setup_from_yaml(sequence_config_yaml)

    def generate_test_cases(self, dataset_dir, dataset_name=""):
        if self.name:
            sequence_dir = os.path.join(dataset_dir, self.name.replace('-', '_'))
            os.mkdir(sequence_dir)
        else:
            sequence_dir = dataset_dir

        sequence_test_cases = []
        
        yaml_keys_list, value_combs_list = self.config.get_yaml_keys_and_value_combs()
        voxel_impl_types = self.config.voxel_impls.get_types()
        sensor_type      = self.config.sensor.type
        for voxel_impl_type, yaml_keys, value_combs in zip(voxel_impl_types, yaml_keys_list, value_combs_list):
            for value_comb in value_combs:
                sequence_name = "-".join([dataset_name, self.name]) if self.name else dataset_name
                yaml_init = """\
                            general:
                                sequence_name:      {}
                                sequence_path: {}
                                ground_truth_file:  {}
                            sensor:
                                type:               {}
                """.format(sequence_name, self.file_path, self.ground_truth_file, sensor_type)
                sequence_config_yaml = ruamel.yaml.round_trip_load(yaml_init)

                for yaml_key, value in zip(yaml_keys, value_comb):
                    collection_key = yaml_key[0]
                    param_key      = yaml_key[1]
                    if not collection_key in sequence_config_yaml:
                        sequence_config_yaml[collection_key] = {}
                    sequence_config_yaml[collection_key][param_key] = flow_style_list(value) if isinstance(value, list) else value

                id = random.randint(0, 1000000000)
                sequence_config_yaml_name = "_".join(
                    [sequence_name.replace('-','_').replace(' ', '_'), delist_value(voxel_impl_type),
                     delist_value(sensor_type), str(id), "config.yaml"])
                sequence_config_yaml_path = os.path.join(sequence_dir, sequence_config_yaml_name)
                with open(sequence_config_yaml_path, 'w') as f:
                    yaml.dump(sequence_config_yaml, f)

                # Create test case
                sequence_test_case = TestCase()
                sequence_test_case.sequence_name    = sequence_name
                sequence_test_case.sensor_type      = delist_value(sensor_type)
                sequence_test_case.voxel_impl       = delist_value(voxel_impl_type)
                if 'dim' in sequence_config_yaml['map'] and 'size' in sequence_config_yaml['map']:
                    map_res = sequence_config_yaml['map']['dim'] / sequence_config_yaml['map']['size']
                else:
                    map_res = 'default'
                sequence_test_case.map_res          =  map_res
                sequence_test_case.downsampling_factor      = sequence_config_yaml['sensor']['downsampling_factor'] \
                    if 'downsampling_factor' in sequence_config_yaml['sensor'] else 'default'
                sequence_test_case.config_yaml_path = os.path.abspath(sequence_config_yaml_path)
                sequence_test_cases.append(sequence_test_case)

        return sequence_test_cases

    def copy_missing(self, config):
        if not self.config:
            self.config = config
        else:
            self.config.copy_missing(config)

    def copy_missing_voxel_impls(self, voxel_impls):
        if not voxel_impls: return
        if not self.config: self.config = Config()
        if not self.config.voxel_impls:
            self.config.voxel_impls = voxel_impls
        else:
            self.config.voxel_impls.copy_missing(voxel_impls)

    def print_all(self):
        print(self.name)
        print(self.file_path)
        if self.ground_truth_file: print(self.ground_truth_file)
        if self.config: self.config.print_all()

    def print_set(self):
        print(self.name)
        print(self.file_path)
        if self.groundtruth_path: print(self.groundtruth_path)
        if self.config: self.config.print_set()

    def print_missing(self):
        print(self.name)
        print(self.file_path)
        if self.groundtruth_path: print(self.groundtruth_path)
        if self.config: self.config.print_missing()
