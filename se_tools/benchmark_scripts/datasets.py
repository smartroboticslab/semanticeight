# SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020 Nils Funk
# SPDX-License-Identifier: BSD-3-Clause

from _dataset import *

EXAMPLE_DATASET_1 = Dataset("example_dataset_1")
EXAMPLE_DATASET_1.setup_from_yaml("./config/datasets/example_dataset_1_dataset_config.yaml")

EXAMPLE_DATASET_2 = Dataset("example_dataset_2")
EXAMPLE_DATASET_2.setup_from_yaml("./config/datasets/example_dataset_2_dataset_config.yaml")