from _benchmark_base import *
from datasets import *

FULL_BENCHMARK = BenchmarkBase("full-benchmark")
FULL_BENCHMARK.setup_from_yaml("./config/benchmark/benchmark_base_config.yaml")
FULL_BENCHMARK.add_datasets([EXAMPLE_DATASET_1, EXAMPLE_DATASET_2])