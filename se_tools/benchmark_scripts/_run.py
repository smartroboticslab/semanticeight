import os
import subprocess

class runCommand:
    def __init__(self):
        self.executable          = None
        self.args                = None
        self.base_benchmark_file = None

    def benchmark(self):
        benchmark_arg = ['--benchmark=' + self.base_benchmark_file + 'result.txt']
        return ' '.join(self.executable + self.args + benchmark_arg)

    def manualBenchmark(self):
        benchmark_arg = ['--benchmark=' + self.base_benchmark_file + 'result_manual_run.txt']
        return ' '.join(self.executable + self.args + benchmark_arg)

    def withoutBenchmark(self):
        return ' '.join(self.executable + self.args)

class SLAMAlgorithm:
    """ A general SLAM algorithm evaluator.
    """

    def __init__(self, bin_path):
        self.bin_path = bin_path

    def run(self, test_case):
        self._setup_from_test_case(test_case)
        self._run_internal()


    def _run_internal(self):
        """ Generate the run command and run
        """
        cmd = self._generate_run_command()

        cmd_filename = os.path.splitext(self.config_yaml_path)[0].replace("config", "command.md")

        with open(cmd_filename, 'w') as f:
            f.write('`' + cmd.manualBenchmark() + '`\n\n'
                    '`' + cmd.withoutBenchmark() + '`')

        try:
            # Doesn't work without shell=True??
            subprocess.check_call(
                cmd.benchmark(), shell=True)
        except Exception:
            pass
            #self.failed = True


class KinectFusion(SLAMAlgorithm):

    def __init__(self, bin_path):
        SLAMAlgorithm.__init__(self, bin_path)

        self.sensor_type      = None
        self.voxel_impl       = None
        self.config_yaml_path = None

    def _setup_from_test_case(self, test_case):
        self.voxel_impl       = test_case.voxel_impl
        self.sensor_type      = test_case.sensor_type
        self.config_yaml_path = test_case.config_yaml_path

    def _generate_run_command(self):
        args = []
        args.extend(['--yaml-file', str(self.config_yaml_path)])
        base_benchmark_file = (self.config_yaml_path.replace("config", "")).replace(".yaml", "")
        executable = [os.path.join(os.path.abspath(self.bin_path), 'se-denseslam-' +
                                   self.voxel_impl + '-' + self.sensor_type + '-main')]
        run_command = runCommand()
        run_command.executable          = executable
        run_command.args                = args
        run_command.base_benchmark_file = base_benchmark_file
        return run_command
