import os
import subprocess

class runCommand:
    def __init__(self):
        self.executable = None
        self.args       = None

    def toString(self):
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
            f.write('`' + cmd.toString().replace("result.txt", "result-manual-run.txt") + '`')

        try:
            # Doesn't work without shell=True??
            subprocess.check_call(
                cmd.toString(), shell=True)
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
        args.extend(['--benchmark=' + self.config_yaml_path.replace("config.yaml", "result.txt")])
        executable = [os.path.join(os.path.abspath(self.bin_path), 'se-denseslam-' +
                                   self.voxel_impl + '-' + self.sensor_type + '-main')]
        run_command = runCommand()
        run_command.executable = executable
        run_command.args       = args
        return run_command
