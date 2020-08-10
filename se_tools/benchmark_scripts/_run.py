# SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020 Nils Funk
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess

class runCommand:
    def __init__(self):
        self.executable    = None
        self.args          = None
        self.base_filename = None
        self.result_dir    = None
        self.output_dir    = None

    def benchmark(self):
        benchmark_arg     = ['--benchmark='          + os.path.join(self.result_dir, self.base_filename + '_result.txt')]
        output_render_arg = ['--output-render-path=' + os.path.join(self.output_dir, self.base_filename + '_render')]
        output_mesh_arg = ['--output-mesh-path=' + os.path.join(self.output_dir, self.base_filename + '_mesh')]
        return ' '.join(self.executable + self.args + benchmark_arg + output_render_arg + output_mesh_arg)

    def manualBenchmark(self):
        benchmark_arg     = ['--benchmark='          + os.path.join(self.result_dir, self.base_filename + '_result_manual_run.txt')]
        output_render_arg = ['--output-render-path=' + os.path.join(self.output_dir, self.base_filename + '_render_manual_run')]
        output_mesh_arg = ['--output-mesh-path=' + os.path.join(self.output_dir, self.base_filename + '_mesh_manual_run')]
        return ' '.join(self.executable + self.args + benchmark_arg + output_render_arg + output_mesh_arg)

    def withoutBenchmark(self): # Force rendering to be enabled
        enable_render_arg = ['--enable-render']
        return ' '.join(self.executable + self.args + enable_render_arg)

class Pipeline:
    """ A general Pipeline evaluator.
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


class Supereight(Pipeline):

    def __init__(self, bin_path):
        Pipeline.__init__(self, bin_path)

        self.sensor_type      = None
        self.voxel_impl       = None
        self.base_filename    = None
        self.output_dir       = None
        self.config_yaml_path = None

    def _setup_from_test_case(self, test_case):
        self.voxel_impl       = test_case.voxel_impl
        self.sensor_type      = test_case.sensor_type
        self.base_filename    = test_case.name
        self.output_dir       = test_case.output_dir
        self.config_yaml_path = test_case.config_yaml_path

    def _generate_run_command(self):
        args = []
        args.extend(['--yaml-file', str(self.config_yaml_path)])
        executable = [os.path.join(os.path.abspath(self.bin_path), 'se-denseslam-' +
                                   self.voxel_impl + '-' + self.sensor_type + '-main')]
        run_command = runCommand()
        run_command.executable    = executable
        run_command.args          = args
        run_command.base_filename = self.base_filename
        run_command.result_dir    = os.path.dirname(self.config_yaml_path)
        run_command.output_dir    = self.output_dir
        return run_command
