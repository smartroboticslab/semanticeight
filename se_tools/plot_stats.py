#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
# SPDX-License-Identifier: BSD-3-Clause

# TODO allow plotting data as it arrives through a pipe so we can do
#   supereight -i ... | ./plot_stats.py
# and see live stats

import argparse
import fileinput
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np



class SEStats:
    """Statistics for a single run of supereight"""
    def __init__(self, filename: str="") -> None:
        self.filename = filename
        self.frames = []
        self.acquisition_time = []
        self.preprocessing_time = []
        self.tracking_time = []
        self.integration_time = []
        self.raycasting_time = []
        self.rendering_time = []
        self.computation_time = []
        self.total_time = []
        self.ram_usage = []
        self.x = []
        self.y = []
        self.z = []
        self.tracked = []
        self.integrated = []

    def append_line(self, line: str) -> None:
        # Ignore lines not starting with whitespace or digits
        if not (line[0].isspace() or line[0].isdecimal()):
            return
        # Split the line at whitespace
        columns = line.replace("\t", " ").split()
        # Ignore line with the wrong number of columns
        if len(columns) != 15:
            return
        # Append the data
        self.frames.append(int(columns[0]))
        self.acquisition_time.append(float(columns[1]))
        self.preprocessing_time.append(float(columns[2]))
        self.tracking_time.append(float(columns[3]))
        self.integration_time.append(float(columns[4]))
        self.raycasting_time.append(float(columns[5]))
        self.rendering_time.append(float(columns[6]))
        self.computation_time.append(float(columns[7]))
        self.total_time.append(float(columns[8]))
        self.ram_usage.append(float(columns[9]))
        self.x.append(float(columns[10]))
        self.y.append(float(columns[11]))
        self.z.append(float(columns[12]))
        self.tracked.append(bool(columns[13]))
        self.integrated.append(bool(columns[14]))

    def last_frame(self) -> 'SEStats':
        # Create an SEStats object containing the data of the last frame
        d = SEStats(self.filename)
        if d.frames:
            d.frames.append(self.frames[-1])
            d.acquisition_time.append(self.acquisition_time[-1])
            d.preprocessing_time.append(self.preprocessing_time[-1])
            d.tracking_time.append(self.tracking_time[-1])
            d.integration_time.append(self.integration_time[-1])
            d.raycasting_time.append(self.raycasting_time[-1])
            d.rendering_time.append(self.rendering_time[-1])
            d.computation_time.append(self.computation_time[-1])
            d.total_time.append(self.total_time[-1])
            d.ram_usage.append(self.ram_usage[-1])
            d.x.append(self.x[-1])
            d.y.append(self.y[-1])
            d.z.append(self.z[-1])
            d.tracked.append(self.tracked[-1])
            d.integrated.append(self.integrated[-1])
        return d

    def plot(self, axes=None) -> None:
        # Create a new subplot only if an existing one wasn't provided.
        if axes is None:
            _, axes = plt.subplots(2, 1)

        # Compute the basename of the file the data came from.
        file_basename = os.path.basename(self.filename)

        timing_labels=['Acquisition', 'Preprocessing', 'Tracking',
                'Integration', 'Raycasting', 'Rendering']
        axes[0].stackplot(self.frames,
                [1000 * x for x in self.acquisition_time],
                [1000 * x for x in self.preprocessing_time],
                [1000 * x for x in self.tracking_time],
                [1000 * x for x in self.integration_time],
                [1000 * x for x in self.raycasting_time],
                [1000 * x for x in self.rendering_time],
                labels=timing_labels)
        axes[0].legend(loc='upper left')
        axes[0].set_xlabel('Frame')
        axes[0].set_ylabel('Time (ms)')
        if file_basename:
            axes[0].set_title(file_basename + '\n\nComputation time')
        else:
            axes[0].set_title('Computation time')

        ram_colour = 'tab:blue'
        axes[1].stackplot(self.frames, self.ram_usage, color=ram_colour)
        axes[1].set_xlabel('Frame')
        axes[1].set_ylabel('RAM (MiB)', color=ram_colour)
        axes[1].set_title('Resource usage')

        time_colour = 'tab:green'
        axes = np.append(axes, axes[1].twinx())
        axes[2].plot(self.frames, [1000 * x for x in self.total_time], color=time_colour)
        axes[2].set_ylabel('Computation time (ms)', color=time_colour)



def parse_arguments():
    parser = argparse.ArgumentParser(
            description=('Plot the statistics shown when running supereight '
                'with the --no-gui option. Run supereight and redirect the '
                'output to a file using '
                'se-denseslam-tsdf-pinholecamera-main --no-gui ... > log.txt '
                'and plot the data using '
                './se_tools/plot_stats.py log.txt'))
    parser.add_argument('files', nargs='*', metavar='FILE', default=['-'],
            help=('A text file containing the output of supereight. With no '
                'FILE or when FILE is -, read standard input. If multiple '
                'files are provided the results for all files are shown in a '
                'single window.'))
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    try:
        args = parse_arguments()

        # Read the data
        data = []
        for file in args.files:
            data.append(SEStats(file))
            for line in fileinput.input(file):
                data[-1].append_line(line)

        # Plot the data
        fig, axes = plt.subplots(2, len(data), constrained_layout=True)
        for i, d in enumerate(data):
            data_axes = [axes[0][i], axes[1][i]]
            d.plot(data_axes)
        with warnings.catch_warnings():
            # Hide warnings due to the multiline title in SEStats.plot()
            warnings.simplefilter('ignore', category=UserWarning)
            plt.show()
    except KeyboardInterrupt:
        pass

