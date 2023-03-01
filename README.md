# semanticeight

This repository contains the core library of the object-level mapping and
exploration framework presented in the ICRA 2023 paper “Finding Things in the
Unknown: Semantic Object-Centric Exploration with an MAV”. The main repository
is [here](https://github.com/smartroboticslab/semantic-exploration-icra-2023).

The 3D mapping code is based on
[supereight](https://bitbucket.org/smartroboticslab/supereight-public).


## Dependencies

* GCC 7+ or clang 6+ (for C++ 17 features and OpenMP)
* CMake 3.10+
* Eigen 3
* OpenCV 3+
* yaml-cpp 5.2+
* OctoMap (optional, for OctoMap output)
* Doxygen (optional, for the documentation)

To install the dependencies on Ubuntu 20.04 run

``` sh
sudo apt-get --yes install git g++ cmake libeigen3-dev libopencv-dev libyaml-cpp-dev liboctomap-dev doxygen
```

## Building

To clone this repository and all submodules run

``` sh
git clone --recurse-submodules https://github.com/smartroboticslab/semanticeight.git
```

or if you already cloned without `--recurse-submodules` run

``` sh
git submodule update --init --recursive
```

Then build the project using CMake

``` sh
mkdir -p build/release
cd build/release
cmake -DCMAKE_BUILD_TYPE=Release ../..
cmake --build .
```

To generate the documentation in HTML format run

``` sh
doxygen
```

and open `doc/html/index.html` with a web browser.


## Usage

The recommended way to use semanticeight in your project is to add it as a git
submodule and to use the `add_subdirectory()` CMake command to build it.

``` cmake
option(SE_BUILD_TESTS "Build the supereight unit tests" OFF)
option(SE_BUILD_GLUT_GUI "Build the OpenGL-based GUI" OFF)
add_subdirectory(semanticeight)

# Link some-target with the semanticeight library.
target_link_libraries(some-target SE::DenseSLAMPinholeCamera)
```


## License

Copyright 2018-2019 Emanuele Vespa</br>
Copyright 2019-2023 Smart Robotics Lab, Imperial College London, Technical University of Munich</br>
Copyright 2019-2022 Nils Funk</br>
Copyright 2019-2023 Sotiris Papatheodorou</br>

semanticeight is distributed under the
[BSD 3-clause license](LICENSES/BSD-3-Clause.txt). Some parts of the code are
distributed under the [MIT license](LICENSES/MIT.txt). See the individual file
headers for which license applies.
