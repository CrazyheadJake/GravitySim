# GratiySim
A GPU based gravity simulator that uses Euler's method and parallel processing to rapidly simulate the motion of celestial bodies.

## Features
- Uses CUDA to multithread acceleration calculations for planets

## Requirements
- C++20
- [CMake](https://cmake.org/) (≥ 3.16 recommended)
- Nvidia graphics card with CUDA support
- (Optional) [GoogleTest](https://github.com/google/googletest) for running the unit tests

## Installation
```bash
git clone https://github.com/CrazyheadJake/GravitySim.git
cd GravitySim
mkdir build && cd build
cmake ..
make
```

## Usage
```bash
./Debug/GravitySim.exe
```
or
```bash
./Release/GravitySim.exe
```

## License
This project is licensed under the [Apache 2.0 License](LICENSE).
