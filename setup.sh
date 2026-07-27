#!/bin/sh
# # to avoid linking shared
# export CXXFLAGS="$CXXFLAGS -fPIC"


external_libs_full_path=$(readlink -f ./external_libs)
rm -rf ${external_libs_full_path}/argparse
rm -rf ${external_libs_full_path}/inih
rm -rf ${external_libs_full_path}/pcg-cpp
rm -rf ${external_libs_full_path}/eigen-3.4.0
rm -rf ${external_libs_full_path}/include
rm -rf ${external_libs_full_path}/share
rm -rf ${external_libs_full_path}/eigen-3.4.0.tar.gz
mkdir ${external_libs_full_path}
cd ${external_libs_full_path}

wget "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"
tar -xzvf "./eigen-3.4.0.tar.gz"
cd eigen-3.4.0
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${external_libs_full_path}
cmake --build .
cmake --install .


cd ${external_libs_full_path}
git clone "https://github.com/imneme/pcg-cpp.git"
git clone "https://github.com/benhoyt/inih.git"
git clone "https://github.com/p-ranav/argparse.git"
