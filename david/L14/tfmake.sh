#!/bin/bash

# from https://www.pugetsystems.com/labs/hpc/Build-TensorFlow-CPU-with-MKL-and-Anaconda-Python-3-6-using-a-Docker-Container-1133/

#bazel build --config=mkl --copt=-march=native  //tensorflow/tools/pip_package:build_pip_package
bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2  //tensorflow/tools/pip_package:build_pip_package
#bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --config=cuda //tensorflow/tools/pip_package:build_pip_package

#bazel-bin/tensorflow/tools/pip_package/build_pip_package ../tensorflow_pkg
#/home/test/.cache/bazel/_bazel_test/fe19afd8eecf77919754cc4ee7501ed0/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/tensorflow/tools/pip_package/build_pip_package /home/test/.cache/bazel/_bazel_test/fe19afd8eecf77919754cc4ee7501ed0/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/tensorflow/tools/pip_package/build_pip_package

#sudo pip install --upgrade ../tensorflow_pkg/tensorflow-*.whl
#ldd /opt/anaconda3/lib/python3.6/site-packages/tensorflow/libtensorflow_framework.so | grep mkl

#bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.1 --copt=-msse4.2 -k //tensorflow/tools/pip_package:build_pip_package 
# --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"