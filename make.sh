#!/usr/bin/env bash
set -ex
cd malis
python setup.py build_ext --inplace
printf "BUILD COMPLETE\n"
cd ..
