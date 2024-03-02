#!/bin/bash

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
SRC_ROOT="$(dirname $SCRIPTPATH)"

cd "${SRC_ROOT}"
python3.10 -m src.bp_simunek.scripts.sample_script_tinyda