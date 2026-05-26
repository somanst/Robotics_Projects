#!/bin/bash
source /environment.sh
dt-launchfile-init
rosrun dwa_package dwa_package.py "$@"
dt-launchfile-join
