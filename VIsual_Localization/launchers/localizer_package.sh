#!/bin/bash
source /environment.sh
dt-launchfile-init
rosrun localizer_package localizer_node.py
dt-launchfile-join
