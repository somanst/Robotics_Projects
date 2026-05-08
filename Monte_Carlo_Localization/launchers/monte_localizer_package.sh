#!/bin/bash
source /environment.sh
dt-launchfile-init
rosrun monte_localizer_package monte_localizer_node.py
dt-launchfile-join
