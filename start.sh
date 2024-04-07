#!/bin/bash

# addgroup root pulse-access
sleep 5
/usr/bin/killall pulseaudio
sleep 3
/usr/bin/pulseaudio --daemonize=true --system --disallow-exit --disallow-module-loading
cd /opt/NodeApp
source .venv/bin/activate
export CHAT_OPENAI_KEY=''
python ./main.py
