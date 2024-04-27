#!/bin/bash

sleep 15
cd /opt/NodeApp
source .venv/bin/activate
export CHAT_OPENAI_KEY=''
python ./main.py
