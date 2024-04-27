#!/bin/bash

# addgroup root pulse-access

while true
do
    DATA=$(ps -ef | grep '/usr/bin/pulseaudio --daemonize=true' | grep -v grep)
    echo "data: $DATA"
    if [ "$DATA" ]; then
        echo "wait and check again ..."
        sleep 2
    else
        echo 'no pulseaudio daemon, start it ...'
        break
    fi
done
sleep 3

/usr/bin/pulseaudio --daemonize=true --system --disallow-exit --disallow-module-loading
sleep 1
/usr/bin/pactl list sinks | grep -E 'Name:' | while read -r line; do
    sink_name=$(echo "$line" | cut -f2 -d' ')
    /usr/bin/pactl get-sink-mute $sink_name
    /usr/bin/pactl set-sink-mute $sink_name false
    /usr/bin/pactl get-sink-mute $sink_name
done

sleep 1
# the input source depends on the machine
/usr/bin/amixer -c 0 cset name='Input Source', 'Rear Mic'
/usr/bin/amixer -c 0 cset name='Rear Mic Boost Volume' 0
