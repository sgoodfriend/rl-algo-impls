#!/bin/bash

frame_index=${1:-0}

for i in *.mp4;  do 
    name=`echo $i | cut -d'.' -f1`
    ffmpeg -i "$i" -vf "select=eq(n\,$frame_index)" -vframes 1 "${name}.png"
done