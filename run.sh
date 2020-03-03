#!/bin/bash
rm karman.bin
rm output
sbatch submit.sbatch

FILE=/home/dcs/csutbb/cfd/output
while [ ! -f "$FILE" ]
do
    sleep 0.5
done
tail -f "$FILE"
