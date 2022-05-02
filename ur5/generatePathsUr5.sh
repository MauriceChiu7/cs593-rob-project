#!/bin/bash

# run the ur5.py in a loop

for i in {73..120}
do
    # echo $i
    python3 ur5.py $i
done
echo All done