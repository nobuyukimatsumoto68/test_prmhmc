#!/bin/bash

switch=4

for i in {0..5}
do
    beta=$(echo "0.75 + $i*0.25" | bc -l)
    echo $beta
    ./tes.out $switch $beta >> results.dat &
done
