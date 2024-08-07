#!/bin/bash

switch=4

for i in {0..4}
do
    beta=$(echo "1.0 + $i*0.25" | bc -l)
    echo $beta
    ./tes.out $switch $beta >> results.dat &
done
