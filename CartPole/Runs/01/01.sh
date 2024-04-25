#!/bin/bash

# MISC
gpu=0
save=False
load=0 

# DATA
Nd=256

# NETWORK
width=100
depth=2
bias=True
zero=False

# PHYSICS
T=1.0
Nt=100

# LOSS
lm=FINAL

# OPTIMIZATION
opt=ADAM
lr=0.001
cnum=1.0
bs=8
ep=10



script_name=$(basename "$0")
script_name=${script_name%.*}
folder_name=$(realpath "$0")
folder_name=${folder_name%/*}
folder_name=${folder_name##*/}
echo $folder_name


for NC in 1 2 3 4
do
    for gfm in F P C S
    do
        for cmod in VALUE NORM NONE
        do
            python Simulations/${folder_name}/start.py --gpu ${gpu}  --save ${save} --load ${load} --Nd ${Nd}  --width ${width} --depth ${depth} --bias ${bias} --zero ${zero} --T ${T}  --Nt ${Nt} --NC ${NC} --lm ${lm} --gfm ${gfm} --opt ${opt} --lr ${lr} --cmod ${cmod} --cnum ${cnum} --bs ${bs} --ep ${ep} --folder_name ${folder_name} --script_name ${script_name}
        done
    done
done
