#!/bin/bash

# MISC
gpu=6
save=False
load=0  
weighting=1-1 #1-1,3-1,7-1,15-1

# DATA
Nd=256

# NETWORK
width=60
depth=2
bias=True
zero=False

# PHYSICS
Nx=32
dt=0.0625 
Nt=128

# LOSS
LT=CONTINUOUS

# OPTIMIZATION
opt=ADAM
lr=0.001
cnum=1.0
bs=8
ep=1000



script_name=$(basename "$0")
script_name=${script_name%.*}
folder_name=$(realpath "$0")
folder_name=${folder_name%/*}
folder_name=${folder_name##*/}
echo $folder_name


for TS in 3 4 5
do
    for cmod in VALUE NORM NONE
    do
        for gfm in F P C S
        do
            python Simulations/${folder_name}/qmc_start.py --gpu ${gpu}  --save ${save} --load ${load} --weighting ${weighting} --Nd ${Nd} --width ${width} --depth ${depth} --bias ${bias} --zero ${zero} --Nx ${Nx} --dt ${dt}  --Nt ${Nt} --TS ${TS} --LT ${LT} --gfm ${gfm} --opt ${opt} --lr ${lr} --cmod ${cmod} --cnum ${cnum} --bs ${bs} --ep ${ep} --folder_name ${folder_name} --script_name ${script_name}
        done
    done
done
