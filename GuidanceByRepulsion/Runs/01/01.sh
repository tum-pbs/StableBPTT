#!/bin/bash



# MISC
gpu=0
save=True
load=0

# DATA
Nd=256

# NETWORK
width=100
depth=2
bias=True
zero=False

# PHYSICS
T=4
Nt=60
NC=2-4
tarx=0.0
tary=0.0

# LOSS
lm=CONTINUOUS

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

for rc in 1.0 0.1 0.01 0.001
do
    for cmod in VALUE NORM NONE
    do
        for gf in F P C S
        do
            python Simulations/${folder_name}/start.py --gpu ${gpu} --save ${save} --load ${load} --Nd ${Nd} --width ${width} --depth ${depth} --bias ${bias} --zero ${zero} --T ${T}  --Nt ${Nt} --NC ${NC} --tarx ${tarx} --tary ${tary} --lm ${lm} --rc ${rc} --gf ${gf} --opt ${opt} --lr ${lr} --cmod ${cmod} --cnum ${cnum} --bs ${bs} --ep ${ep} --folder_name ${folder_name} --script_name ${script_name}
        done
    done
done
