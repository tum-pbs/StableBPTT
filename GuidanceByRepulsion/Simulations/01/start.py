from data import generate_data
from controller import build_fully_connected_network
from simulator import DRIVER_EVADER_SYSTEM
from training import TRAINING_SETUP

import tensorflow as tf
import numpy as np
import argparse,os

#tf.config.run_functions_eagerly(True)

parser = argparse.ArgumentParser(description='CmdLine Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# MISC
parser.add_argument(  '--gpu',  default=0,  type=str, help='Cuda visible devices')
parser.add_argument(  '--save',  default=False,  type=bool, help='Save model?')
parser.add_argument(  '--load',  default=0,  type=str, help='Load model? Path or 0')

# DATA
parser.add_argument(  '--Nd',  default=0,  type=int, help='Number of data points')

# NETWORK
parser.add_argument(  '--width',  default=0,  type=int, help='Network width')
parser.add_argument(  '--depth',  default=0,  type=int, help='Network depth')
parser.add_argument(  '--bias',  default=True,  type=bool, help='Use bias?')
parser.add_argument(  '--zero',  default=False,  type=bool, help='Use zero initialization?')

# PHYSICS
parser.add_argument(  '--T',  default=0,  type=float, help='Total time (physics system)')
parser.add_argument(  '--Nt',  default=0,  type=int, help='Number of solver time steps') 
parser.add_argument(  '--NC',  default=0,  type=str, help='Number of drivers and evaders, Complexity') 
parser.add_argument(  '--tarx',  default=0,  type=float, help='Target x') 
parser.add_argument(  '--tary',  default=0,  type=float, help='Target y') 


# LOSS
parser.add_argument(  '--lm',  default=0,  type=str, help='Loss mode (CONTINUOUS, FINAL)') 
parser.add_argument(  '--rc',  default=0,  type=float, help='Regularization coefficient')


# OPTIMIZATION
parser.add_argument(  '--gf',  default=0,  type=str, help='Training mode (gradient flow): NORMAL (BPTT), STOP (one step gradients), PHY')
parser.add_argument(  '--opt',  default=0,  type=str, help='Optimizer: ADAM')
parser.add_argument(  '--lr',  default=0,  type=float, help='Learning rate')
parser.add_argument(  '--cmod',  default=0,  type=str, help='Clipping mode: NONE, VALUE')
parser.add_argument(  '--cnum',  default=0,  type=float, help='Clipping number')
parser.add_argument(  '--bs',  default=0,  type=int, help='Batch size')
parser.add_argument(  '--ep',  default=0,  type=int, help='Number of epochs')

#PATHS
parser.add_argument(  '--script_name',  default=0,  type=str, help='Script name')
parser.add_argument(  '--folder_name',  default=0,  type=str, help='Folder name')



p = {}
p.update(vars(parser.parse_args()))
dt = p['T']/p['Nt']
p['dt']=dt
n_dr = int(p['NC'][0])
p['n_dr']=n_dr
n_ev = int(p['NC'][-1])
p['n_dr']=n_ev
for i in p.keys():
    print(i,p[i]) 




path = 'Data/'+p['folder_name']+'/'+p['script_name']+'_'
i = 0
pi = str(i).zfill(4)
while(os.path.exists(path+pi)):
    i = i+1
    pi = str(i).zfill(4)
sim_path = path+pi+'/'
results_path = sim_path+'results.txt'
dict_path = sim_path+'params.txt'
dict_np = sim_path+'params.npy'
network_path = sim_path+'network.h5'
os.makedirs(os.path.dirname(sim_path))
with open(dict_path, 'x') as file:
    for i in p.keys():
        print(i.ljust(20,' '),p[i],file=file)
np.save(dict_np,p)



### SIMULATION ###

os.environ["CUDA_VISIBLE_DEVICES"]=p['gpu']
tf.random.set_seed(42)

train_data,test_data = generate_data(p['Nd'],n_dr,n_ev)
dataloader = tf.data.Dataset.from_tensor_slices(train_data).batch(p['bs'])
data = dataloader,train_data,test_data 

if p['load']=='0':
    controller = build_fully_connected_network(p['width'],p['depth'],p['bias'],p['zero'],n_dr,n_ev)
    print('NEW MODEL')
else:
    controller = tf.keras.models.load_model(p['load'])
    print('MODEL LOADED')

dt = p['T']/p['Nt']

DE = DRIVER_EVADER_SYSTEM(n_dr,n_ev,0.1)
dist = 0.5
DE.set_evader_parameters(15.0,4.0,0.2,dist)
DE.set_driver_parameters(0.2,1.0,0.2)
de_step = DE.construct_simulator(dt)
tarx = p['tarx']
tary = p['tary']
de_loss = DE.construct_loss_function([tarx,tary],n_dr)


loss_form_params = [de_step,controller,de_loss,p['Nt'],p['lm'],p['rc']]

opt_params = {i:p[i] for i in ['opt','lr','cmod','cnum','bs','ep']}
TS = TRAINING_SETUP(data,de_step,controller,p['Nt'],de_loss,
                    p['lm'],p['rc'],p['gf'],opt_params)
res = TS.run()

np.savetxt(results_path,res)
if p['save']: controller.save(network_path)
