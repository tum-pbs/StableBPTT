from qmc_training_setup import TRAINING_SETUP
from qmc_solver import QMC
from model import get_cnn
from data import generate_data
import tensorflow as tf
import numpy as np
import argparse,os

parser = argparse.ArgumentParser(description='CmdLine Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# MISC
parser.add_argument(  '--gpu',  default=0,  type=str, help='Cuda visible devices')
parser.add_argument(  '--save',  default=False,  type=bool, help='Save model?')
parser.add_argument(  '--load',  default=0,  type=str, help='Load model? Path or 0')
parser.add_argument(  '--weighting',  default=0,  type=str, help='Dataset: weighting of eigenstates')

# DATA
parser.add_argument(  '--Nd',  default=0,  type=int, help='Number of data points')


# NETWORK
parser.add_argument(  '--width',  default=0,  type=int, help='Network width')
parser.add_argument(  '--depth',  default=0,  type=int, help='Network depth')
parser.add_argument(  '--bias',  default=True,  type=bool, help='Use bias?')
parser.add_argument(  '--zero',  default=False,  type=bool, help='Use zero initialization?')

# PHYSICS
parser.add_argument(  '--Nx',  default=0,  type=int, help='Resolution (physics system)')
parser.add_argument(  '--dt',  default=0,  type=float, help='Time step (physics system)')
parser.add_argument(  '--Nt',  default=0,  type=int, help='Number of solver time steps') 
parser.add_argument(  '--TS',  default=0,  type=int, help='Target state (number of eigenstate: 1-groundstate, 2-first excited state, ...)') 

# LOSS
parser.add_argument(  '--LT',  default=0,  type=str, help='Loss type (CONTINUOUS, FINAL)') 

# GRADIENT FLOW
parser.add_argument(  '--gfm',  default=0,  type=str, help='Gradient flow mode')

# OPTIMIZATION
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



os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
np.random.seed(42)
tf.random.set_seed(42)

### SIMULATION ###

os.environ["CUDA_VISIBLE_DEVICES"]=p['gpu']

train,test = generate_data(p['Nd']*2,p['Nx'],p['weighting'])
dataloader = tf.data.Dataset.from_tensor_slices(train).batch(p['bs'])
data = [dataloader,train,test]

if p['load']=='0':
    network = get_cnn(p['Nx'],p['width'],p['bias'],p['zero'])
    print('NEW MODEL')
else:
    network = tf.keras.models.load_model(p['load'])
    print('MODEL LOADED')

solver = QMC(p['dt'],p['Nx']).step
opt_p = {i:p[i] for i in ['opt','lr','cmod','cnum','bs','ep']} 
ts = TRAINING_SETUP(data,network,solver,p['Nt'],p['Nx'],p['TS'],p['LT'],p['gfm'],opt_p)
res = ts.run()
res = np.array(res)



np.savetxt(results_path,res)
if p['save']: network.save(network_path)
