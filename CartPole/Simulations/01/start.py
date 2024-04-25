from multi_cartpole_data import generate_data
from multi_cartpole_controller import build_fully_connected_network
from cartpole_simulator import build_cartpole_step,build_cartpole_loss
from loss_formulation import LOSS_FORMULATION
from training import TRAINING_SETUP

import tensorflow as tf
import numpy as np
import argparse,os


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
parser.add_argument(  '--NC',  default=0,  type=int, help='Number of poles, complexity') 

# LOSS
parser.add_argument(  '--lm',  default=0,  type=str, help='Loss mode (CONTINUOUS, FINAL)') 

# GRADIENT FLOW
parser.add_argument(  '--gfm',  default=0,  type=str, help='Gradient Flow Mode')

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
dt = p['T']/p['Nt']
p['dt']=dt
for i in p.keys():
    print(i,p[i]) 



#path = '/home/schnell/Projects/Python/servus05_BPTT_Paper/Data/'
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
# load with: np.load(p,allow_pickle='TRUE').item()


os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
np.random.seed(42)
tf.random.set_seed(42)

### SIMULATION ###

os.environ["CUDA_VISIBLE_DEVICES"]=p['gpu']

train_data,test_data = generate_data(p['Nd'],p['NC'])
dataloader = tf.data.Dataset.from_tensor_slices(train_data).batch(p['bs'])
data = [dataloader,train_data,test_data]

if p['load']=='0':
    #network = get_model_1(p['Nx'],100,True)
    controller = build_fully_connected_network(p['width'],p['depth'],p['bias'],p['zero'],p['NC'])
    print('NEW MODEL')
else:
    controller = tf.keras.models.load_model(p['load'])
    print('MODEL LOADED')



cartpole_step = build_cartpole_step(p['dt'])
cartpole_loss = build_cartpole_loss()
opt_p = {i:p[i] for i in ['opt','lr','cmod','cnum','bs','ep']}

TS = TRAINING_SETUP(data,controller,cartpole_step,p['Nt'],
                    cartpole_loss,p['lm'],p['gfm'],opt_p)

res = TS.run()

np.savetxt(results_path,res)
if p['save']: controller.save(network_path)
