import numpy as np
import matplotlib.pyplot as plt


colors = ['#ef476f','#ffd166','#06d6a0','#073b4c']
labels={'F':"R",'P':'M','C':"C",'S':'S'}

gfm_dict = {
    'F':0,
    'P':1,
    'C':2,
    'S':3
}

def smooth(a):
    kernel_size = 30
    kernel = np.ones(kernel_size) / kernel_size
    a_ext = np.concatenate([a[0]*np.ones((kernel_size-1,)),a])
    b = np.convolve(a_ext, kernel, mode='valid')
    return b


def plot_result(run, ax,cmod,nc):
    results = np.loadtxt(run+'/results.txt')
    params = np.load(run+'/params.npy',allow_pickle=True).item()
    
    if params['cmod']!=cmod: return 0
    if params['NC']!=nc: return 0

    ci1 = gfm_dict[params['gfm']]
    label = params['gfm']
    curve = results[:,2] 
    curve = smooth(curve)
    if params['cmod']=='NORM':
        ax.plot(curve,label=labels[label],color=colors[ci1])
    else:
        ax.plot(curve,color=colors[ci1])


fig = plt.figure(figsize=(7,5))
gs = fig.add_gridspec(2,2)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax = [ax0,ax1,ax2,ax3]


for cmod in ['VALUE','NORM','NONE']:
    for nc in [1,2,3,4]:
        for i in range(48):
            run = 'Data/01/01_'+str(i).zfill(4)
            try: plot_result(run,ax[nc-1],cmod,nc)
            except Exception: pass

for iax,axi in enumerate(ax):
    axi.set_yscale('log')
    axi.set_ylim([1*10**-6,3])
    axi.set_title('Number of Poles: '+str(1+iax),fontsize=12)
    if iax>1: axi.set_xlabel('Epochs',fontsize=12)
    if iax%2==0: axi.set_ylabel('Loss',fontsize=12)
    if iax==2:axi.legend(loc=8,ncol=2)

plt.suptitle('   Cartpole',fontsize=22)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('Plots/01/cartpole.png')


