import numpy as np
import matplotlib.pyplot as plt

colors = ['#ef476f','#ffd166','#06d6a0','#073b4c']
labels={'F':"R",'P':'M','C':"C",'S':'S'}
gfm_dict = {'F':0,'P':1,'C':2,'S':3}


def smooth(a):
    kernel_size = 30
    kernel = np.ones(kernel_size) / kernel_size
    a_ext = np.concatenate([a[0]*np.ones((kernel_size-1,)),a])
    b = np.convolve(a_ext, kernel, mode='valid')
    return b


def plot_result(run, ax, nc):  
    if nc==6:
        nc = 5
        pl = 3
    else: pl = 2
    results = np.loadtxt(run+'/results.txt')
    params = np.load(run+'/params.npy',allow_pickle=True).item()
    if params['TS']!=nc:
        return 0
    ci1 = gfm_dict[params['gfm']]
    curve = results[:,pl] 
    curve = smooth(curve)
    ax.plot(curve,label=labels[params['gfm']],color=colors[ci1])
    return params





fig = plt.figure(figsize=(8,5))
gs = fig.add_gridspec(2,2)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax = [ax0,ax1,ax2,ax3]


for nc in [3,4,5,6]:
    runs =[]
    for i in range(48):
        run = 'Data/01/01_'+str(i).zfill(4)
        runs.append(run)

    for i,run in enumerate(runs):
        try:
            params = plot_result(run,ax[nc-3],nc)
        except Exception as Error:
            pass


for iax,axi in enumerate(ax):
    axi.set_title('Target state: '+str(2+iax),fontsize=12)
    axi.set_yscale('log')
    axi.set_xlabel('Epochs',fontsize=12)
    axi.set_ylabel('Loss',fontsize=12)
    if iax!=3: 
        axi.set_ylim([16.5,135])
        axi.set_yticks([20,30,40,60,90])
        axi.set_yticklabels(['20','30','40','60','90'])
    if iax==3: 
        axi.set_title('Update size',fontsize=12)
        axi.set_ylabel('L2 norm',fontsize=12)
    if iax==0:
        handles, labels = axi.get_legend_handles_labels()
        handle_list, label_list = [], []
        for handle, label in zip(handles, labels):
            if label not in label_list:
                handle_list.append(handle)
                label_list.append(label)
        axi.legend(handle_list, label_list,loc='upper center',ncol=2,fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('Plots/01/QMC.png')
plt.close()