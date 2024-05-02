import numpy as np
import matplotlib.pyplot as plt


colors = ['#ef476f','#ffd166','#06d6a0','#073b4c']
gfm_dict = {'F':0,'P':1,'C':2,'S':3}
regu_dict={0.001:4, 0.01:3, 0.1:2, 1.0:1}
labels={'F':"R",'P':'M','C':"C",'S':'S'}

def smooth(a):
    kernel_size = 30
    kernel = np.ones(kernel_size) / kernel_size
    a_ext = np.concatenate([a[0]*np.ones((kernel_size-1,)),a])
    b = np.convolve(a_ext, kernel, mode='valid')
    return b

def plot_result(run, ax):
    results = np.loadtxt(run+'/results.txt')
    params = np.load(run+'/params.npy',allow_pickle=True).item()
    ci = gfm_dict[params['gf']]
    label = params['gf']
    i=regu_dict[params['rc']]-1
    curve = results[:,2] 
    curve = smooth(curve)
    ax[i].plot(curve,label=labels[label],color=colors[ci])
    ax[i].set_title('Regularization: '+str(params['rc']))
    return params


fig = plt.figure(figsize=(8,5))
gs = fig.add_gridspec(2,2)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax = [ax0,ax1,ax2,ax3]



runs =[]
for i in range(48):
    run = 'Data/01/01_'+str(i).zfill(4)
    runs.append(run)
    
for i,run in enumerate(runs):
    try:
        params = plot_result(run,ax)
    except Exception as Error:
        pass

for iax,axi in enumerate(ax[:]):
    axi.set_yscale('log')
    axi.set_ylim([0.4,6])
    axi.set_yticks([0.5,0.7,1,2,3,5])
    axi.set_yticklabels(['0.5','0.7','1','2','3','5'])
    if iax>1: axi.set_xlabel('Epochs',fontsize=12)
    if iax%2==0: axi.set_ylabel('Loss',fontsize=12)
    if iax==0:
        handles, labels = axi.get_legend_handles_labels()
        handle_list, label_list = [], []
        for handle, label in zip(handles, labels):
            if label not in label_list:
                handle_list.append(handle)
                label_list.append(label)
        axi.legend(handle_list, label_list,loc=8,ncol=2)

plt.suptitle('Guidance by Repulsion',fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('Plots/01/Guidance_by_repulsion.png')
