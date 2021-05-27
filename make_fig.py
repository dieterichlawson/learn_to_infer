import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
import scipy
from scipy.spatial.distance  import pdist, squareform
data = np.load("probe.npz")

weights = data['tfmr_attn_weights']

cs = data['cs']
xs = data['xs']
log_resps = data['log_resps']

num_heads = 16
num_points = 100
num_clusters = 8

#[num_layers, num_heads, num_points, num_points]
weights = weights[:,0,:,:num_points,:num_points]
#[num_points]
sub_cs = cs[0,:num_points]
#[num_points, 4]
xs = xs[0,:num_points]
#[num_points, 8]
log_resps = log_resps[0,:num_points]

# group the points by cluster, and order the clusters by size
pts_per_cluster = np.array([np.sum(sub_cs ==i) for i in range(8)])
ordered_cs = np.argsort(pts_per_cluster)[::-1]
inds = np.concatenate([np.nonzero(sub_cs == i)[0] for i in ordered_cs], axis=0)

weights = weights[:,:,inds][:,:,:,inds]
xs = xs[inds]
log_resps = log_resps[inds]
affinity_mat = squareform(np.square(pdist(xs, metric='euclidean')))

# compute the location of gridlines
pts_per_cluster = np.array([np.sum(sub_cs ==i) for i in ordered_cs])
cum_pts_per_cluster = np.cumsum(pts_per_cluster)

log_ws = np.log(weights)
log_ws = np.nan_to_num(log_ws, posinf=88, neginf=-88)

def plot_mat(ax, mat):
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_xticks(cum_pts_per_cluster, minor=True)
  ax.set_yticks(cum_pts_per_cluster, minor=True)
  ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
  ax.imshow(mat, interpolation='nearest')

fig, ax = plt.subplots(nrows=13, ncols=16, figsize = (4*16, 4*12))

ax[0,0].set_title("Pairwise Distance Matrix")
plot_mat(ax[0,0], -affinity_mat)

for i in range(num_clusters):
  k = ordered_cs[i]
  print(k)
  lrs = log_resps[:,k]
  mat = np.tile(lrs[np.newaxis,:], [num_points, 1])
  ax[0,i+1].set_title("Cluster %d Membership Probs" % (i+1))
  plot_mat(ax[0, i+1], mat)
 
for i in range(12):
  flat_log_ws = log_ws[i,...].reshape([num_heads,-1])
  #clustering = SpectralClustering(n_clusters=3, assign_labels='discretize').fit_predict(flat_log_ws)
  #order = np.concatenate([np.nonzero(clustering == i)[0] for i in range(3)], axis=0)
  
  order = np.argsort(PCA(n_components=4).fit(flat_log_ws).score_samples(flat_log_ws))
  layer_log_ws = log_ws[i, order,...]
  for j in range(16):
    plot_mat(ax[i+1,j], layer_log_ws[j])
      #ax[i,j].set_xticks([])
      #ax[i,j].set_yticks([])
      #ax[i,j].set_xticks(cum_pts_per_cluster, minor=True)
      #ax[i,j].set_yticks(cum_pts_per_cluster, minor=True)
      #ax[i,j].grid(which='minor', color='black', linestyle='-', linewidth=1)
      #ax[i,j].imshow(layer_log_ws[j], interpolation='nearest')

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.1)
plt.savefig("wts.pdf")

