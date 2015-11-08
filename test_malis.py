import numpy as np
import malis as m
import h5py
np.set_printoptions(precision=4)

segTrue = np.array([0, 1, 1, 1, 2, 2, 0, 5, 5, 5, 5],dtype=np.int32);
node1 = np.arange(segTrue.shape[0]-1,dtype=np.int32)
node2 = np.arange(1,segTrue.shape[0],dtype=np.int32)
nVert = segTrue.shape[0]
edgeWeight = np.array([0, 1, 2, 0, 2, 0, 0, 1, 2, 2.5],dtype=np.float32);
edgeWeight = edgeWeight/edgeWeight.max()
print segTrue
print edgeWeight

nPairPos = m.malis_loss_weights(segTrue, node1, node2, edgeWeight, 1)
nPairNeg = m.malis_loss_weights(segTrue, node1, node2, edgeWeight, 0)
print np.vstack((nPairPos,nPairNeg))
# print nPairNeg

idxkeep = (edgeWeight > 0).astype(np.int32)
cc = m.connected_components(nVert,node1,node2,idxkeep)
print cc


# node1, node2 = m.nodelist_like((2,3,4),-np.eye(3))
# print node1
# print node2

hdf5_raw_file = '/groups/turaga/turagalab/greentea/project_data/dataset_06/fibsem_medulla_7col/trvol-250-1-h5/img_normalized.h5'
hdf5_gt_file = '/groups/turaga/turagalab/greentea/project_data/dataset_06/fibsem_medulla_7col/trvol-250-1-h5/groundtruth_seg.h5'
hdf5_aff_file = '/groups/turaga/turagalab/greentea/project_data/dataset_06/fibsem_medulla_7col/trvol-250-1-h5/groundtruth_aff.h5'

#hdf5_raw_file = 'zebrafish_friedrich/raw.hdf5'
#hdf5_gt_file = 'zebrafish_friedrich/labels_2.hdf5'


hdf5_raw = h5py.File(hdf5_raw_file, 'r')
hdf5_gt = h5py.File(hdf5_gt_file, 'r')
hdf5_aff = h5py.File(hdf5_aff_file, 'r')

nhood = -np.eye(3)
seg = np.asarray(hdf5_gt['main']).astype(np.int32)
aff = m.seg_to_affgraph(seg,nhood)
cc,ccSizes = m.connected_components_affgraph(aff,nhood)
aff2 = m.seg_to_affgraph(cc,nhood)
cc2,ccSizes2 = m.connected_components_affgraph(aff2,nhood)

print "Comparing 'seg' and 'cc':"
frac_disagree = np.mean(seg.ravel()!=cc.ravel())
ri,V_rand,prec,rec = m.rand_index(seg,cc)
print "Connected components disagree at %f%% locations" % (frac_disagree*100)
print "\tRand index: %f, V_rand: %f, prec: %f, rec: %f" % (ri,V_rand,prec,rec)

print "Comparing 'cc' and 'cc2':"
frac_disagree = np.mean(cc.ravel()!=cc2.ravel())
ri,V_rand,prec,rec = m.rand_index(cc,cc2)
print "Connected components disagree at %f%% locations" % (frac_disagree*100)
print "\tRand index: %f, V_rand: %f, prec: %f, rec: %f" % (ri,V_rand,prec,rec)
