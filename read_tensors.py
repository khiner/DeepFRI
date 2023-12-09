import numpy as np


cmap = np.load('./pt_tensors/input_cmap.npy')
seq = np.load('./pt_tensors/input_seq.npy')
label = np.load('./pt_tensors/labels.npy')

print(f'Cmap:\n{cmap[0]}\n')
print(f'Seq:\n{seq[0]}\n')

embedding = np.load('./pt_tensors/embed_tensor.npy')
# print(f'Input layer (Seqs only):\n{embedding[0]}\n')
# print(embedding[0].shape)

# norm = np.load('./pt_tensors/normalized.npy')
# print(f'Norm:\n{norm[0]}\n')

# batch_mm = np.load('./pt_tensors/batch_matmult.npy')
# print(f'Battch_mm:\n{batch_mm[0]}\n')

# gcnn_act = np.load('./pt_tensors/gcnn_activation.npy')
# print(f'gcnn_act:\n{gcnn_act[0]}\n')

sumPool = np.load('./pt_tensors/sum_tensor.npy')
print(f'Sum Pooling:\n{sumPool}\n')

fcTensor = np.load('./pt_tensors/fc_tensor.npy')
print(f'FC Layers:\n{fcTensor[0]}\n')

outTensor = np.load('./pt_tensors/out_tensor.npy')
print(f'Output:\n{outTensor[0][0]}\n')

