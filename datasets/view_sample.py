import h5py
import numpy
import matplotlib.pyplot as plt

f = h5py.File('../../training.h5', 'r')
n_samples = len(f['train']['img'])
colours={0:'black', 1:'blue', 2:'orange', 3:'green', 4:'red', 5:'yellow'}

for i in range(n_samples):
    img = f['train']['img'][i]
    kp_2D = f['train']['kp_2D'][i]
    kpx = kp_2D[:,0]
    kpy = kp_2D[:,1]
    img = img.transpose(1,2,0)
    plt.imshow(img)
    # Root of hand drawn
    plt.scatter(kpx[0], kpy[0], color=colours[0])
    # Attach all knuckles to root of hand, then draw then fingers
    for i in range(1, 6):
        plt.plot([kpx[0], kpx[i*4]], [kpy[0], kpy[i*4]], color=colours[0])
        plt.plot(kpx[i*4-3:(i+1)*4-3], kpy[i*4-3:(i+1)*4-3], marker='o', color=colours[i])
    plt.show()
