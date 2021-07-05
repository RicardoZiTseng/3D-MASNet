import numpy as np
import scipy.ndimage
from numpy.core.umath_tests import inner1d
import nibabel as nib

def dice(img1, img2, idx=None):
    """Calculate the dice coeficient between two images of a specific class.
    Args:
        img1: numpy array
        img2: numpy array
        idx:  the label class. In iSeg dataset, 0,1,2,3 represent background, CSF, GM and WM, respectively.
    """
    if idx:
        img1 = img1 == idx
        img2 = img2 == idx
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)
    if img1.shape != img2.shape:
        raise ValueError("Shape missmatch: img1 and img2 must got same shape. But got {} for img1 and {} for img2".format(img1.shape, img2.shape))
    intersection = np.logical_and(img1, img2)
    dsc = 2.0 * intersection.sum() / (img1.sum() + img2.sum())
    return dsc

def ModHausdorffDist(A,B):
    """
    borrow from: https://github.com/zhengyang-wang/3D-Unet--Tensorflow/blob/master/utils/HausdorffDistance.py
    This function computes the Modified Hausdorff Distance (MHD) which is
    proven to function better than the directed HD as per Dubuisson et al.
    in the following work:
    
    M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
    matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
    http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=576361
    
    The function computed the forward and reverse distances and outputs the
    maximum/minimum of both.
    Optionally, the function can return forward and reverse distance.
    
    Format for calling function:
    
    [MHD,FHD,RHD] = ModHausdorffDist(A,B);
    
    where
    MHD = Modified Hausdorff Distance.
    FHD = Forward Hausdorff Distance: minimum distance from all points of B
         to a point in A, averaged for all A
    RHD = Reverse Hausdorff Distance: minimum distance from all points of A
         to a point in B, averaged for all B
    A -> Point set 1, [row as observations, and col as dimensions]
    B -> Point set 2, [row as observations, and col as dimensions]
    
    No. of samples of each point set may be different but the dimension of
    the points must be the same.
    
    Edward DongBo Cui Stanford University; 06/17/2014
    """

    # Find pairwise distance
    D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
    # Calculating the forward HD: mean(min(each col))
    FHD = np.mean(np.min(D_mat,axis=1))
    # Calculating the reverse HD: mean(min(each row))
    RHD = np.mean(np.min(D_mat,axis=0))
    # Calculating mhd
    MHD = np.max(np.array([FHD, RHD]))
    return(MHD, FHD, RHD)

def MHD(pred, label):
    '''Compute 3D MHD for a single class.

    Args:
        pred: An array of size [Depth, Height, Width], with only 0 or 1 values
        label: An array of size [Depth, Height, Width], with only 0 or 1 values

    Returns:
        3D MHD for a single class
    '''
    D, H, W = label.shape
    pred_d = np.array([pred[:, i, j] for i in range(H) for j in range(W)])
    pred_h = np.array([pred[i, :, j] for i in range(D) for j in range(W)])
    pred_w = np.array([pred[i, j, :] for i in range(D) for j in range(H)])

    label_d = np.array([label[:, i, j] for i in range(H) for j in range(W)])
    label_h = np.array([label[i, :, j] for i in range(D) for j in range(W)])
    label_w = np.array([label[i, j, :] for i in range(D) for j in range(H)])

    MHD_d = ModHausdorffDist(pred_d, label_d)[0]
    MHD_h = ModHausdorffDist(pred_h, label_h)[0]
    MHD_w = ModHausdorffDist(pred_w, label_w)[0]

    ret = np.mean([MHD_d, MHD_h, MHD_w])

    return ret

