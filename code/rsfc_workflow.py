import os
import pickle
import numpy as np
import os.path as op
import nibabel as nib
from glob import glob
from nilearn.masking import unmask
from nilearn.masking import apply_mask
from nilearn.input_data import NiftiMasker
from nilearn.connectome import ConnectivityMeasure
from brainspace.gradient import GradientMaps
import matplotlib.pyplot as plt
from nilearn import plotting


#establish project direcotry
project_directory = '/home/data/nbc/misc-projects/ifg_meta-gradients'
rsfc_gradient_output_dir = op.join(project_directory, 'derivatives', 'rsfc_gradients')
os.makedirs(rsfc_gradient_output_dir, exist_ok=True)

#load ROI and convert image space locations (ijk) to coordinates (xyz)
roi = op.join(project_directory, 'code', 'Left_IFG_MNI_2mm.nii')
roi_img = nib.load(roi)
roi_img_mask = NiftiMasker(mask_img=roi_img).fit()

if not op.isfile(op.join(rsfc_gradient_output_dir, 'corrmat.pkl')):
    hcp_data_dir = '/home/data/nbc/external-datasets/hcp/niconn-hcp/hcp-openaccess/HCP1200/derivatives/gsr+smooth'
    hcp_subs = os.listdir(hcp_data_dir)

    correlation = ConnectivityMeasure(kind='correlation')
    corrmat = np.zeros((len(np.where(roi_img.get_fdata())[0]), len(np.where(roi_img.get_fdata())[0])))
    for j, sub in enumerate(hcp_subs):
        print(j)
        if not op.isfile(op.join(rsfc_gradient_output_dir, sub, 'corrmat.pkl')):
            runs = glob(op.join(hcp_data_dir, sub, '*clean_smooth.nii.gz'))
            sub_mean_corrmat = np.zeros((len(np.where(roi_img.get_fdata())[0]), len(np.where(roi_img.get_fdata())[0])))
            for i, run in enumerate(runs):
                print(run)
                #roi_ts = apply_mask(run, roi_img)
                roi_ts = roi_img_mask.transform(run)
                roi_ts_na_idx = np.where(np.any(roi_ts, axis=0) == False)[0]
                roi_ts = np.delete(roi_ts, roi_ts_na_idx, axis=1)
                tmp_sub_corrmat = correlation.fit_transform([roi_ts])[0]
                #transform to z-scores
                tmp_sub_corrmat = np.arctanh(tmp_sub_corrmat)
                tmp_sub_corrmat = np.insert(tmp_sub_corrmat, np.subtract(roi_ts_na_idx, np.arange(len(roi_ts_na_idx))), 0, axis=0)
                tmp_sub_corrmat = np.insert(tmp_sub_corrmat, np.subtract(roi_ts_na_idx, np.arange(len(roi_ts_na_idx))), 0, axis=1)
                np.fill_diagonal(tmp_sub_corrmat, np.arctanh(1-np.finfo(float).eps))
                sub_mean_corrmat = sub_mean_corrmat + tmp_sub_corrmat

            sub_mean_corrmat = sub_mean_corrmat/(i+1)
            #save correlation matrix
            os.makedirs(op.join(rsfc_gradient_output_dir, sub), exist_ok=True)
            with open(op.join(rsfc_gradient_output_dir, sub, 'corrmat.pkl'), 'wb') as fo:
                pickle.dump(sub_mean_corrmat, fo, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            with open(op.join(rsfc_gradient_output_dir, sub, 'corrmat.pkl'), 'rb') as fo:
                sub_mean_corrmat = pickle.load(fo)

        corrmat = corrmat + sub_mean_corrmat

    corrmat = corrmat/(j+1)
    #save correlation matrix
    with open(op.join(rsfc_gradient_output_dir, 'corrmat.pkl'), 'wb') as fo:
        pickle.dump(corrmat, fo, protocol=pickle.HIGHEST_PROTOCOL)

else:
    with open(op.join(rsfc_gradient_output_dir, 'corrmat.pkl'), 'rb') as fo:
        corrmat = pickle.load(fo)

corrmat = np.tanh(corrmat)
plt.imshow(corrmat, cmap='seismic')
plt.savefig(op.join(rsfc_gradient_output_dir, 'corrmat.png'))
plt.close()

# Build gradients using diffusion maps
gm = GradientMaps(n_components=10, approach='dm', kernel='cosine')
gm.fit(corrmat, sparsity=0.9)

#save gradients and lambdas
with open(op.join(rsfc_gradient_output_dir, 'lambdas.txt'), 'w') as fo:
    np.savetxt(fo, gm.lambdas_)

for i_grad in range(gm.gradients_.shape[1]):
    #save gradient maps
    tmp_grad_img = unmask(gm.gradients_[:,i_grad], roi_img)
    nib.save(tmp_grad_img, op.join(rsfc_gradient_output_dir, 'gradient-{}.nii.gz'.format(i_grad)))
    plotting.plot_stat_map(op.join(rsfc_gradient_output_dir, 'gradient-{}.nii.gz'.format(i_grad)),
                           draw_cross=False,
                           dim=-0.3)
    plt.savefig(op.join(rsfc_gradient_output_dir, 'gradient-{}.png'.format(i_grad)))
    plt.close()

    #save sorted correlation matrix for each gradient
    grad_sort_inds = np.argsort(gm.gradients_[:,i_grad])
    grad_sort_corr_mat = corrmat.copy()
    grad_sort_corr_mat = grad_sort_corr_mat[grad_sort_inds,:]
    grad_sort_corr_mat = grad_sort_corr_mat[:,grad_sort_inds]

    with open(op.join(rsfc_gradient_output_dir, 'gradient-{}_corrmat-sorted.pkl'.format(i_grad)), 'wb') as fo:
        pickle.dump(grad_sort_corr_mat, fo, protocol=pickle.HIGHEST_PROTOCOL)
    plt.imshow(grad_sort_corr_mat, cmap='seismic')
    plt.savefig(op.join(rsfc_gradient_output_dir, 'gradient-{}_corrmat-sorted.png'.format(i_grad)))
    plt.close()
