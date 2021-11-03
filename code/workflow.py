import os
import pickle
import numpy as np
import os.path as op
import nibabel as nib
from nimare.extract import fetch_neuroquery
from nimare.io import convert_neurosynth_to_dataset
from nimare.extract import download_abstracts
from nimare.dataset import Dataset
from nimare.utils import vox2mm
from nimare.meta.cbma.ale import ALE
from nilearn.masking import unmask
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
from brainspace.gradient import GradientMaps
import matplotlib.pyplot as plt


#function for download neuroquery data, taken from
#https://nimare.readthedocs.io/en/latest/auto_examples/01_datasets/download_neurosynth.html#do-the-same-with-neuroquery
#replaced 7547 with 6308 because of "combined" data availability
def download_neuroquery(out_dir):
    files = fetch_neuroquery(
        data_dir=out_dir,
        version="1",
        overwrite=False,
        source="combined",
        vocab="neuroquery6308",
        type="tfidf",
    )

    # Note that the files are saved to a new folder within "out_dir" named "neuroquery".
    neuroquery_db = files[0]

    # Note that the conversion function says "neurosynth".
    # This is just for backwards compatibility.
    neuroquery_dset = convert_neurosynth_to_dataset(
        coordinates_file=neuroquery_db["coordinates"],
        metadata_file=neuroquery_db["metadata"],
        annotations_files=neuroquery_db["features"],
    )
    neuroquery_dset.save(op.join(out_dir, "neuroquery_dataset.pkl.gz"))

    # NeuroQuery also uses PMIDs as study IDs.
    neuroquery_dset = download_abstracts(neuroquery_dset, "miriedel@fiu.edu")
    neuroquery_dset.save(op.join(out_dir, "neuroquery_dataset_with_abstracts.pkl.gz"))


#establish project direcotry
project_directory = '/home/data/nbc/misc-projects/ifg_meta-gradients'
macm_gradient_output_dir = op.join(project_directory, 'derivatives', 'macm_gradients')
os.makedirs(macm_gradient_output_dir, exist_ok=True)

#load ROI and convert image space locations (ijk) to coordinates (xyz)
roi = op.join(project_directory, 'code', 'Left_IFG_MNI_2mm.nii')
roi_img = nib.load(roi)
roi_idx = np.vstack(np.where(roi_img.get_fdata())).T
roi_coords = vox2mm(roi_idx, roi_img.affine)

if not op.isfile(op.join(macm_gradient_output_dir, 'corrmat.pkl')):

    #location of the neuroquery dataset
    neuroquery_dset = op.join(project_directory, 'code', 'neuroquery', 'neuroquery_dataset_with_abstracts.pkl.gz')
    #dont download every time
    if not op.isfile(neuroquery_dset):
        download_neuroquery(op.join(project_directory, 'code', 'neuroquery'))
    else:
        dset = Dataset.load(neuroquery_dset)

    #generate an ALE image for each coordinate by running ALE algorithm on studies
    #reporting foci within 6mm of the coordinate of interest
    macm_ales = np.zeros((roi_coords.shape[0], 228453))
    for i_coord in range(roi_coords.shape[0]):
        coord = roi_coords[i_coord, :][None, :]
        coord_ids = dset.get_studies_by_coordinate(coord, r=6)
        coord_dset = dset.slice(coord_ids)

        ale = ALE(kernel__fwhm=15)
        images = ale.fit(coord_dset)

        macm_ales[i_coord,:] = images.maps['stat']

    #apply correlation to study x brain voxel matrix
    correlation = ConnectivityMeasure(kind='correlation')
    macm_ales_correlation_matrix = correlation.fit_transform([np.transpose(macm_ales)])[0]
    #save correlation matrix
    with open(op.join(macm_gradient_output_dir, 'corrmat.pkl'), 'wb') as fo:
        pickle.dump(macm_ales_correlation_matrix, fo, protocol=pickle.HIGHEST_PROTOCOL)

else:
    with open(op.join(macm_gradient_output_dir, 'corrmat.pkl'), 'rb') as fo:
        macm_ales_correlation_matrix = pickle.load(fo)

plt.imshow(macm_ales_correlation_matrix, cmap='seismic')
plt.savefig(op.join(macm_gradient_output_dir, 'corrmat.png'))
plt.close()

# Build gradients using diffusion maps
gm = GradientMaps(n_components=10, approach='dm', kernel='cosine')
gm.fit(macm_ales_correlation_matrix, sparsity=0.9)

#save gradients and lambdas
with open(op.join(macm_gradient_output_dir, 'lambdas.txt'), 'w') as fo:
    np.savetxt(fo, gm.lambdas_)

for i_grad in range(gm.gradients_.shape[1]):
    #save gradient maps
    tmp_grad_img = unmask(gm.gradients_[:,i_grad], roi_img)
    nib.save(tmp_grad_img, op.join(macm_gradient_output_dir, 'gradient-{}.nii.gz'.format(i_grad)))
    plotting.plot_stat_map(op.join(macm_gradient_output_dir, 'gradient-{}.nii.gz'.format(i_grad)),
                           draw_cross=False,
                           dim=-0.3)
    plt.savefig(op.join(macm_gradient_output_dir, 'gradient-{}.png'.format(i_grad)))
    plt.close()

    #save sorted correlation matrix for each gradient
    grad_sort_inds = np.argsort(gm.gradients_[:,i_grad])
    grad_sort_corr_mat = macm_ales_correlation_matrix.copy()
    grad_sort_corr_mat = grad_sort_corr_mat[grad_sort_inds,:]
    grad_sort_corr_mat = grad_sort_corr_mat[:,grad_sort_inds]

    with open(op.join(macm_gradient_output_dir, 'gradient-{}_corrmat-sorted.pkl'.format(i_grad)), 'wb') as fo:
        pickle.dump(grad_sort_corr_mat, fo, protocol=pickle.HIGHEST_PROTOCOL)
    plt.imshow(grad_sort_corr_mat, cmap='seismic')
    plt.savefig(op.join(macm_gradient_output_dir, 'gradient-{}_corrmat-sorted.png'.format(i_grad)))
    plt.close()
