from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# instantiate the nnUNetPredictor
predictor = nnUNetPredictor(
	tile_step_size=0.5,
	use_gaussian=True,
	use_mirroring=True,
	perform_everything_on_device=True,
	device=torch.device('cuda', 0),
	verbose=False,
	verbose_preprocessing=False,
	allow_tqdm=True
)
# initializes the network architecture, loads the checkpoint
predictor.initialize_from_trained_model_folder(
	join(nnUNet_results, 'Dataset011_seeg_contacts/nnUNetTrainer__nnUNetPlans__3d_fullres'),
	use_folds=(0,),
	checkpoint_name='checkpoint_final.pth',
)
# variant 1: give input and output folders
predictor.predict_from_files(join(nnUNet_raw),
								join(nnUNet_raw, 'templbl'),
								save_probabilities=False, overwrite=False,
								num_processes_preprocessing=2, num_processes_segmentation_export=2,
								folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)