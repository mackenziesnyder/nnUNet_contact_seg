# import nibabel as nib
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import map_coordinates

# def extract_oblique_slice(nifti_img, entry_world, exit_world, slice_thickness=1.0, num_points=256):
#     data = nifti_img.get_fdata()
#     affine = nifti_img.affine
#     inv_affine = np.linalg.inv(affine)

#     direction = np.array(exit_world, dtype=np.float64) - np.array(entry_world, dtype=np.float64)

#     direction /= np.linalg.norm(direction)

#     # Build orthonormal basis
#     u = direction
#     temp = np.array([1, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1, 0])
#     v = np.cross(u, temp)
#     v /= np.linalg.norm(v)
#     w = np.cross(u, v)

#     center = (entry_world + exit_world) / 2.0

#     grid_v = np.linspace(-slice_thickness * num_points / 2,
#                          slice_thickness * num_points / 2, num_points)
#     grid_w = np.linspace(-slice_thickness * num_points / 2,
#                          slice_thickness * num_points / 2, num_points)
#     vv, ww = np.meshgrid(grid_v, grid_w)

#     coords_world = center[:, None, None] + (v[:, None, None] * vv) + (w[:, None, None] * ww)
#     coords_world = coords_world.reshape(3, -1)
#     coords_voxel = nib.affines.apply_affine(inv_affine, coords_world.T).T

#     slice_values = map_coordinates(data, coords_voxel, order=1, mode='nearest')
#     slice_img = slice_values.reshape((num_points, num_points))

#     return slice_img, center, v, w

# def project_contact(contact_world, center, v, w):
#     rel = contact_world - center
#     x = np.dot(rel, v)
#     y = np.dot(rel, w)
#     return x, y

# # --- TEST SETUP ---

# nifti_path = "/local/scratch/nnUNet_contact_seg/nnunet_contact_seg/workflow/scripts/sub-P167_run-01_space-T1w_ct.nii.gz"  # Replace with your test CT image
# img = nib.load(nifti_path)

# # Define test entry/exit (in world space)
# entry_world = np.array([-23.045753,21.187383,-59.080582])
# exit_world = np.array([-72.258925,11.394724,-46.995941])

# # Optional: test some contact coordinates near the trajectory
# contact_worlds = [
#     np.array([-24.000,23.200,-57.500]),
#     np.array([-28.500,22.300,-56.300]),
#     np.array([33.400,21.500,-55.200]),
#     np.array([-38.300,20.600,-54.000]),
#     np.array([-43.000,19.700,-53.100]),
#     np.array([-47.600,18.600,-52.300]),
#     np.array([-52.300,17.600,-51.200]),
#     np.array([-57.000,16.500,-50.200]),
#     np.array([-61.500,15.700,-49.000]),
#     np.array([-66.300,14.700,-47.800]),
# ]

# # Get oblique slice
# slice_img, center, v, w = extract_oblique_slice(img, entry_world, exit_world)

# # Project contact points
# num_points = slice_img.shape[0]
# half_range = (num_points / 2)
# to_pixel = lambda x: (x + half_range) * (num_points / (2 * half_range))

# contact_2d = [project_contact(p, center, v, w) for p in contact_worlds]
# contact_px = [(to_pixel(x), to_pixel(y)) for x, y in contact_2d]

# # Plot result
# plt.figure(figsize=(6, 6))
# plt.imshow(slice_img.T, cmap="gray", origin="lower")
# for x, y in contact_px:
#     plt.plot(x, y, 'ro', markersize=4)
# plt.title("Oblique Slice with Projected Contacts")
# plt.axis("off")
# plt.tight_layout()
# plt.show()

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

def extract_oblique_slice(img, entry_world, exit_world, slice_size=256, slice_thickness=1.0):
    data = img.get_fdata()
    affine = img.affine
    inv_affine = np.linalg.inv(affine)

    # Direction vector from entry to exit (world coords)
    direction = np.array(exit_world) - np.array(entry_world)
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("Entry and exit points are identical.")
    direction /= length

    # Create orthonormal basis: u (direction), v, w
    u = direction
    # Choose an arbitrary vector not parallel to u
    temp = np.array([1, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1, 0])
    v = np.cross(u, temp)
    v /= np.linalg.norm(v)
    w = np.cross(u, v)

    # Slice center at midpoint of trajectory
    center = (entry_world + exit_world) / 2

    # Create grid for slice (v-w plane)
    grid_range = slice_thickness * slice_size / 2
    grid_v = np.linspace(-grid_range, grid_range, slice_size)
    grid_w = np.linspace(-grid_range, grid_range, slice_size)
    vv, ww = np.meshgrid(grid_v, grid_w)

    # Coordinates in world space of each point on the slice
    coords_world = center[:, None, None] + (v[:, None, None] * vv) + (w[:, None, None] * ww)
    coords_world = coords_world.reshape(3, -1)

    # Convert world coords to voxel coords
    coords_voxel = nib.affines.apply_affine(inv_affine, coords_world.T).T

    # Sample image at voxel coords
    slice_vals = map_coordinates(data, coords_voxel, order=1, mode='nearest')
    slice_img = slice_vals.reshape(slice_size, slice_size)

    return slice_img, center, v, w

# Example usage:

nifti_path = "/local/scratch/nnUNet_contact_seg/nnunet_contact_seg/workflow/scripts/sub-P167_run-01_space-T1w_ct.nii.gz"
img = nib.load(nifti_path)

entry_world = np.array([-23.045753, 21.187383, -59.080582])
exit_world = np.array([-72.258925, 11.394724, -46.995941])

slice_img, center, v, w = extract_oblique_slice(img, entry_world, exit_world)

# Project entry and exit to slice coordinates (v,w plane)
def project_point_to_slice(pt_world, center, v, w):
    vec = pt_world - center
    return (np.dot(vec, v), np.dot(vec, w))

entry_2d = project_point_to_slice(entry_world, center, v, w)
exit_2d = project_point_to_slice(exit_world, center, v, w)

# Convert to pixel coords (slice_img coords go from 0 to slice_size)
slice_size = slice_img.shape[0]
half = slice_size / 2

def to_pixel(coords_2d):
    return (coords_2d[0] + half, coords_2d[1] + half)

entry_px = to_pixel(entry_2d)
exit_px = to_pixel(exit_2d)

print("Entry 2D coords:", entry_2d)
print("Exit 2D coords:", exit_2d)

# Plot slice and line between entry and exit
plt.figure(figsize=(6,6))
plt.imshow(slice_img.T, cmap='gray', origin='lower')
plt.plot([entry_px[0], exit_px[0]], [entry_px[1], exit_px[1]], 'b-', linewidth=2, label='Trajectory')
plt.scatter(*entry_px, c='green', s=50, label='Entry')
plt.scatter(*exit_px, c='red', s=50, label='Exit')
plt.legend()
plt.title("Oblique slice tangent to trajectory with entry-exit line")
plt.axis('off')
plt.show()


