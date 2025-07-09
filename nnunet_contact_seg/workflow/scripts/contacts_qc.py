import nibabel as nib
import re
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import csv
from nilearn import plotting
from svgutils.compose import Unit
from svgutils.transform import GroupElement, SVGFigure, fromstring
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import io
import base64

def svg2str(display_object, dpi):
    """Serialize a nilearn display object to string."""

    image_buf = StringIO()
    display_object.frame_axes.figure.savefig(
        image_buf, dpi=dpi, format="svg", facecolor="k", edgecolor="k"
    )
    return image_buf.getvalue()


def extract_svg(display_object, dpi=250):
    """Remove the preamble of the svg files generated with nilearn."""
    image_svg = svg2str(display_object, dpi)

    image_svg = re.sub(' height="[0-9]+[a-z]*"', "", image_svg, count=1)
    image_svg = re.sub(' width="[0-9]+[a-z]*"', "", image_svg, count=1)
    image_svg = re.sub(
        " viewBox", ' preseveAspectRation="xMidYMid meet" viewBox', image_svg, count=1
    )
    start_tag = "<svg "
    start_idx = image_svg.find(start_tag)
    end_tag = "</svg>"
    end_idx = image_svg.rfind(end_tag)

    # rfind gives the start index of the substr. We want this substr
    # included in our return value so we add its length to the index.
    end_idx += len(end_tag)

    return image_svg[start_idx:end_idx]


def clean_svg(bg1_svgs, bg2_svgs, ref=0):
    # Expecting exactly one slice per direction in each group
    svgs = bg1_svgs + bg2_svgs
    roots = [f.getroot() for f in svgs]

    sizes = []
    for f in svgs:
        viewbox = [float(v) for v in f.root.get("viewBox").split(" ")]
        width = int(viewbox[2])
        height = int(viewbox[3])
        sizes.append((width, height))

    sizes = np.array(sizes)
    width = sizes[ref, 0]
    scales = width / sizes[:, 0]
    heights = sizes[:, 1] * scales

    # Total height based on number of slices in bg1
    nsvgs = len(bg1_svgs)
    fig = SVGFigure(Unit(f"{width}px"), Unit(f"{heights[:nsvgs].sum()}px"))

    yoffset = 0
    for i, r in enumerate(roots):
        r.moveto(0, yoffset, scale_x=scales[i])
        if i < nsvgs - 1:
            yoffset += heights[i]

    # Create two background groups for blending
    newroots = [
        GroupElement(roots[:nsvgs], {"class": "background-svg ct"}),
        GroupElement(roots[nsvgs:], {"class": "background-svg t1w"}),
    ]

    fig.append(newroots)
    fig.root.attrib.pop("width", None)
    fig.root.attrib.pop("height", None)
    fig.root.set("preserveAspectRatio", "xMidYMid meet")

    with TemporaryDirectory() as tmpdirname:
        out_file = Path(tmpdirname) / "tmp.svg"
        fig.save(str(out_file))
        svg = out_file.read_text().splitlines()

    if svg[0].startswith("<?xml"):
        svg = svg[1:]

    svg.insert(
        2,
        """\
        <style type="text/css">
        .background-svg.ct {
            opacity: 1;
            transition: opacity 0.2s ease;
        }
        .background-svg.t1w {
            opacity: 0;
            transition: opacity 0.2s ease;
        }
        </style>
        <script>
        function updateBlend(slider) {
            const alpha = parseFloat(slider.value) / 100;
            document.querySelectorAll('.background-svg.ct').forEach(el => {
                el.style.opacity = 1 - alpha;
            });
            document.querySelectorAll('.background-svg.t1w').forEach(el => {
                el.style.opacity = alpha;
            });
        }
        </script>
        """
    )

    return svg

def compute_vector(point1, point2):
    vector = np.array(point1) - np.array(point2)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.array([0, 0, 0])
    return vector / norm

def find_slice_axes(contact_fcsv_planned_path):
    # find vector between enterance and exit for each label
    # return dict with label with the slice dir in coronal view for each label
    #{label: coronal view slice axis}
    points_dict = {}
    slice_dict = {}
    with open(contact_fcsv_planned_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            label = row[11]
            try:
                x, y, z = float(row[1]), float(row[2]), float(row[3])
            except ValueError:
                continue
            if label not in points_dict:
                points_dict[label] = []
            points_dict[label].append((x, y, z))
                
    for label, points in points_dict.items():
        if len(points) != 2:
            print(f"Warning: label '{label}' has {len(points)} points, expected 2")
            continue
        entry, exit = points
        vector = compute_vector(entry, exit)
        slice_dict[label] = {
            'entry': entry,
            'exit': exit,
            'vector': vector
        }

    print("number of unique labels:", len(slice_dict))
    return slice_dict

def extract_oblique_slice(nifti_img, entry_world, exit_world, slice_thickness=1.0, num_points=256):
    """
    Extract a 2D oblique slice from a 3D NIfTI image along the line between entry_world and exit_world.
    Both points must be in WORLD (mm) coordinates.
    """
    data = nifti_img.get_fdata()
    affine = nifti_img.affine
    inv_affine = np.linalg.inv(affine)

    # Direction vector (normalized)
    entry_world = np.array(entry_world)
    exit_world = np.array(exit_world)
    direction = exit_world - entry_world
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("Entry and exit points are identical.")
    direction /= length

    # Create orthonormal basis: u, v, w
    u = direction
    temp = np.array([1, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1, 0])
    v = np.cross(u, temp)
    v /= np.linalg.norm(v)
    w = np.cross(u, v)

    # Define grid in world space
    center = (entry_world + exit_world) / 2.0
    grid_v = np.linspace(-slice_thickness * num_points / 2,
                         slice_thickness * num_points / 2, num_points)
    grid_w = np.linspace(-slice_thickness * num_points / 2,
                         slice_thickness * num_points / 2, num_points)
    vv, ww = np.meshgrid(grid_v, grid_w)

    # Construct world coordinates of the plane
    coords_world = center[:, None, None] + (v[:, None, None] * vv) + (w[:, None, None] * ww)
    coords_world = coords_world.reshape(3, -1)

    # Convert world → voxel
    coords_voxel = nib.affines.apply_affine(inv_affine, coords_world.T).T

    # Sample from image
    slice_values = map_coordinates(data, coords_voxel, order=1, mode='nearest')
    slice_img = slice_values.reshape((num_points, num_points))

    return slice_img


def group_coords(contact_fcsv_labelled_path):
    # returns a dictionary with each coords grouped together
    # {label: 10 tuples of x,y,z}
    coords_dict = {}
    with open(contact_fcsv_labelled_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            label = row[11].split('-')[0]

            # also need to save label num 
            # label_num = row[11]
            try:
                x, y, z = float(row[1]), float(row[2]), float(row[3])
            except ValueError:
                continue

            if label not in coords_dict:
                coords_dict[label] = []

            coords_dict[label].append((x, y, z))
    print("number of unique labels: ", len(coords_dict))
    return coords_dict

def output_html_file(ct_img_path,t1w_img_path,contact_fcsv_planned_path,contact_fcsv_labelled_path,output_html):

    # Load CT image
    ct_img = nib.load(ct_img_path)
    ct_img = nib.as_closest_canonical(ct_img)

    # t1w img
    t1w_img = nib.load(t1w_img_path)
    t1w_img = nib.as_closest_canonical(t1w_img)

    slice_info = find_slice_axes(contact_fcsv_planned_path)
    coords = group_coords(contact_fcsv_labelled_path)

    html_parts = []
    for i, (label, points) in enumerate(coords.items()):
        
        if i == 1:
            break

        middle_index = len(points) // 2
        middle_point = points[middle_index]
        print("middle point: ", middle_point)

        plot_args_ref = {"dim": -0.5} 
        plot_args_ct = {"dim": -0.5, "vmin": 0, "vmax": 2000} 

        # For CT + contacts overlays (background group 1)
        display_x_ct_contacts = plotting.plot_anat(ct_img, display_mode="x", draw_cross=False, cut_coords=[middle_point[0]], **plot_args_ct)
        display_x_ct_contacts.add_markers(points, marker_color="orange", marker_size=3)
        bg_x_ct_contacts_svgs = [fromstring(extract_svg(display_x_ct_contacts, 300))]
        display_x_ct_contacts.close()

        display_y_ct_contacts = plotting.plot_anat(ct_img, display_mode="y", draw_cross=False, cut_coords=[middle_point[1]], **plot_args_ct)
        display_y_ct_contacts.add_markers(points, marker_color="orange", marker_size=3)
        bg_y_ct_contacts_svgs = [fromstring(extract_svg(display_y_ct_contacts, 300))]
        display_y_ct_contacts.close()

        display_z_ct_contacts = plotting.plot_anat(ct_img, display_mode="z", draw_cross=False, cut_coords=[middle_point[2]], **plot_args_ct)
        display_z_ct_contacts.add_markers(points, marker_color="orange", marker_size=3)
        bg_z_ct_contacts_svgs = [fromstring(extract_svg(display_z_ct_contacts, 300))]
        display_z_ct_contacts.close()

        # For T1w + contacts overlays (background group 2)
        display_x_t1w_contacts = plotting.plot_anat(t1w_img, display_mode="x", draw_cross=False, cut_coords=[middle_point[0]], **plot_args_ref)
        display_x_t1w_contacts.add_markers(points, marker_color="orange", marker_size=3)
        bg_x_t1w_contacts_svgs = [fromstring(extract_svg(display_x_t1w_contacts, 300))]
        display_x_t1w_contacts.close()

        display_y_t1w_contacts = plotting.plot_anat(t1w_img, display_mode="y", draw_cross=False, cut_coords=[middle_point[1]], **plot_args_ref)
        display_y_t1w_contacts.add_markers(points, marker_color="orange", marker_size=3)
        bg_y_t1w_contacts_svgs = [fromstring(extract_svg(display_y_t1w_contacts, 300))]
        display_y_t1w_contacts.close()

        display_z_t1w_contacts = plotting.plot_anat(t1w_img, display_mode="z", draw_cross=False, cut_coords=[middle_point[2]], **plot_args_ref)
        display_z_t1w_contacts.add_markers(points, marker_color="orange", marker_size=3)
        bg_z_t1w_contacts_svgs = [fromstring(extract_svg(display_z_t1w_contacts, 300))]
        display_z_t1w_contacts.close()

        # Trajectory plotting 
        if label in slice_info:
            entry_world = np.array(slice_info[label]['entry'])
            exit_world = np.array(slice_info[label]['exit'])
            entry_voxel = nib.affines.apply_affine(np.linalg.inv(ct_img.affine), entry_world)
            exit_voxel = nib.affines.apply_affine(np.linalg.inv(ct_img.affine), exit_world)

            # Extract oblique slice
            slice_img_ct = extract_oblique_slice(ct_img, entry_voxel, exit_voxel)
            slice_img_t1w = extract_oblique_slice(t1w_img, entry_voxel, exit_voxel)
            
            # Convert to fake NIfTI for plotting in nilearn
            affine = np.eye(4)
            fake_img_ct = nib.Nifti1Image(slice_img_ct.T[:, :, np.newaxis], affine)
            fake_img_t1w = nib.Nifti1Image(slice_img_t1w.T[:, :, np.newaxis], affine)

            # Plot both using nilearn
            plot_args_oblique = {"dim": -0.5, "vmin": 0, "vmax": 2000}

            display_oblique_ct = plotting.plot_anat(fake_img_ct, display_mode="z", draw_cross=False, cut_coords=[slice_img_ct.shape[1] // 2], **plot_args_oblique)
            svg_oblique_ct = [fromstring(extract_svg(display_oblique_ct, dpi=300))]
            display_oblique_ct.close()

            display_oblique_t1w = plotting.plot_anat(fake_img_t1w, display_mode="z", draw_cross=False, cut_coords=[slice_img_ct.shape[1] // 2], dim=-0.5)
            svg_oblique_t1w = [fromstring(extract_svg(display_oblique_t1w, dpi=300))]
            display_oblique_t1w.close()

            # Combine SVGs for slider blending
            final_svg_oblique = "\n".join(clean_svg(svg_oblique_ct, svg_oblique_t1w))

        final_svg_x = "\n".join(clean_svg(bg_x_ct_contacts_svgs, bg_x_t1w_contacts_svgs))
        final_svg_y = "\n".join(clean_svg(bg_y_ct_contacts_svgs, bg_y_t1w_contacts_svgs))
        final_svg_z = "\n".join(clean_svg(bg_z_ct_contacts_svgs, bg_z_t1w_contacts_svgs))
        # will need to include the label in the header
        html_parts.append(f"""
            <div style="margin-bottom: 40px;">
                <p style="font-size:20px;"><b>{label}</b></p>
                <div style="display: flex; justify-content: center; gap: 10px;">
                    <div style="width: 400px;">{final_svg_x}</div>
                    <div style="width: 400px;">{final_svg_y}</div>
                    <div style="width: 400px;">{final_svg_z}</div>
                    <div style="width: 400px;">{final_svg_oblique}</div>
                </div>
                <hr style="height:4px;border-width:0;color:black;background-color:black;margin-top:30px;">
            </div>
        """)

        print("finished label: ", label)
        i += 1
    
    with open(output_html, "w") as f:
        f.write(f"""
        <html><body>
            <center>
                <h3 style="font-size:42px">CT and T1w Img</h3>
                <p style="margin:20px;">
                    <label for="blendSlider" style="font-size:18px;">CT ↔ T1w:</label>
                    <input id="blendSlider" type="range" min="0" max="100" value="0" oninput="updateBlend(this)" style="width: 300px; vertical-align: middle;">
                </p>
                {''.join(html_parts)}
            </center>
        </body></html>
        """)

    print(f"HTML output saved to {output_html}")

if __name__ == "__main__":
    ct_img_path = snakemake.input["ct_img"]
    t1w_img_path = snakemake.input["t1w_img"]
    contact_fcsv_planned_path = snakemake.input["contact_fcsv_planned"]
    contact_fcsv_labelled_path = snakemake.input["contact_fcsv_labelled"]
    output_html = snakemake.output["html"]

    output_html_file(ct_img_path,t1w_img_path,contact_fcsv_planned_path,contact_fcsv_labelled_path,output_html)