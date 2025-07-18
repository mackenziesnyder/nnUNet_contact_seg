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

def clean_matplotlib_svgs(bg1_svgs_strs, bg2_svgs_strs, ref=0):
    """
    Clean and stack matplotlib-generated SVGs vertically with toggleable opacity.

    Args:
        bg1_svgs_strs: List of SVG strings for background 1 (e.g., CT).
        bg2_svgs_strs: List of SVG strings for background 2 (e.g., T1w).
        ref: Index to use as reference for width scaling.

    Returns:
        List of strings: Final cleaned SVG lines.
    """      
    bg1_svgs = [fromstring(s) for s in bg1_svgs_strs]
    bg2_svgs = [fromstring(s) for s in bg2_svgs_strs]

    svgs = bg1_svgs + bg2_svgs
    roots = [s.getroot() for s in svgs]

    sizes = []
    for f in svgs:
        vb = f.root.get("viewBox")
        if vb is None:
            raise ValueError("SVG missing viewBox attribute. Matplotlib SVGs must have it.")
        viewbox = list(map(float, vb.strip().split()))
        width = viewbox[2]
        height = viewbox[3]
        sizes.append((width, height))

    sizes = np.array(sizes)
    ref_width = sizes[ref, 0]
    scales = ref_width / sizes[:, 0]
    heights = sizes[:, 1] * scales

    nsvgs = len(bg1_svgs)
    fig = SVGFigure(Unit(f"{ref_width}px"), Unit(f"{heights[:nsvgs].sum()}px"))

    yoffset = 0
    for i, root in enumerate(roots):
        root.moveto(0, yoffset, scale_x=scales[i], scale_y=scales[i])
        if i < nsvgs - 1:
            yoffset += heights[i]

    newroots = [
        GroupElement(roots[:nsvgs], {"class": "background-svg ct"}),
        GroupElement(roots[nsvgs:], {"class": "background-svg t1w"}),
    ]

    fig.append(newroots)
    fig.root.attrib.pop("width", None)
    fig.root.attrib.pop("height", None)
    fig.root.set("preserveAspectRatio", "xMidYMid meet")

    # Save and inject styles/scripts
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
    return slice_dict

def extract_oblique_slice(nifti_img, entry, exit, width=40, num_points=256):
    data = nifti_img.get_fdata()
    affine = nifti_img.affine
    inv_affine = np.linalg.inv(affine)

    direction = exit - entry
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("Entry and exit points are the same.")

    u = direction / length  # Normalize trajectory direction

    # Create orthogonal vectors v, w for the plane
    arbitrary = np.array([1, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1, 0])
    v = np.cross(u, arbitrary)
    v /= np.linalg.norm(v)

    # Grid in v (width direction) and t (along trajectory)
    s_vals = np.linspace(-width/2, width/2, num_points)
    t_vals = np.linspace(-10, length + 10, num_points)  # add padding before/after

    S, T = np.meshgrid(s_vals, t_vals)

    # World coordinates of the slice points
    coords_world = entry[:, None, None] + T[None, :, :] * u[:, None, None] + S[None, :, :] * v[:, None, None]
    coords_world = coords_world.reshape(3, -1)

    # Convert to voxel space
    coords_voxel = nib.affines.apply_affine(inv_affine, coords_world.T).T

    # Clamp to CT bounds
    for i in range(3):
        coords_voxel[i] = np.clip(coords_voxel[i], 0, data.shape[i] - 1)

    slice_vals = map_coordinates(data, coords_voxel, order=1, mode='nearest')
    slice_img = slice_vals.reshape(num_points, num_points)

    # Return image and transformation info
    return slice_img, s_vals, t_vals, u, v, entry

def project_point_to_slice(P, origin, u, v):
    vec = np.array(P) - origin
    s = np.dot(vec, v)
    t = np.dot(vec, u)
    return s, t

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
            label_num = row[11].split('-')[1]
            try:
                x, y, z = float(row[1]), float(row[2]), float(row[3])
            except ValueError:
                continue

            if label not in coords_dict:
                coords_dict[label] = []

            coords_dict[label].append(((x, y, z), label_num))
    return coords_dict


def render_oblique_slice_to_svg(img, entry_world, exit_world, points, **args):
    
    slice_img, s_vals, t_vals, u, v, origin = extract_oblique_slice(img, entry_world, exit_world)

    # Project entry and exit to slice coords
    entry_st = project_point_to_slice(entry_world, origin, u, v)
    exit_st = project_point_to_slice(exit_world, origin, u, v)

    # Rotate image 90 degrees clockwise
    rotated_img = np.rot90(slice_img, k=-1)

    # Plot rotated image
    plt.imshow(rotated_img, cmap='gray', extent=[t_vals[0], t_vals[-1], s_vals[0], s_vals[-1]], **args)

    # Project entry and exit to rotated slice coords (s becomes y-axis, t becomes x-axis)
    plt.scatter(entry_st[1], entry_st[0], color='green', label='Entry Point', s=50)
    plt.scatter(exit_st[1], exit_st[0], color='red', label='Exit Point', s=50)

    for (pt, label) in points:
        s, t = project_point_to_slice(pt, origin, u, v)
        plt.scatter(t, s, color='orange', s=50)
        plt.text(t - 2, s - 2, str(label), color='purple', fontsize=12, weight='bold')

    plt.legend()
    plt.axis('off')
    plt.tight_layout(pad=0)
    svg_buffer = io.StringIO()
    plt.savefig(svg_buffer, format='svg', bbox_inches='tight', pad_inches=0)
    svg_str = svg_buffer.getvalue()
    svg_buffer.close()
    plt.close()
    svg_str = re.sub(r'<\?xml.*?\?>', '', svg_str, flags=re.DOTALL)
    svg_str = re.sub(r'<!DOCTYPE.*?>', '', svg_str, flags=re.DOTALL)
    svg_str = svg_str.lstrip()
    return svg_str

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
    for label, points in coords.items():

        middle_index = len(points) // 2
        middle_point = points[middle_index][0]

        plot_args_ref = {"dim": -0.5} 
        plot_args_ct = {"dim": -0.5, "vmin": 0, "vmax": 2000} 
        matpltlib_refs_t1w =  {} 
        matpltlib_refs_ct =  {"vmin": 0, "vmax": 2000} 

        
        marker_coords = [val[0] for val in points]
        # For CT + contacts overlays (background group 1)
        display_x_ct_contacts = plotting.plot_anat(ct_img, display_mode="x", draw_cross=False, cut_coords=[middle_point[0]], **plot_args_ct)
        display_x_ct_contacts.add_markers(marker_coords, marker_color="orange", marker_size=3)
        bg_x_ct_contacts_svgs = [fromstring(extract_svg(display_x_ct_contacts, 300))]
        display_x_ct_contacts.close()

        display_y_ct_contacts = plotting.plot_anat(ct_img, display_mode="y", draw_cross=False, cut_coords=[middle_point[1]], **plot_args_ct)
        display_y_ct_contacts.add_markers(marker_coords, marker_color="orange", marker_size=3)
        bg_y_ct_contacts_svgs = [fromstring(extract_svg(display_y_ct_contacts, 300))]
        display_y_ct_contacts.close()

        display_z_ct_contacts = plotting.plot_anat(ct_img, display_mode="z", draw_cross=False, cut_coords=[middle_point[2]], **plot_args_ct)
        display_z_ct_contacts.add_markers(marker_coords, marker_color="orange", marker_size=3)
        bg_z_ct_contacts_svgs = [fromstring(extract_svg(display_z_ct_contacts, 300))]
        display_z_ct_contacts.close()

        # For T1w + contacts overlays (background group 2)
        display_x_t1w_contacts = plotting.plot_anat(t1w_img, display_mode="x", draw_cross=False, cut_coords=[middle_point[0]], **plot_args_ref)
        display_x_t1w_contacts.add_markers(marker_coords, marker_color="orange", marker_size=3)
        bg_x_t1w_contacts_svgs = [fromstring(extract_svg(display_x_t1w_contacts, 300))]
        display_x_t1w_contacts.close()

        display_y_t1w_contacts = plotting.plot_anat(t1w_img, display_mode="y", draw_cross=False, cut_coords=[middle_point[1]], **plot_args_ref)
        display_y_t1w_contacts.add_markers(marker_coords, marker_color="orange", marker_size=3)
        bg_y_t1w_contacts_svgs = [fromstring(extract_svg(display_y_t1w_contacts, 300))]
        display_y_t1w_contacts.close()

        display_z_t1w_contacts = plotting.plot_anat(t1w_img, display_mode="z", draw_cross=False, cut_coords=[middle_point[2]], **plot_args_ref)
        display_z_t1w_contacts.add_markers(marker_coords, marker_color="orange", marker_size=3)
        bg_z_t1w_contacts_svgs = [fromstring(extract_svg(display_z_t1w_contacts, 300))]
        display_z_t1w_contacts.close()

        if label in slice_info:
            entry_world = np.array(slice_info[label]['entry'])
            exit_world = np.array(slice_info[label]['exit'])
            
            # SVGs with overlays
            svg_oblique_ct = render_oblique_slice_to_svg(ct_img, entry_world, exit_world, points, **matpltlib_refs_ct)
            svg_oblique_t1w = render_oblique_slice_to_svg(t1w_img, entry_world, exit_world, points, **matpltlib_refs_t1w)

            # Combine SVGs for slider or side-by-side
            final_svg_oblique = "\n".join(clean_matplotlib_svgs([svg_oblique_ct], [svg_oblique_t1w]))

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