#!/usr/bin/env python3
#using afids/afids-auto/afids-auto-train/workflow/scripts/reg_qc.py script
# -*- coding: utf-8 -*-

import re
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import nibabel as nib
from nibabel.orientations import aff2axcodes
import numpy as np
import csv
from nilearn import plotting
from scipy.ndimage import binary_dilation
from svgutils.compose import Unit
from svgutils.transform import GroupElement, SVGFigure, fromstring

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

def load_fcsv_points(fcsv_path):
    points = []
    with open(fcsv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(points) >= 10:
                break
            if not row or row[0].startswith('#'):
                continue
            if len(row) < 4 or not row[0].isdigit():
                continue
            try:
                x, y, z = float(row[1]), float(row[2]), float(row[3])
                points.append((x, y, z))
            except ValueError:
                continue
    print("points: ", points)
    print("number of points: ", len(points))
    return np.array(points)

def output_html_file(ct_img_path, t1w_img_path, contacts_path, contact_fcsv_labelled_path, output_html):
    """Processes a single subject's images and generates an HTML output."""

    # Load CT image
    ct_img = nib.load(ct_img_path)
    ct_img = nib.as_closest_canonical(ct_img)

    # Load contact segmentation
    contact_img = nib.load(contacts_path)
    contact_img = nib.as_closest_canonical(contact_img)

    # Convert to binary mask
    contact_data = contact_img.get_fdata() > 0

    # Dilate the mask (3x3x3 cube)
    dilated_data = binary_dilation(contact_data, iterations=2)

    # Create a new NIfTI image with same affine
    dilated_contact_img = nib.Nifti1Image(dilated_data.astype(np.float32), affine=contact_img.affine)

    # t1w img
    t1w_img = nib.load(t1w_img_path)
    t1w_img = nib.as_closest_canonical(t1w_img)

    print("CT orientation:", aff2axcodes(ct_img.affine))
    print("T1w orientation:", aff2axcodes(t1w_img.affine))
    print("contact fcsv path: ", contact_fcsv_labelled_path)
    
    # contacts 
    points = load_fcsv_points(contact_fcsv_labelled_path)
    cut_coords_x = [float(coord[0]) for coord in points]
    cut_coords_y = [float(coord[1]) for coord in points]
    cut_coords_z = [float(coord[2]) for coord in points]
    print("x-coords: ", cut_coords_x)
    print("y-coords: ", cut_coords_y)
    print("z-coords: ", cut_coords_z)

    # load the images from file paths
    print("CT shape / affine:", ct_img.shape, ct_img.affine)
    print("T1w shape / affine:", t1w_img.shape, t1w_img.affine)
    print("Contact shape / affine:", contact_img.shape, contact_img.affine)

    plot_args_ref = {"dim": -0.5} 
    plot_args_ct = {"dim": -0.5, "vmin": 0, "vmax": 2000}    

    # For CT + contacts overlays (background group 1)
    display_x_ct_contacts = plotting.plot_anat(ct_img, display_mode="x", draw_cross=False, cut_coords=cut_coords_x, **plot_args_ct)
    display_x_ct_contacts.add_overlay(dilated_contact_img, cmap="autumn")
    bg_x_ct_contacts_svgs = [fromstring(extract_svg(display_x_ct_contacts, 300))]
    display_x_ct_contacts.close()

    display_y_ct_contacts = plotting.plot_anat(ct_img, display_mode="y", draw_cross=False, cut_coords=cut_coords_y, **plot_args_ct)
    display_y_ct_contacts.add_overlay(dilated_contact_img, cmap="autumn")
    bg_y_ct_contacts_svgs = [fromstring(extract_svg(display_y_ct_contacts, 300))]
    display_y_ct_contacts.close()

    display_z_ct_contacts = plotting.plot_anat(ct_img, display_mode="z", draw_cross=False, cut_coords=cut_coords_z, **plot_args_ct)
    display_z_ct_contacts.add_overlay(dilated_contact_img, cmap="autumn")
    bg_z_ct_contacts_svgs = [fromstring(extract_svg(display_z_ct_contacts, 300))]
    display_z_ct_contacts.close()

    # For T1w + contacts overlays (background group 2)
    display_x_t1w_contacts = plotting.plot_anat(t1w_img, display_mode="x", draw_cross=False, cut_coords=cut_coords_x, **plot_args_ref)
    display_x_t1w_contacts.add_overlay(dilated_contact_img, cmap="autumn")
    bg_x_t1w_contacts_svgs = [fromstring(extract_svg(display_x_t1w_contacts, 300))]
    display_x_t1w_contacts.close()

    display_y_t1w_contacts = plotting.plot_anat(t1w_img, display_mode="y", draw_cross=False, cut_coords=cut_coords_y, **plot_args_ref)
    display_y_t1w_contacts.add_overlay(dilated_contact_img, cmap="autumn")
    bg_y_t1w_contacts_svgs = [fromstring(extract_svg(display_y_t1w_contacts, 300))]
    display_y_t1w_contacts.close()

    display_z_t1w_contacts = plotting.plot_anat(t1w_img, display_mode="z", draw_cross=False, cut_coords=cut_coords_z, **plot_args_ref)
    display_z_t1w_contacts.add_overlay(dilated_contact_img, cmap="autumn")
    bg_z_t1w_contacts_svgs = [fromstring(extract_svg(display_z_t1w_contacts, 300))]
    display_z_t1w_contacts.close()

    final_svg_x = "\n".join(clean_svg(bg_x_ct_contacts_svgs, bg_x_t1w_contacts_svgs))
    final_svg_y = "\n".join(clean_svg(bg_y_ct_contacts_svgs, bg_y_t1w_contacts_svgs))
    final_svg_z = "\n".join(clean_svg(bg_z_ct_contacts_svgs, bg_z_t1w_contacts_svgs))

    #display results
    with open(output_html, "w") as f:
        f.write(f"""
            <html><body>
                <center>
                    <h3 style="font-size:42px">CT and T1w Img</h3>
                    <p style="margin:20px;">
                        <label for="blendSlider" style="font-size:18px;">CT ↔ T1w:</label>
                        <input id="blendSlider" type="range" min="0" max="100" value="0" oninput="updateBlend(this)" style="width: 300px; vertical-align: middle;">
                    </p>
                    <p>{final_svg_x}</p>
                    <p>{final_svg_y}</p>
                    <p>{final_svg_z}</p>
                    <hr style="height:4px;border-width:0;color:black;background-color:black;margin:30px;">
                </center>
            </body></html>
        """)

    print(f"HTML output saved to {output_html}")

if __name__ == "__main__":
    ct_img_path = snakemake.input["ct_img"]
    t1w_img_path = snakemake.input["t1w_img"]
    contact_seg_path = snakemake.input["contact_seg"]
    contact_fcsv_labelled_path = snakemake.input["contact_fcsv_labelled"]
    output_html = snakemake.output["html"]

    output_html_file(ct_img_path,t1w_img_path,contact_seg_path,contact_fcsv_labelled_path,output_html)