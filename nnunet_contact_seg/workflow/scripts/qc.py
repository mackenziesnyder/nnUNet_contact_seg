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


def clean_svg(fg_svgs, bg_svgs, ref=0):
    # Find and replace the figure_1 id.
    svgs = bg_svgs + fg_svgs
    roots = [f.getroot() for f in svgs]

    sizes = []
    for f in svgs:
        viewbox = [float(v) for v in f.root.get("viewBox").split(" ")]
        width = int(viewbox[2])
        height = int(viewbox[3])
        sizes.append((width, height))
    nsvgs = len([bg_svgs])

    sizes = np.array(sizes)

    # Calculate the scale to fit all widths
    width = sizes[ref, 0]
    scales = width / sizes[:, 0]
    heights = sizes[:, 1] * scales

    # Compose the views panel: total size is the width of
    # any element (used the first here) and the sum of heights
    fig = SVGFigure(Unit(f"{width}px"), Unit(f"{heights[:nsvgs].sum()}px"))

    yoffset = 0
    for i, r in enumerate(roots):
        r.moveto(0, yoffset, scale_x=scales[i])
        if i == (nsvgs - 1):
            yoffset = 0
        else:
            yoffset += heights[i]

    # Group background and foreground panels in two groups
    if fg_svgs:
        newroots = [
            GroupElement(roots[:nsvgs], {"class": "background-svg"}),
            GroupElement(roots[nsvgs:], {"class": "foreground-svg"}),
        ]
    else:
        newroots = roots

    fig.append(newroots)
    fig.root.attrib.pop("width", None)
    fig.root.attrib.pop("height", None)
    fig.root.set("preserveAspectRatio", "xMidYMid meet")

    with TemporaryDirectory() as tmpdirname:
        out_file = Path(tmpdirname) / "tmp.svg"
        fig.save(str(out_file))
        # Post processing
        svg = out_file.read_text().splitlines()

    # Remove <?xml... line
    if svg[0].startswith("<?xml"):
        svg = svg[1:]

    # Add styles for the flicker animation
    if fg_svgs:
        svg.insert(
            2,
            """\
<style type="text/css">
@keyframes flickerAnimation%s { 0%% {opacity: 1;} 100%% { opacity:0; }}
.foreground-svg { animation: 1s ease-in-out 0s alternate none infinite running flickerAnimation%s;}
.foreground-svg:hover { animation-play-state: running;}
</style>"""
            % tuple([uuid4()] * 2),
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
    plot_args_ct = {"dim": -0.5, "vmin": 0, "vmax": 100}  

    # Generate foreground and background images

    # ct as forground:
    display_x_ct = plotting.plot_anat(ct_img, display_mode="x", draw_cross=False, cut_coords=cut_coords_x, **plot_args_ct)
    display_x_ct.add_overlay(dilated_contact_img, cmap="autumn")
    fg_x_svgs = [fromstring(extract_svg(display_x_ct, 300))]
    display_x_ct.close()

    display_y_ct = plotting.plot_anat(ct_img, display_mode="y", draw_cross=False, cut_coords=cut_coords_y, **plot_args_ct)
    display_y_ct.add_overlay(dilated_contact_img, cmap="autumn")
    fg_y_svgs = [fromstring(extract_svg(display_y_ct, 300))]
    display_y_ct.close()

    # display_z_ct = plotting.plot_anat(ct_img, display_mode="z", draw_cross=False, cut_coords=cut_coords_z, **plot_args_ct)
    # display_z_ct.add_overlay(contact_img, cmap="autumn")
    # fg_z_svgs = [fromstring(extract_svg(display_z_ct, 300))]
    # display_z_ct.close()

    # t1w image as background
    display_x_t1w = plotting.plot_anat(t1w_img, display_mode="x", draw_cross=False, cut_coords=cut_coords_x, **plot_args_ref)
    # display_x_t1w.add_overlay(contact_img, cmap="autumn", alpha=1.0)
    bg_x_svgs = [fromstring(extract_svg(display_x_t1w, 300))]
    display_x_t1w.close()

    display_y = plotting.plot_anat(t1w_img, display_mode="y", draw_cross=False, cut_coords=cut_coords_y, **plot_args_ref)
    # plotting.plot_roi(contact_img, display=display_y, cmap="Reds", alpha=1.0)
    bg_y_svgs = [fromstring(extract_svg(display_y, 300))]
    display_y.close()

    # display_z = plotting.plot_anat(t1w_img, display_mode="z", draw_cross=False, **plot_args_ref)
    # # # plotting.plot_roi(contact_img, display=display_z, cmap="Reds", alpha=1.0)
    # bg_z_svgs = [fromstring(extract_svg(display_z, 300))]
    # display_z.close()

    # generate final SVGs by overlaying foreground and background
    final_svg_x = "\n".join(clean_svg(fg_x_svgs, bg_x_svgs))
    final_svg_y = "\n".join(clean_svg(fg_y_svgs, bg_y_svgs))
    # final_svg_z = "\n".join(clean_svg(fg_z_svgs, bg_z_svgs))

    #display results
    with open(output_html, "w") as f:
        f.write(f"""
            <html><body>
                <center>
                    <h3 style="font-size:42px">CT and T1w Img</h3>
                    <p>{final_svg_x}</p>
                    <p>{final_svg_y}</p>
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