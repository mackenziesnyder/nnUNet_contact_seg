def get_reg_matrix():
    if not config["manual_reg_matrix"]:
        return bids(
            root=config["output_dir"],
            datatype="registration",
            space="T1w",
            suffix="xfm.txt",
            **inputs["post_ct"].wildcards,
        )
    else:
        return bids(
            root=config["bids_dir"],
            suffix="xfm.txt",
            **inputs["post_ct"].wildcards,
        )

def get_registered_ct_image():
    if not config["manual_reg_matrix"]:
        return bids(
            root=config["output_dir"],
            datatype="registration",
            space="T1w",
            suffix="ct.nii.gz",
            **inputs["post_ct"].wildcards,
        ),
    else:
        return bids(
            root=config["bids_dir"],
            suffix="ct",
            datatype="ct",
            session="post",
            acq="Electrode",
            extension=".nii.gz",
            **inputs["post_ct"].wildcards,
        )

def get_final_output():
    final = []
    final.extend(
        inputs["post_ct"].expand(
            bids(
                root=config["output_dir"],
                suffix="dseg.nii.gz",
                datatype="contact_seg",
                space="T1w",
                desc="contacts_nnUNet",
                **inputs["post_ct"].wildcards,
            )
        )
    )
    if config["label"]:
        final.extend(
            inputs["post_ct"].expand(
                bids(
                    root=config["output_dir"],
                    datatype="coords",
                    suffix="labelled_nnUNet.fcsv",
                    **inputs["post_ct"].wildcards,
                )
            )
        )
    if config["transform"]:
        final.extend(
            inputs["post_ct"].expand(
                bids(
                    root=config["output_dir"],
                    datatype="coords",
                    suffix="transformed_nnUNet.fcsv",
                    **inputs["post_ct"].wildcards,
                )
            )
        )
    if config["contacts_qc"]:
        final.extend(
            inputs["post_ct"].expand(
                bids(
                    root=config["output_dir"],
                    datatype="qc",
                    desc="contacts_qc",
                    suffix="contacts.html",
                    **inputs["post_ct"].wildcards,
                )
            )
        )
    return final
