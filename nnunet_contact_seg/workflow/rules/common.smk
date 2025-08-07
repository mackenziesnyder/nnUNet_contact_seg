import os
from appdirs import AppDirs

def get_download_dir():
    if "nnUNet_SEEG_CACHE_DIR" in os.environ.keys():
        download_dir = os.environ["nnUNet_SEEG_CACHE_DIR"]
    else:
        # create local download dir if it doesn't exist
        dirs = AppDirs("nnUNet", "arun")
        download_dir = dirs.user_cache_dir
    return download_dir

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


def get_final_output():
    final = []
    final.extend(
        inputs["post_ct"].expand(
            bids(
                root=config["output_dir"],
                suffix="nnUNet.fcsv",
                datatype="coords",
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
    return final
