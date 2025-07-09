def get_final_output():
    final = []
    final.extend(
        inputs["post_ct"].expand(
            bids(
                root=config["output_dir"],
                suffix="nnUNet.fcsv",
                **inputs["post_ct"].wildcards,
            )
        )
    )
    if config["label"]:
        final.extend(
            inputs["post_ct"].expand(
                bids(
                    root=config["output_dir"],
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
                    suffix="transformed_nnUNet.fcsv",
                    **inputs["post_ct"].wildcards,
                )
            )
        )
    # if config["reg_qc"]:
    #     final.extend(
    #         inputs["post_ct"].expand(
    #             bids(
    #                 root=config["output_dir"],
    #                 datatype="qc",
    #                 desc="reg_qc",
    #                 suffix=".html",
    #                 **inputs["post_ct"].wildcards,
    #             )
    #         )
    #     )
    if config["contacts_qc"]:
        final.extend(
            inputs["post_ct"].expand(
                bids(
                    root=config["output_dir"],
                    datatype="qc",
                    desc="contacts_qc",
                    suffix=".html",
                    **inputs["post_ct"].wildcards,
                )
            )
        )
    return final
