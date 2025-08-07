#!/usr/bin/env python3
from pathlib import Path
import sys
from snakebids import bidsapp, plugins
import os

app = bidsapp.app(
    [
        plugins.SnakemakeBidsApp(Path(__file__).resolve().parent),
        plugins.BidsValidator(),
        plugins.Version(distribution="nnUNet_contact_seg"),
    ]
)

def get_parser():
    """Exposes parser for sphinx doc generation, cwd is the docs dir."""
    return app.build_parser().parser


if __name__ == "__main__":
    app.run()
