# BIDs app for localizing stereoelectroencephalography (SEEG) contact from post-operative CT scans.

Uses the nnUNetv2 framework (https://github.com/MIC-DKFZ/nnUNet) to train a 3D U-Net model. Incorporates the Snakemake/SnakeBIDS workflow management tools.

Project is managed with uv (https://github.com/astral-sh/uv).

## uv Installation

Install uv with our standalone installers:

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or, from [PyPI](https://pypi.org/project/uv/):

```bash
# With pip.
pip install uv
```

```bash
# Or pipx.
pipx install uv
```
## Using BIDs app
try 'uv run' followed by SnakeBIDs/BIDs app syntax
```bash
# On macOS and Linux.
uv run run.py {/path/to/bids/dir} {/path/to/derivatives/dir} 
```