# Minimal vendored subset of cocodataset/panopticapi (https://github.com/cocodataset/panopticapi).
#
# DVIS_Plus's dataset mappers and panoptic-eval modules do
# ``from panopticapi.utils import rgb2id, IdGenerator`` at import time. The
# inf_server inference path never touches panoptic dataset I/O or evaluation, but
# those imports run when the DVIS packages are loaded. panopticapi is git-only
# (not on PyPI), so we vendor just the handful of pure-python helpers needed to
# satisfy the imports rather than pull in the whole repo. See ``utils.py``.
