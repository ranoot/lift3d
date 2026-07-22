"""Wire protocol for the lift3d viz channel (run_semantic_universe -> viz_server).

Identical ndarray envelope to ``inf_server/protocol.py`` -- a numpy array rides as
``{"__ndarray__": True, "shape": [...], "dtype": "<str>", "data": <raw bytes>}``, so the
C++ VizPublisher (msgpack_lite.h) and this Python consumer agree byte-for-byte and numpy
arrays rebuild without a python-list round trip. Message catalogue (keyed by ``type``):

  {"type": "begin", "classes": [{"id": int, "name": str, "kind": int}, ...],  # Universe vocab
                    "seg_vocab": [str, ...],   # InfClient vocab: index == seg label_map class id
                    "seg_overlay": str, "seg_alpha": float, "seg_min_area": int}

  {"type": "frame", "idx": int, "capture_secs": float,
                    "robot_pose": <f4 (16,)>, "robot": <f4 (3,)>,
                    "cam": {"K": <f4 (9,)>, "T": <f4 (16,)>, "w": int, "h": int},
                    "image": <u1 (H,W,3)>,               # optional
                    "seg": {"label_map": <i2 (H,W)>, "id_map": <i2 (H,W)>},  # optional
                    "objects":     [entity, ...],        # optional
                    "proposals":   [entity, ...],        # optional
                    "superpoints": [entity+kind, ...]}   # optional, only on VCCS refresh
      entity = {"id": int, "class_id": int, "name": str,
                "centroid": <f4 (3,)>, "points": <f4 (M,3)>}   # (+ "kind": int for superpoints)
      objects may also carry "text": str -- free text attached to that object (today the
      sign-understanding result), drawn as a label at its centroid.

  {"type": "finish", "positions": <f4 (N,3)>, "class_ids": <i2 (N,)>, "kinds": <i1 (N,)>}
"""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np

_NDARRAY_TAG = "__ndarray__"


def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {
            _NDARRAY_TAG: True,
            "shape": list(arr.shape),
            "dtype": arr.dtype.str,
            "data": arr.tobytes(),
        }
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"cannot serialize {type(obj)!r}")


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get(_NDARRAY_TAG):
        arr = np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"]))
        return arr.reshape(obj["shape"])
    return obj


def pack(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_encode, use_bin_type=True)


def unpack(buf: bytes) -> Any:
    return msgpack.unpackb(buf, object_hook=_decode, raw=False, strict_map_key=False)
