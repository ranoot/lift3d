"""Wire protocol for the inf_server IPC channel.

All messages are msgpack maps. Numpy arrays ride as a self-describing envelope
``{"__ndarray__": True, "shape": [...], "dtype": "<str>", "data": <raw bytes>}``
so the peer (Python here, C++/cppzmq on the consumer side) can rebuild the buffer
from a known dtype + shape without paying for a python-list round trip. This is
the same idea as ``mask3d_feat/main.py``'s ``pack_tensor`` but it keeps the data
as raw little-endian bytes instead of a flattened list (and without that file's
``.to_list()`` typo).

Request schema (client -> server), keyed by ``cmd``:

  {"cmd": "ping"}
      -> {"ok": True}

  {"cmd": "set_vocab", "thing_classes": [str, ...], "stuff_classes": [str, ...]}
      Build/refresh the model + open-vocab text classifier for this vocabulary
      and start a fresh tracking sequence.
      -> {"ok": True, "num_classes": int}

  {"cmd": "reset"}
      Begin a new tracking sequence (clears the tracker's memory; the next frame
      is treated as the first frame of a video).
      -> {"ok": True}

  {"cmd": "frame", "image": <ndarray uint8 (H, W, 3) RGB>}
      Run one frame through the stateful online model. The server owns
      normalization (PIXEL_MEAN/STD) and network-side resize; the returned maps
      are at the *input* (H, W) so they line up with the caller's camera image.
      -> {"ok": True, "h": H, "w": W,
          "label_map": <ndarray int16 (H, W)>,   # class id per pixel, -1 = background
          "id_map":    <ndarray int16 (H, W)>,   # instance/query id per pixel, -1 = bg
          "instances": [{"id": int, "label": int, "score": float,
                          "embedding": <ndarray float32 (C,)>}, ...]}

Any handler error returns ``{"ok": False, "error": "<message>"}``.
"""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np

_NDARRAY_TAG = "__ndarray__"


def _encode(obj: Any) -> Any:
    """msgpack ``default`` hook: turn numpy arrays/scalars into plain types."""
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {
            _NDARRAY_TAG: True,
            "shape": list(arr.shape),
            "dtype": arr.dtype.str,  # e.g. "<i2", "<f4"
            "data": arr.tobytes(),
        }
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"cannot serialize {type(obj)!r}")


def _decode(obj: Any) -> Any:
    """msgpack ``object_hook``: rebuild numpy arrays from the envelope."""
    if isinstance(obj, dict) and obj.get(_NDARRAY_TAG):
        arr = np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"]))
        return arr.reshape(obj["shape"])
    return obj


def pack(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_encode, use_bin_type=True)


def unpack(buf: bytes) -> Any:
    return msgpack.unpackb(
        buf, object_hook=_decode, raw=False, strict_map_key=False
    )
