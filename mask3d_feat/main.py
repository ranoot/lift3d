"""Mask3D backbone feature server (IPC).

We only need the Res16UNet34C *backbone* to turn a sparse voxel cloud into
per-voxel feature vectors -- never the full Mask3D head. Two things make that
awkward, both handled below before any Mask3D import:

  1. Mask3D's ``models/__init__.py`` eagerly does ``from models.mask3d import
     Mask3D``, and ``models/mask3d.py`` imports ``hydra``, ``torch_scatter`` and
     the compiled ``third_party.pointnet2`` ext at module load. None of those are
     installed in this cu118 venv and none are needed for the backbone, so we
     pre-seed lightweight stubs in ``sys.modules`` -- the real mask3d.py still
     imports (keeping the package __init__ intact) but its heavy deps are inert.
  2. Res16UNet34C requires a ``config`` object (attribute access for
     ``bn_momentum`` / ``conv1_kernel_size`` / ``dilations``). We supply a minimal
     SimpleNamespace with the ScanNet backbone values from conf/model/mask3d.yaml.

Wire protocol (msgpack, REP socket) mirrors inf_server/protocol.py's ndarray
envelope. NOTE: MinkowskiEngine coalesces duplicate voxel coordinates, so the
returned feature count M may be < N; the response includes the output coords so
the caller can align features back to voxels.

  {"cmd": "ping"}
      -> {"ok": True}
  {"cmd": "features", "coords": <int32 (N,3)>, "feats": <float32 (N,C)>}
      -> {"ok": True, "coords": <int32 (M,4)>, "feats": <float32 (M,96)>}
  any handler error -> {"ok": False, "error": "<message>"}
"""

import os
import sys
import types
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any

import msgpack
import numpy as np
import torch
import zmq

HERE = os.path.dirname(os.path.abspath(__file__))
MASK3D_DIR = os.path.join(HERE, "Mask3D")
CKPT = os.path.join(MASK3D_DIR, "checkpoints", "scannet_val.ckpt")
IPC_ENDPOINT = "ipc:///tmp/mask3d_feat.ipc"
IN_CHANNELS = 3  # matches the scannet_val backbone (0 missing/unexpected keys)

# --- neutralize Mask3D's heavy, backbone-irrelevant imports ------------------
sys.path.insert(0, MASK3D_DIR)


def _stub(name: str, **attrs) -> None:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


_noop = lambda *a, **k: None  # noqa: E731
_stub("hydra")
_stub("torch_scatter", scatter_mean=_noop, scatter_max=_noop, scatter_min=_noop)
_stub("third_party")
_stub("third_party.pointnet2")
_stub("third_party.pointnet2.pointnet2_utils", furthest_point_sample=_noop)

import MinkowskiEngine as ME  # noqa: E402
from models.res16unet import Res16UNet34C  # noqa: E402

# --- ndarray-envelope msgpack codec (same shape as inf_server/protocol.py) ---
_NDARRAY_TAG = "__ndarray__"


def _encode(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {_NDARRAY_TAG: True, "shape": list(arr.shape),
                "dtype": arr.dtype.str, "data": arr.tobytes()}
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"cannot serialize {type(obj)!r}")


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get(_NDARRAY_TAG):
        return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    return obj


def pack(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_encode, use_bin_type=True)


def unpack(buf: bytes) -> Any:
    return msgpack.unpackb(buf, object_hook=_decode, raw=False, strict_map_key=False)


def build_backbone() -> torch.nn.Module:
    # ScanNet backbone hyperparameters from conf/model/mask3d.yaml.
    cfg = SimpleNamespace(dilations=(1, 1, 1, 1), conv1_kernel_size=5, bn_momentum=0.02)
    bb = Res16UNet34C(in_channels=IN_CHANNELS, out_channels=20, config=cfg)

    ckpt = torch.load(CKPT, map_location="cpu")
    sd = ckpt["state_dict"]  # Lightning wraps weights here
    backbone_sd = OrderedDict(
        (k[len("model.backbone."):], v)
        for k, v in sd.items() if k.startswith("model.backbone.")
    )
    missing, unexpected = bb.load_state_dict(backbone_sd, strict=False)
    print(f"backbone loaded: {len(missing)} missing, {len(unexpected)} unexpected keys",
          flush=True)
    return bb.cuda().eval()


def handle(req: dict, backbone: torch.nn.Module) -> dict:
    cmd = req.get("cmd")
    if cmd == "ping":
        return {"ok": True}
    if cmd == "features":
        coords = np.ascontiguousarray(req["coords"], dtype=np.int32)  # (N,3)
        feats = np.ascontiguousarray(req["feats"], dtype=np.float32)  # (N,C)
        N = coords.shape[0]
        batched = np.hstack([np.zeros((N, 1), np.int32), coords])     # (N,4), batch 0
        x = ME.SparseTensor(
            features=torch.from_numpy(feats).cuda(),
            coordinates=torch.from_numpy(batched).cuda(),
        )
        with torch.no_grad():
            out = backbone(x)
        return {
            "ok": True,
            "coords": out.C.cpu().numpy().astype(np.int32),   # (M,4): batch + xyz
            "feats": out.F.cpu().numpy().astype(np.float32),  # (M,96)
        }
    return {"ok": False, "error": f"unknown cmd {cmd!r}"}


def main() -> None:
    backbone = build_backbone()
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)  # request/reply server binds a REP socket
    socket.bind(IPC_ENDPOINT)
    print(f"mask3d_feat backbone server listening on {IPC_ENDPOINT}", flush=True)
    try:
        while True:
            req = unpack(socket.recv())
            try:
                resp = handle(req, backbone)
            except Exception as e:  # keep the server alive on a bad request
                resp = {"ok": False, "error": repr(e)}
            socket.send(pack(resp))
    except KeyboardInterrupt:
        pass
    finally:
        socket.close()
        ctx.term()


if __name__ == "__main__":
    main()
