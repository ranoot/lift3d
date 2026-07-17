"""inf_server: ZeroMQ/msgpack IPC server around the OV-DVIS++ online model.

Mirrors ``mask3d_feat/main.py`` but uses a ``zmq.REP`` socket (the correct role
for a request/reply server -- mask3d_feat binds a ``REQ`` socket, which is the
wrong pattern) and a structured msgpack protocol (see ``protocol.py``).

Run:  uv run python main.py  [--endpoint ipc:///tmp/inf_server.ipc] [--device cuda]
"""

from __future__ import annotations

import argparse

import zmq

import protocol
from dvis_runner import DEFAULT_CONFIG, DEFAULT_WEIGHTS, DvisRunner

DEFAULT_ENDPOINT = "ipc:///tmp/inf_server.ipc"

# ---- tuning knobs (edit these) --------------------------------------------
# The runner resolves each frame with VPS panoptic assignment on top of the
# FC-CLIP geometric ensemble (see dvis_runner.py). These are the levers for
# recall of open-vocabulary / novel classes (signs, fire extinguishers, ...).
OBJECT_MASK_THRESH = 0.10  # VPS keep gate: drop queries whose winning class score <= this
OVERLAP_THRESH = 0.7      # per-segment stability: min (kept / original) mask-area ratio
ENSEMBLE_ALPHA = 0.5      # geometric-ensemble weight for SEEN (in-train) classes
ENSEMBLE_BETA = 1.3       # geometric-ensemble weight for UNSEEN classes; raise toward 1.0
                          # to trust CLIP more for novel classes (sign, extinguisher, ...)

# Per-class synonyms. Key = the canonical class name EXACTLY as spelled in run.yaml's
# `thing:`/`stuff:` list; value = extra phrasings. The model embeds every synonym and keeps
# the best-matching one per pixel (max-ensemble), so more phrasings only raise recall.
# Classes not listed here are used as-is. These strings stay inside Python -- the C++ side
# never sees them, so object labels / topics / dynamic-class matching keep the clean name.
SYNONYMS: dict[str, list[str]] = {
    "extinguisher": ["fire extinguisher", "red fire extinguisher", "wall-mounted extinguisher"],
    "sign":         ["label", "label with text", "signage", "placard", "wall sign", "paper", "paper with text", "Plaque with text", "Sticker with text", "Neon Sign", "Signpost", "sign with text"],
}
# ---------------------------------------------------------------------------


def handle(runner: DvisRunner, req: dict) -> dict:
    cmd = req.get("cmd")
    if cmd == "ping":
        return {"ok": True}
    if cmd == "set_vocab":
        n = runner.set_vocab(
            list(req.get("thing_classes", [])),
            list(req.get("stuff_classes", [])),
        )
        return {"ok": True, "num_classes": n}
    if cmd == "reset":
        runner.reset()
        return {"ok": True}
    if cmd == "frame":
        return {"ok": True, **runner.infer(req["image"])}
    return {"ok": False, "error": f"unknown cmd: {cmd!r}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    args = ap.parse_args()

    runner = DvisRunner(
        weights_path=args.weights,
        config_file=args.config,
        device=args.device,
        object_mask_thresh=OBJECT_MASK_THRESH,
        overlap_thresh=OVERLAP_THRESH,
        ensemble_alpha=ENSEMBLE_ALPHA,
        ensemble_beta=ENSEMBLE_BETA,
        synonyms=SYNONYMS,
    )

    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.REP)
    socket.bind(args.endpoint)
    print(f"inf_server listening on {args.endpoint} (device={args.device})", flush=True)

    try:
        while True:
            req = protocol.unpack(socket.recv())
            try:
                reply = handle(runner, req)
            except Exception as exc:  # never let a bad request kill the server
                import traceback

                traceback.print_exc()
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            socket.send(protocol.pack(reply))
    except KeyboardInterrupt:
        pass
    finally:
        socket.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
