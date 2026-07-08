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
        weights_path=args.weights, config_file=args.config, device=args.device
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
