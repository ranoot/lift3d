"""Tiny IPC client to smoke-test a running inf_server."""
import sys
import numpy as np
import zmq
import protocol

ep = sys.argv[1] if len(sys.argv) > 1 else "ipc:///tmp/inf_server.ipc"
s = zmq.Context.instance().socket(zmq.REQ)
s.connect(ep)


def call(o):
    s.send(protocol.pack(o))
    return protocol.unpack(s.recv())


print("ping:", call({"cmd": "ping"}))
print("set_vocab:", call({"cmd": "set_vocab",
                          "thing_classes": ["chair", "table", "person"],
                          "stuff_classes": ["wall", "floor", "ceiling"]}))
img = (np.random.rand(480, 640, 3) * 255).astype("uint8")
r = call({"cmd": "frame", "image": img})
print("frame0:", r["ok"], r["label_map"].shape, r["label_map"].dtype,
      r["id_map"].shape, "instances", len(r["instances"]))
print("frame1:", call({"cmd": "frame", "image": img})["ok"])
print("reset:", call({"cmd": "reset"}))
print("bad cmd:", call({"cmd": "nope"}))
print("SMOKE_OK")
