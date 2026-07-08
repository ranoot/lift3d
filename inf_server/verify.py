"""Standalone verification: build + load + classifier + two-frame forward."""
import numpy as np
from dvis_runner import DvisRunner

r = DvisRunner(device="cuda")
print(">> set_vocab ...", flush=True)
n = r.set_vocab(
    thing_classes=["chair", "table", "person", "door", "window"],
    stuff_classes=["wall", "floor", "ceiling"],
)
print(f">> num_classes = {n}", flush=True)

img = (np.random.rand(480, 640, 3) * 255).astype("uint8")
print(">> frame 0 (resume=False) ...", flush=True)
out0 = r.infer(img)
print("   label_map", out0["label_map"].shape, out0["label_map"].dtype,
      "id_map", out0["id_map"].shape, "instances", len(out0["instances"]), flush=True)
if out0["instances"]:
    i0 = out0["instances"][0]
    print("   inst0:", {k: (v.shape if hasattr(v, "shape") else v) for k, v in i0.items()}, flush=True)

print(">> frame 1 (resume=True) ...", flush=True)
out1 = r.infer(img)
print("   label_map", out1["label_map"].shape, "instances", len(out1["instances"]), flush=True)

print(">> reset + frame ...", flush=True)
r.reset()
out2 = r.infer(img)
print("   ok, instances", len(out2["instances"]), flush=True)
print(">> VERIFY_OK", flush=True)
