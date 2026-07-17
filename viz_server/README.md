# viz_server

Rerun visualizer for the lift3d semantic Universe. It is the **visualizer half** of the
visualizer/logic split: `run_semantic_universe` (C++) streams the pipeline's per-frame and
final state over ZMQ + msgpack, and this process renders it with [rerun](https://rerun.io).
All rerun/Arrow dependencies live here, so the C++ build carries none and can be built on a
locked-down system.

## Run

Start the visualizer **first** (it binds the PULL socket), then run the C++ side pointed at
the same endpoint:

```bash
# live viewer
uv run --project viz_server viz_server/main.py --spawn
./build-cuda/run_semantic_universe --config run.yaml --viz ipc:///tmp/lift3d_viz.ipc

# record to a .rrd instead (exits after the run's finish message)
uv run --project viz_server viz_server/main.py --out /tmp/sem.rrd
./build-cuda/run_semantic_universe --config run.yaml --viz ipc:///tmp/lift3d_viz.ipc
```

The entity tree, palettes, segmentation overlays, and legend match the old
`universe/viz_semantic_universe.cpp` exactly; only the transport moved to IPC. Per-frame
entities are logged on the timeline (overwrite each frame); the full world map arrives once
at `finish` and is logged as a static snapshot. See `protocol.py` for the message catalogue.
