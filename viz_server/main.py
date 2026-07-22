"""viz_server: rerun visualizer for the lift3d semantic Universe.

The visualizer half of the visualizer/logic split. ``run_semantic_universe`` streams the
pipeline's per-frame + final state over ZMQ (PUSH), and this process (PULL) renders it with
rerun -- either into a live viewer (``--spawn``) or a ``.rrd`` file (``--out``). All rerun/
Arrow deps live here, so the C++ build carries none. The rendering mirrors the old
``universe/viz_semantic_universe.cpp`` one-for-one (same entity tree, palettes, overlays,
and legend), so the output is unchanged; only the transport moved to IPC.

Start this FIRST (it binds the PULL socket), then run the C++ side with a matching --viz:
    uv run --project viz_server viz_server/main.py --spawn
    ./build-cuda/run_semantic_universe --config run.yaml --viz ipc:///tmp/lift3d_viz.ipc

Per-frame entities (pose, camera, image, overlays, superpoints, proposals, objects) are
logged on the timeline and overwrite the same paths each frame (bounded memory in the live
viewer). The heavy full world map arrives once at ``finish`` and is logged as a static
snapshot, exactly as the old binary did.
"""

from __future__ import annotations

import argparse

import numpy as np
import rerun as rr
import zmq

import protocol

DEFAULT_ENDPOINT = "ipc:///tmp/lift3d_viz.ipc"
UNLABELED_ID = 0xFFFF  # AnnotationContext id for the grey "unlabeled" class

# ClassKind enum (universe.h): Unknown = 0, Thing = 1, Stuff = 2.
KIND_THING = 1
KIND_STUFF = 2

# Deterministic, high-contrast palette keyed by stable class id (mirrors classColor in the
# old C++ viz -- same 20 colours, same order, so the legend matches).
_CLASS_PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
    [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
    [210, 245, 60], [250, 190, 190], [0, 128, 128], [230, 190, 255],
    [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195],
    [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
], dtype=np.uint8)


def class_color(cid: int) -> np.ndarray:
    if cid < 0:
        return np.array([110, 110, 110], dtype=np.uint8)
    return _CLASS_PALETTE[cid % len(_CLASS_PALETTE)]


def superpoint_color(gid: int) -> np.ndarray:
    """Pseudo-random but stable colour keyed on an id (Knuth multiplicative hash), matching
    superpointColor in the old C++ viz so proposal/object/superpoint colours are identical."""
    h = (int(gid) * 2654435761) & 0xFFFFFFFF
    h ^= h >> 15
    return np.array([
        40 + (h & 0xFF) % 200,
        40 + ((h >> 8) & 0xFF) % 200,
        40 + ((h >> 16) & 0xFF) % 200,
    ], dtype=np.uint8)


def _set_time(idx: int, secs: float) -> None:
    """Set the frame/capture timelines, tolerant of rerun API drift across versions."""
    try:  # rerun >= 0.23 unified API
        rr.set_time("frame", sequence=idx)
        rr.set_time("capture", duration=secs)
    except (AttributeError, TypeError):  # older split API
        rr.set_time_sequence("frame", idx)
        rr.set_time_seconds("capture", secs)


def _transform3d(pose16: np.ndarray) -> rr.Transform3D:
    """Row-major 4x4 rigid pose -> rerun Transform3D (translation + 3x3 linear part)."""
    m = np.asarray(pose16, dtype=np.float32).reshape(4, 4)
    return rr.Transform3D(translation=m[:3, 3], mat3x3=m[:3, :3])


def _rgb(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a)


class Viz:
    def __init__(self) -> None:
        self.classes: dict[int, tuple[str, int]] = {}   # id -> (name, kind)  [Universe vocab]
        self.seg_vocab: list[str] = []                  # index == seg label_map/id_map class id
        self.seg_overlay = "off"
        self.seg_alpha = 0.5
        self.seg_min_area = 80
        self.traj: list[list[float]] = []               # accumulated robot positions
        self.last_objects: list[dict] = []              # retained for the static finish snapshot
        # Run-wide proposal-centroid scatter (world/seeds/all_centroids): every proposal ever
        # seen, distinctly coloured by a global running index (a dense clump that never became
        # one object == a merging failure; a bare region == a proposal gap).
        self.seed_ctr: list[np.ndarray] = []
        self.seed_col: list[np.ndarray] = []
        self.seed_lbl: list[str] = []
        self.seed_running = 0
        # Sign text per object id (from the sign-understanding egress): kept so a resolved sign
        # stays visible, and so the text log only fires when an object's text actually changes.
        self.obj_text: dict[int, str] = {}

    # ---- begin -------------------------------------------------------------
    def on_begin(self, msg: dict) -> None:
        self.classes = {int(c["id"]): (c["name"], int(c["kind"])) for c in msg["classes"]}
        self.seg_vocab = list(msg.get("seg_vocab", []))
        self.seg_overlay = msg.get("seg_overlay", "off")
        self.seg_alpha = float(msg.get("seg_alpha", 0.5))
        self.seg_min_area = int(msg.get("seg_min_area", 80))

        # AnnotationContext on "world": each class id -> name + palette colour, rendered as a
        # legend + used to colour the class-keyed world/map points and seg-by-class overlay.
        infos = [(cid, name, tuple(int(x) for x in class_color(cid)))
                 for cid, (name, _kind) in sorted(self.classes.items())]
        infos.append((UNLABELED_ID, "unlabeled", tuple(int(x) for x in class_color(-1))))
        rr.log("world", rr.AnnotationContext(infos), static=True)

    def class_name(self, cid: int) -> str:
        c = self.classes.get(int(cid))
        return c[0] if c else ""

    def seg_class_name(self, cid: int) -> str:
        """Resolve a 2D seg label_map class id through the InfClient vocab (its own id
        space -- distinct from the Universe legend used for 3D points)."""
        cid = int(cid)
        return self.seg_vocab[cid] if 0 <= cid < len(self.seg_vocab) else ""

    # ---- frame -------------------------------------------------------------
    def on_frame(self, msg: dict) -> None:
        idx = int(msg["idx"])
        _set_time(idx, float(msg["capture_secs"]))

        # Robot pose + trajectory + camera frustum (poses both interpolated to the image time).
        rr.log("world/robot", _transform3d(msg["robot_pose"]))
        self.traj.append([float(x) for x in np.asarray(msg["robot"])])
        rr.log("world/trajectory", rr.LineStrips3D([np.array(self.traj, dtype=np.float32)]))

        cam = msg["cam"]
        K = np.asarray(cam["K"], dtype=np.float32)
        w, h = int(cam["w"]), int(cam["h"])
        rr.log("world/camera", _transform3d(cam["T"]))
        try:
            rr.log("world/camera", rr.Pinhole(focal_length=[float(K[0]), float(K[4])],
                                              resolution=[w, h], image_plane_distance=0.4))
        except TypeError:  # older rerun without image_plane_distance kwarg
            rr.log("world/camera", rr.Pinhole(focal_length=[float(K[0]), float(K[4])],
                                              resolution=[w, h]))

        image = msg.get("image")
        if image is not None:
            rr.log("world/camera/image", rr.Image(_rgb(image)))

        # Painted DVIS segmentation overlay(s) for this frame (color-coded fill + labelled box
        # per region). Only when a raw image + masks are present and dims agree.
        seg = msg.get("seg")
        if (seg is not None and image is not None and self.seg_overlay != "off"
                and seg["label_map"].shape[:2] == image.shape[:2]):
            self._log_overlays(image, seg)

        # Superpoints (ephemeral; only sent on the frames VCCS recomputed).
        if "superpoints" in msg:
            self._log_superpoints(msg["superpoints"])

        # Proposals (re-seeded + re-grown every frame; replace the prior frame's).
        if "proposals" in msg:
            self._log_proposals(msg["proposals"])

        # Objects (tier 3, the primary output) -- logged on the timeline AND retained so the
        # finish snapshot can re-log the final set static.
        if "objects" in msg:
            self.last_objects = list(msg["objects"])
            self._log_objects(msg["objects"], static=False)

    # ---- segmentation overlays ---------------------------------------------
    def _log_overlays(self, image: np.ndarray, seg: dict) -> None:
        label_map = seg["label_map"]
        id_map = seg.get("id_map")
        if self.seg_overlay in ("class", "both"):
            self._log_one_overlay("world/camera/seg_by_class", image, label_map, id_map, "class")
        if self.seg_overlay in ("instance", "both"):
            self._log_one_overlay("world/camera/seg_by_instance", image, label_map, id_map,
                                  "instance")

    def _log_one_overlay(self, ent: str, image: np.ndarray, label_map: np.ndarray,
                         id_map, by: str) -> None:
        rr.log(ent, rr.Image(self._paint(image, label_map, id_map, by)))
        centers, halfs, colors, labels = self._seg_boxes(label_map, id_map, by)
        # Always log (empty => clears the prior frame's boxes at this time cursor).
        rr.log(ent + "/labels", rr.Boxes2D(centers=centers, half_sizes=halfs,
                                           colors=colors, labels=labels, show_labels=True))

    def _paint(self, image: np.ndarray, label_map: np.ndarray, id_map, by: str) -> np.ndarray:
        out = image.astype(np.float32)
        by_inst = by == "instance" and id_map is not None
        keymap = id_map if by_inst else label_map
        valid = keymap >= 0
        color_img = np.zeros_like(out)
        if by_inst:
            for k in np.unique(keymap[valid]):
                color_img[keymap == k] = superpoint_color(int(k))
        else:
            for c in np.unique(label_map[valid]):
                color_img[label_map == c] = class_color(int(c))
        a = float(np.clip(self.seg_alpha, 0.0, 1.0))
        out[valid] = (1.0 - a) * out[valid] + a * color_img[valid]
        return out.astype(np.uint8)

    def _seg_boxes(self, label_map: np.ndarray, id_map, by: str):
        by_inst = by == "instance" and id_map is not None
        keymap = id_map if by_inst else label_map
        centers, halfs, colors, labels = [], [], [], []
        for key in np.unique(keymap[keymap >= 0]):
            ys, xs = np.where(keymap == key)
            if xs.size < self.seg_min_area:
                continue
            umin, umax = int(xs.min()), int(xs.max())
            vmin, vmax = int(ys.min()), int(ys.max())
            wid, hei = umax - umin + 1, vmax - vmin + 1
            cls = int(label_map[ys[0], xs[0]])           # class of the first pixel (as in C++)
            centers.append([umin + wid / 2.0, vmin + hei / 2.0])
            halfs.append([wid / 2.0, hei / 2.0])
            colors.append(superpoint_color(int(key)) if by_inst else class_color(cls))
            name = self.seg_class_name(cls)
            labels.append(name if name else "?")
        return (np.array(centers, dtype=np.float32).reshape(-1, 2),
                np.array(halfs, dtype=np.float32).reshape(-1, 2),
                np.array(colors, dtype=np.uint8).reshape(-1, 3),
                labels)

    # ---- 3D layers ---------------------------------------------------------
    @staticmethod
    def _concat_points(entities, color_fn):
        """Flatten a list of {points,centroid,id,...} into (member xyz, member colours,
        centroid xyz, centroid colours, centroid labels) with one colour per entity."""
        mpos, mcol, ctr, ccol, clbl = [], [], [], [], []
        for e in entities:
            col = color_fn(e)
            pts = np.asarray(e["points"], dtype=np.float32).reshape(-1, 3)
            if pts.size:
                mpos.append(pts)
                mcol.append(np.tile(col, (pts.shape[0], 1)))
            ctr.append(np.asarray(e["centroid"], dtype=np.float32).reshape(3))
            ccol.append(col)
            name, text = e.get("name", ""), e.get("text", "")
            clbl.append(f"{name} | {text}" if text else name)
        mpos = np.concatenate(mpos) if mpos else np.zeros((0, 3), np.float32)
        mcol = np.concatenate(mcol) if mcol else np.zeros((0, 3), np.uint8)
        ctr = np.array(ctr, dtype=np.float32).reshape(-1, 3)
        ccol = np.array(ccol, dtype=np.uint8).reshape(-1, 3)
        return mpos, mcol, ctr, ccol, clbl

    def _log_objects(self, objects, static: bool) -> None:
        mpos, mcol, ctr, ccol, clbl = self._concat_points(
            objects, lambda o: superpoint_color(int(o["id"])))
        rr.log("world/objects/points",
               rr.Points3D(mpos, colors=mcol, radii=0.045, show_labels=False), static=static)
        rr.log("world/objects/centroids",
               rr.Points3D(ctr, colors=ccol, labels=clbl, radii=0.15), static=static)
        self._log_object_text(objects, static)

    def _log_object_text(self, objects, static: bool) -> None:
        """Objects carrying free text (today the sign-understanding result) get an
        always-labelled red marker at their centroid under world/objects/sign_text, plus a line
        in the `sign` text stream the first time that text appears. Sign text is sticky, so a
        frame without any text leaves the previous markers standing."""
        pos, labels = [], []
        for o in objects:
            text = o.get("text", "")
            if not text:
                continue
            oid, name = int(o["id"]), o.get("name", "")
            pos.append(np.asarray(o["centroid"], dtype=np.float32).reshape(3))
            labels.append(f"{name}: {text}")
            if self.obj_text.get(oid) != text:
                self.obj_text[oid] = text
                rr.log("sign", rr.TextLog(f"object {oid} ({name}) -> {text}", level="INFO"))
        if not pos:
            return
        rr.log("world/objects/sign_text",
               rr.Points3D(np.array(pos, dtype=np.float32).reshape(-1, 3),
                           colors=np.tile(np.array([255, 64, 64], np.uint8), (len(pos), 1)),
                           labels=labels, radii=0.25, show_labels=True), static=static)

    def _log_proposals(self, proposals) -> None:
        mpos, mcol, ctr, ccol, clbl = self._concat_points(
            proposals, lambda s: superpoint_color(int(s["id"])))
        rr.log("world/proposals/points",
               rr.Points3D(mpos, colors=mcol, radii=0.035, show_labels=False))
        rr.log("world/proposals/centroids",
               rr.Points3D(ctr, colors=ccol, labels=clbl, radii=0.06, show_labels=False))
        # Accumulate into the run-wide proposal-centroid scatter (distinct colour per proposal
        # via a global running index), logged static at finish.
        for s in proposals:
            self.seed_ctr.append(np.asarray(s["centroid"], dtype=np.float32).reshape(3))
            self.seed_col.append(superpoint_color(self.seed_running))
            self.seed_lbl.append(s.get("name", ""))
            self.seed_running += 1

    def _log_superpoints(self, sps) -> None:
        sp_pos, sp_col, th_pos, th_col = [], [], [], []
        ctr_pos, ctr_col, ctr_lbl = [], [], []
        for s in sps:
            col = superpoint_color(int(s["id"]))
            is_thing = int(s.get("kind", 0)) == KIND_THING and int(s["class_id"]) >= 0
            pts = np.asarray(s["points"], dtype=np.float32).reshape(-1, 3)
            if pts.size:
                sp_pos.append(pts)
                sp_col.append(np.tile(col, (pts.shape[0], 1)))
                if is_thing:
                    th_pos.append(pts)
                    th_col.append(np.tile(col, (pts.shape[0], 1)))
            if is_thing:
                ctr_pos.append(np.asarray(s["centroid"], dtype=np.float32).reshape(3))
                ctr_col.append(col)
                ctr_lbl.append(s.get("name", ""))

        def cat(a, cols):
            p = np.concatenate(a) if a else np.zeros((0, 3), np.float32)
            c = np.concatenate(cols) if cols else np.zeros((0, 3), np.uint8)
            return p, c

        p_all, c_all = cat(sp_pos, sp_col)
        p_th, c_th = cat(th_pos, th_col)
        rr.log("world/superpoints/all",
               rr.Points3D(p_all, colors=c_all, radii=0.02, show_labels=False))
        rr.log("world/superpoints/things",
               rr.Points3D(p_th, colors=c_th, radii=0.025, show_labels=False))
        rr.log("world/superpoints/centroids",
               rr.Points3D(np.array(ctr_pos, dtype=np.float32).reshape(-1, 3),
                           colors=np.array(ctr_col, dtype=np.uint8).reshape(-1, 3),
                           labels=ctr_lbl, radii=0.08, show_labels=False))

    # ---- finish ------------------------------------------------------------
    def on_finish(self, msg: dict) -> None:
        pos = np.asarray(msg["positions"], dtype=np.float32).reshape(-1, 3)
        cls = np.asarray(msg["class_ids"]).astype(np.int32)
        kinds = np.asarray(msg["kinds"]).astype(np.int32)

        thing = (kinds == KIND_THING) & (cls >= 0)
        stuff = (kinds == KIND_STUFF) & (cls >= 0)
        other = ~(thing | stuff)

        def log_map(ent: str, mask: np.ndarray, ids: np.ndarray) -> None:
            rr.log(ent, rr.Points3D(pos[mask], class_ids=ids.astype(np.uint16),
                                    radii=0.02, show_labels=False), static=True)

        log_map("world/map/things", thing, cls[thing])
        log_map("world/map/stuff", stuff, cls[stuff])
        log_map("world/map/unlabeled", other, np.full(int(other.sum()), UNLABELED_ID))

        # Final objects as an always-visible static snapshot (from the last frame's list).
        if self.last_objects:
            self._log_objects(self.last_objects, static=True)

        # All proposals from the whole run, as a small distinct-coloured static scatter.
        if self.seed_ctr:
            rr.log("world/seeds/all_centroids",
                   rr.Points3D(np.array(self.seed_ctr, dtype=np.float32).reshape(-1, 3),
                               colors=np.array(self.seed_col, dtype=np.uint8).reshape(-1, 3),
                               labels=self.seed_lbl, radii=0.035, show_labels=False),
                   static=True)

        n = int(pos.shape[0])
        print(f"[viz] finish: world={n} alive points "
              f"({int(thing.sum())} thing, {int(stuff.sum())} stuff, {int(other.sum())} other) "
              f"| {len(self.last_objects)} objects | {len(self.seed_ctr)} proposal centroids"
              f"{f' | {len(self.obj_text)} with sign text' if self.obj_text else ''}",
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                    help="ZMQ endpoint to bind the PULL socket (the runner connects PUSH here)")
    ap.add_argument("--spawn", action="store_true",
                    help="open a live rerun viewer (default; else save to --out)")
    ap.add_argument("--out", default="", help="save the recording to this .rrd instead of --spawn")
    ap.add_argument("--exit-on-finish", action="store_true",
                    help="exit after the first finish message (default in --out file mode)")
    args = ap.parse_args()

    rr.init("lift3d/semantic_universe")
    file_mode = bool(args.out) and not args.spawn
    if file_mode:
        rr.save(args.out)
    else:
        rr.spawn()
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PULL)
    sock.bind(args.endpoint)
    print(f"viz_server listening on {args.endpoint} "
          f"({'file:' + args.out if file_mode else 'live viewer'})", flush=True)

    viz = Viz()
    exit_on_finish = args.exit_on_finish or file_mode
    try:
        while True:
            msg = protocol.unpack(sock.recv())
            mtype = msg.get("type")
            try:
                if mtype == "begin":
                    viz.on_begin(msg)
                elif mtype == "frame":
                    viz.on_frame(msg)
                elif mtype == "finish":
                    viz.on_finish(msg)
                    if exit_on_finish:
                        break
                else:
                    print(f"[viz] unknown message type: {mtype!r}", flush=True)
            except Exception as exc:  # never let one bad message kill the viewer
                import traceback
                traceback.print_exc()
                print(f"[viz] error handling {mtype!r}: {exc}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
