"""DvisRunner: abstracts loading + running the OV-DVIS++ online model.

This is the model-side counterpart to ``mask3d_feat``'s inline backbone load. It
owns the detectron2 config assembly, the dynamic open-vocabulary text classifier,
the stateful frame-by-frame tracker, and the per-pixel resolution of the tracker
output into label/instance maps ready for 2D->3D lifting.

The vendored ``DVIS_Plus/`` tree is put on ``sys.path`` (mirroring how
``mask3d_feat/main.py`` does ``sys.path.insert(0, "Mask3D")``) so detectron2's
registries pick up the OV meta-architectures and backbones.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_DVIS = os.path.join(_HERE, "DVIS_Plus")
# _HERE first so the vendored ``panopticapi`` shim resolves regardless of cwd.
for _p in (_DVIS, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import MetadataCatalog  # noqa: E402
from detectron2.modeling import build_model  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from detectron2.structures import ImageList  # noqa: E402

# Importing these registers the OV meta-architectures, backbones and decoders in
# detectron2's global registries and exposes the matching config defaults. Their
# package __init__s eagerly register builtin datasets that read CWD-relative data
# files (DVIS assumes you run from its repo root), so import them with cwd set to
# DVIS_Plus/. The dataset registrations are unused at inference but must not error.
_prev_cwd = os.getcwd()
os.chdir(_DVIS)
try:
    from mask2former import add_maskformer2_config  # noqa: E402
    from mask2former_video import add_maskformer2_video_config  # noqa: E402
    from dvis_Plus import add_minvis_config, add_dvis_config  # noqa: E402
    import ov_dvis  # noqa: E402,F401  (registers DVIS_online_OV, CLIP, decoders)
    from ov_dvis import add_ov_dvis_config  # noqa: E402
finally:
    os.chdir(_prev_cwd)

DEFAULT_CONFIG = os.path.join(
    _DVIS, "configs", "open_vocabulary", "DVIS_Online_supervised_convnextl.yaml"
)
DEFAULT_WEIGHTS = os.path.join(_DVIS, "ov_online_supervised_convnextl.pth")


def _patch_clip_text_encode(backbone) -> None:
    """Make the CLIP backbone's text encoder layout-correct for modern open_clip.

    DVIS's ``CLIP.encode_text`` permutes tokens to LND (sequence-first), matching
    older open_clip. open_clip >= ~2.24 builds the text transformer with
    ``batch_first=True`` (N, L, D) and its own ``encode_text`` does not permute,
    so DVIS's manual permute makes the batch dim be read as the sequence length
    (e.g. mask '[1,1,112,112]' invalid for size 5929). This override drops the
    permute, replicating open_clip's own ``encode_text`` while reusing DVIS's
    backbone components. Consistent with the training notebook, which likewise
    patched CLIP text attention rather than relying on stock open_clip.
    """

    def encode_text(self, text, normalize: bool = False):
        clip = self.clip_model
        cast_dtype = clip.transformer.get_cast_dtype()
        x = clip.token_embedding(text).to(cast_dtype)  # (N, L, D)
        x = x + clip.positional_embedding.to(cast_dtype)
        x = clip.transformer(x, attn_mask=clip.attn_mask)  # batch_first, no permute
        x = clip.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    backbone.encode_text = types.MethodType(encode_text, backbone)


class DvisRunner:
    """Build-on-first-vocab, then run frames through the online tracker."""

    def __init__(
        self,
        weights_path: str = DEFAULT_WEIGHTS,
        config_file: str = DEFAULT_CONFIG,
        device: str = "cuda",
        min_size_test: int | None = None,
        score_thr: float = 0.5,
    ):
        self.weights_path = weights_path
        self.device = device
        self.score_thr = score_thr

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        add_maskformer2_video_config(cfg)
        add_minvis_config(cfg)
        add_dvis_config(cfg)
        add_ov_dvis_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.MODEL.DEVICE = device
        cfg.freeze()
        self.cfg = cfg
        # network-side shortest-edge resize target; matches INPUT.MIN_SIZE_TEST
        self.min_size_test = min_size_test or cfg.INPUT.MIN_SIZE_TEST

        self.model: Any = None
        self.name: str | None = None
        self.tc = None  # text classifier (computed per vocab)
        self.nt = None  # num templates
        self.num_classes = 0
        self._vocab_seq = 0
        self.keep = False  # tracker resume flag (False => next frame starts a video)

    # ------------------------------------------------------------------ vocab
    def set_vocab(self, thing_classes: list[str], stuff_classes: list[str]) -> int:
        """Register the vocabulary, (build &) load the model, compute classifier.

        Each call uses a fresh dataset name because detectron2's MetadataCatalog
        is set-once per key; the model is built only on the first call and reused
        via ``set_metadata`` afterwards.
        """
        thing = sorted(thing_classes)
        stuff = sorted(stuff_classes)
        classes_ov = thing + stuff
        if not classes_ov:
            raise ValueError("vocabulary is empty")

        name = f"inf_vocab_{self._vocab_seq}"
        self._vocab_seq += 1
        meta = MetadataCatalog.get(name)
        meta.set(
            classes_ov=classes_ov,
            thing_classes=thing,
            stuff_classes=stuff,
            thing_dataset_id_to_contiguous_id={i: i for i in range(len(thing))},
        )

        if self.model is None:
            self.cfg.defrost()
            self.cfg.DATASETS.TRAIN = (name,)
            self.cfg.DATASETS.TEST = (name,)
            self.cfg.DATASETS.TEST2TRAIN = [name]
            self.cfg.freeze()
            self.model = build_model(self.cfg)  # dispatches to DVIS_online_OV.from_config
            DetectionCheckpointer(self.model).resume_or_load(
                self.weights_path, resume=False
            )
            _patch_clip_text_encode(self.model.backbone)
            self.model.eval()
        else:
            # refresh: register the new vocab's class prepares + drop stale cache
            self.model.set_metadata(name, meta)

        tc, nt = self.model._set_class_information(name, train=False)
        tc, nt = self.model.get_text_classifier_with_void(tc, nt, name=name)
        self.tc, self.nt, self.name, self.num_classes = tc, nt, name, len(classes_ov)
        self.reset()
        return self.num_classes

    # ------------------------------------------------------------------ reset
    def reset(self) -> None:
        """Start a new tracking sequence (next frame is frame 0)."""
        if self.model is not None:
            self.model.tracker._clear_memory()
        self.keep = False

    # ----------------------------------------------------------------- infer
    @torch.no_grad()
    def infer(self, image_rgb: np.ndarray) -> dict:
        """Run one RGB frame (H, W, 3 uint8) -> resolved per-pixel maps."""
        if self.model is None:
            raise RuntimeError("set_vocab must be called before infer")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3) RGB, got {image_rgb.shape}")
        H0, W0 = int(image_rgb.shape[0]), int(image_rgb.shape[1])

        model = self.model
        x = torch.as_tensor(
            np.ascontiguousarray(image_rgb.transpose(2, 0, 1)), device=self.device
        ).float()  # (3, H0, W0) in [0, 255]

        # network-side shortest-edge resize (keep aspect)
        scale = self.min_size_test / min(H0, W0)
        nh, nw = int(round(H0 * scale)), int(round(W0 * scale))
        if (nh, nw) != (H0, W0):
            x = F.interpolate(
                x[None], size=(nh, nw), mode="bilinear", align_corners=False
            )[0]

        x = (x - model.pixel_mean) / model.pixel_std
        images = ImageList.from_tensors([x], model.size_divisibility)
        pad_h, pad_w = images.tensor.shape[-2:]

        features = model.backbone(images.tensor)
        features["text_classifier"] = self.tc
        features["num_templates"] = self.nt
        out = model.sem_seg_head(features)

        frame_embds = out["pred_embds"]                       # (b, c, t, q)
        frame_embds_no_norm = out["pred_embds_without_norm"]  # (b, c, t, q)
        mask_features = out["mask_features"].unsqueeze(0)
        object_labels = model._get_instance_labels(out["pred_logits"])

        track_out = model.tracker(
            frame_embds,
            mask_features,
            resume=self.keep,
            frame_classes=object_labels,
            frame_embeds_no_norm=frame_embds_no_norm,
            cur_feature=None,  # segmenter emits no 'transformer_features'
            text_classifier=self.tc,
            num_templates=self.nt,
        )
        self.keep = True

        # upsample query masks from the padded /4 grid back to the caller's (H0, W0):
        # to padded size -> crop the padding -> resize to original.
        m = track_out["pred_masks"][0, :, 0]  # (q, h, w) low-res logits
        m = F.interpolate(
            m[None], size=(pad_h, pad_w), mode="bilinear", align_corners=False
        )[0]
        m = m[:, :nh, :nw]
        m = F.interpolate(
            m[None], size=(H0, W0), mode="bilinear", align_corners=False
        )[0]  # (q, H0, W0) logits

        cls_logits = track_out["pred_logits"][0, 0]      # (q, c) incl. void column(s)
        embds = track_out["pred_embds"][0, :, 0].permute(1, 0)  # (q, c_emb)
        return self._resolve_per_pixel(m, cls_logits, embds, H0, W0)

    # --------------------------------------------------------- per-pixel resolve
    def _resolve_per_pixel(self, mask_logits, cls_logits, embds, H, W) -> dict:
        """Single-frame panoptic-style assignment of pixels to queries.

        Each pixel goes to the query maximizing mask_prob * class_score; pixels
        below ``score_thr`` are background (-1). Embeddings ride in a compact
        per-instance table (a dense H*W*C float map is too big for IPC).
        """
        masks = mask_logits.sigmoid()                    # (q, H, W)
        scores = cls_logits.softmax(-1)[:, :-1]          # drop trailing void -> (q, ncls)
        labels = scores.argmax(-1)                       # (q,)
        qscore = scores.max(-1).values                   # (q,)

        weighted = masks * qscore[:, None, None]         # (q, H, W)
        conf, winner = weighted.max(0)                   # (H, W), (H, W)
        bg = conf < self.score_thr

        id_map = winner.to(torch.int32)
        label_map = labels[winner].to(torch.int32)
        id_map[bg] = -1
        label_map[bg] = -1

        present = torch.unique(winner[~bg]) if (~bg).any() else winner.new_empty((0,))
        embds_cpu = embds.detach().float().cpu().numpy()
        labels_cpu = labels.detach().cpu().numpy()
        qscore_cpu = qscore.detach().cpu().numpy()
        instances = [
            {
                "id": int(q),
                "label": int(labels_cpu[q]),
                "score": float(qscore_cpu[q]),
                "embedding": embds_cpu[q].astype(np.float32),
            }
            for q in present.tolist()
        ]

        return {
            "h": H,
            "w": W,
            "label_map": label_map.cpu().numpy().astype(np.int16),
            "id_map": id_map.cpu().numpy().astype(np.int16),
            "instances": instances,
        }
