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
    from ov_dvis.meta_architecture_ov import get_classification_logits  # noqa: E402
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
        object_mask_thresh: float = 0.0,
        overlap_thresh: float = 0.8,
        ensemble_alpha: float = 0.4,
        ensemble_beta: float = 0.8,
        synonyms: dict[str, list[str]] | None = None,
    ):
        self.weights_path = weights_path
        self.device = device
        # VPS panoptic-resolution + FC-CLIP geometric-ensemble knobs (see infer/_resolve_vps)
        self.object_mask_thresh = object_mask_thresh
        self.overlap_thresh = overlap_thresh
        self.ensemble_alpha = ensemble_alpha
        self.ensemble_beta = ensemble_beta
        # canonical class name -> extra synonym phrases; expanded into the model-side
        # comma string in set_vocab (see _expand). Never leaves Python.
        self.synonyms = synonyms or {}

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
        self.num_thing = 0  # thing count => classes_ov[:num_thing] are thing classes
        self._vocab_seq = 0
        self.keep = False  # tracker resume flag (False => next frame starts a video)

    # ------------------------------------------------------------------ vocab
    def _expand(self, name: str) -> str:
        """Canonical class name -> comma-joined "canonical,syn1,syn2" prompt string.

        The model (prepare_class_names_from_metadata) splits on commas and max-ensembles
        over the synonyms, so more phrasings only raise recall. Returns ``name`` unchanged
        when the class has no synonyms. The canonical name stays first and is de-duplicated
        if a caller lists it among its own synonyms.
        """
        syns = self.synonyms.get(name)
        if not syns:
            return name
        out = [name]
        for s in syns:
            s = s.strip()
            if s and s not in out:
                out.append(s)
        return ",".join(out)

    def set_vocab(self, thing_classes: list[str], stuff_classes: list[str]) -> int:
        """Register the vocabulary, (build &) load the model, compute classifier.

        Each call uses a fresh dataset name because detectron2's MetadataCatalog
        is set-once per key; the model is built only on the first call and reused
        via ``set_metadata`` afterwards.
        """
        thing = sorted(thing_classes)  # sort by canonical name (must match the C++ vocab)
        stuff = sorted(stuff_classes)
        if not thing and not stuff:
            raise ValueError("vocabulary is empty")
        # Enrich each slot with synonyms AFTER sorting, so the label-id order stays keyed on
        # the canonical name (only the string content of each class grows). The model splits
        # these comma strings itself (prepare_class_names_from_metadata); C++ never sees them.
        classes_ov = [self._expand(c) for c in thing] + [self._expand(c) for c in stuff]

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
        # classes_ov == thing + stuff, so a contiguous class id < num_thing is a thing.
        # This is the ordering the VPS resolution and the C++ vocab both assume.
        self.num_thing = len(thing)
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

        cls_logits = self._ensemble_class_logits(track_out, features)  # (q, c) log-probs
        embds = track_out["pred_embds"][0, :, 0].permute(1, 0)  # (q, c_emb)
        return self._resolve_vps(m, cls_logits, embds, H0, W0)

    # ----------------------------------------------------- FC-CLIP geometric ensemble
    def _ensemble_class_logits(self, track_out, features) -> torch.Tensor:
        """Blend the tracker's in-vocab class scores with the out-of-vocab CLIP branch.

        This is FC-CLIP's open-vocabulary mechanism (mirrors
        ``meta_architecture_ov.py``'s ensemble at ~1283-1324, specialised to the
        single-frame ``t == 1`` case). Without it the runner scores every query with
        only the mask-head classifier, which is unreliable for classes absent from the
        training vocab (signs, extinguishers, ...). Returns per-query log-probabilities
        over ``[real classes .. void]``.
        """
        model = self.model
        clip_feat = features["clip_vis_dense"]
        # pool the dense CLIP features under each query's tracked mask (convnext path)
        mask_for_pooling = F.interpolate(
            track_out["pred_masks"][0].transpose(0, 1),  # (q,t,h,w) -> (t,q,h,w), t=1
            size=clip_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        pooled = model.mask_pooling(clip_feat, mask_for_pooling)      # (t, q, c)
        pooled = model.backbone.visual_prediction_forward(pooled)     # (t, q, c_embed)
        out_vocab = get_classification_logits(
            pooled, self.tc, model.backbone.clip_model.logit_scale, self.nt
        )  # (t, q, C+1)

        in_vocab = track_out["pred_logits"][0]                        # (t, q, C+1)
        in_probs = in_vocab[..., :-1].softmax(-1)                     # drop void -> (t,q,C)
        out_probs = out_vocab[..., :-1].softmax(-1)
        overlap = model.category_overlapping_mask.to(in_probs)        # (C,) 1=seen, 0=unseen
        a, b = self.ensemble_alpha, self.ensemble_beta
        seen = (in_probs ** (1 - a) * out_probs ** a).log() * overlap
        unseen = (in_probs ** (1 - b) * out_probs ** b).log() * (1 - overlap)
        cls_results = seen + unseen                                   # (t, q, C)
        # re-attach void so void-ish queries stay suppressible by object_mask_thresh
        is_void = in_vocab.softmax(-1)[..., -1:]                      # (t, q, 1)
        cls_probs = torch.cat(
            [cls_results.softmax(-1) * (1.0 - is_void), is_void], dim=-1
        )  # (t, q, C+1)
        return (cls_probs + 1e-8).log().mean(0)                       # (q, C+1), t-mean

    # ----------------------------------------------------- VPS panoptic per-pixel resolve
    def _resolve_vps(self, mask_logits, cls_logits, embds, H, W) -> dict:
        """Video-panoptic per-pixel assignment (mirrors ``inference_video_vps``, t == 1).

        Each kept query claims the pixels where it wins ``score * mask_prob``; a
        per-segment stability filter (``overlap_thresh``) drops flaky masks. Thing vs
        stuff is decided by ``class_id < num_thing`` (the ``thing + stuff`` ordering the
        vocab guarantees). ``label_map`` carries the class index for every assigned pixel
        (thing and stuff); ``id_map`` carries a per-instance id for thing pixels only
        (stuff and background stay -1), which is all the downstream instance-guided
        seeding consumes.
        """
        pred_cls = cls_logits.softmax(-1)                # (q, C+1)
        scores, labels = pred_cls.max(-1)                # (q,)
        void = pred_cls.shape[-1] - 1
        keep = labels.ne(void) & (scores > self.object_mask_thresh)

        label_map = torch.full((H, W), -1, dtype=torch.int32, device=mask_logits.device)
        id_map = torch.full((H, W), -1, dtype=torch.int32, device=mask_logits.device)
        instances: list[dict] = []

        if keep.any():
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_logits[keep].sigmoid()          # (nk, H, W) prob
            cur_embds = embds[keep]                          # (nk, c_emb)
            cur_prob = cur_scores.view(-1, 1, 1) * cur_masks
            cur_mask_ids = cur_prob.argmax(0)                # (H, W) winning query per pixel

            seg_id = 0
            for k in range(cur_classes.shape[0]):
                pred_class = int(cur_classes[k].item())
                isthing = pred_class < self.num_thing
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)
                mask_area = int(mask.sum().item())
                original_area = int((cur_masks[k] >= 0.5).sum().item())
                if mask_area == 0 or original_area == 0:
                    continue
                if mask_area / original_area < self.overlap_thresh:  # unstable segment
                    continue
                label_map[mask] = pred_class
                if isthing:
                    seg_id += 1
                    id_map[mask] = seg_id
                    instances.append(
                        {
                            "id": seg_id,
                            "label": pred_class,
                            "score": float(cur_scores[k].item()),
                            "embedding": cur_embds[k].detach().float().cpu().numpy().astype(np.float32),
                        }
                    )

        return {
            "h": H,
            "w": W,
            "label_map": label_map.cpu().numpy().astype(np.int16),
            "id_map": id_map.cpu().numpy().astype(np.int16),
            "instances": instances,
        }

    # --------------------------------------------------------- per-pixel resolve
