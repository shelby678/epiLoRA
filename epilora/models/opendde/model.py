"""Frozen OpenDDE Pairformer trunk (opendde_abag.pt) + per-residue MLP epitope
head -- the epiLoRA recipe applied to OpenDDE's antibody-antigen model.

The OpenDDE model (github.com/aurekaresearch/OpenDDE, pip name ``opendde``) is
an AlphaFold3-style co-folding network: an InputFeatureEmbedder + MSA module +
48-block PairformerStack "trunk" (run 10 recycling cycles) feeding a diffusion
module and a ConfidenceHead. This file uses the trunk, loaded from the
antibody-antigen-tuned checkpoint (``opendde_abag.pt``), as a frozen
per-residue feature extractor: one protein token per residue, so the trunk's
single representation ``s`` -- (N_token, 384) -- carries one row per residue.
The trainable part is only the per-residue MLP head from model.py
(head_dim=128 by default, an actual MLP unlike the champion's direct Linear).

Input is the antigen SEQUENCE (the trunk runs MSA-free and template-free via
opendde's own dummy-feature path, opendde.data.utils.make_dummy_feature --
the same path the shipped CLI takes with --use_msa false, so no mmseqs2 or
template databases are needed) PLUS the antigen's backbone CA geometry, which
enters through the confidence lane below.

The pair representation ``z`` -- (N_token, N_token, 384) -- is consumed the
way OpenDDE itself consumes it: through the checkpoint's own ConfidenceHead
(AF3 Algorithm 31, opendde/model/modules/confidence.py). OpenDDE never
averages pair rows -- its recipe is to run MORE PairformerStack blocks over z
and read per-residue outputs off the single stream. Concretely, the lane
mirrors ConfidenceHead.forward + memory_efficient_forward exactly:

    z'  = z + outer_sum(linear_s1(s_inputs), linear_s2(s_inputs))  # re-init
    z' += distance_embedding(CA-CA geometry of the antigen)       # trained bins
    s'  = confidence_head.pairformer_stack(LN(clamp(s)), z')      # 4 blocks

where the distance embedding is the confidence head's own trained binned
one-hot + raw-distance projections (LinearNoBias_d / _d_wo_onehot), computed
from the antigen's ACTUAL CA coordinates -- true geometry where OpenDDE feeds
its diffusion-predicted coordinates. The cached per-residue features are then
[s ; s'] -- the trunk single stream plus the confidence-block single stream
that z and the 3D geometry have flowed into. ``use_pair=False`` (the ablation)
drops the lane and trains on s alone, sequence-only.

Determinism: opendde's Featurizer applies a random rigid transform to the
reference-conformer positions by default (ref_pos_augment=True); the subclass
below turns that off, which makes the trunk output bit-identical across
processes (verified). Recycling and the confidence lane run under no_grad, so
no gradient ever flows through any frozen module -- gradients reach only the
head.

Embedding cache: the trunk + confidence lane is expensive (~1 min per
~350-residue antigen on an A6000, vs milliseconds for the head), and it is
frozen, so [s ; s'] is cached per (checkpoint, cycles, use_pair, sequence, CA
geometry) as an .npy under ``emb_cache`` -- O(L), a few MB per antigen -- and
reused across steps/folds/ablations. The CA-geometry term in the key matters:
the distance embedding makes the features a function of the structure too, so
two records sharing a sequence but not a structure get separate entries.
train.py points the cache at the coords cache next to --structures. With no
``emb_cache`` (e.g. predict.py) every forward recomputes the trunk.

Must run under the opendde environment (python >= 3.11, torch 2.7), NOT the
fair-esm env -- see machine_config.yaml's env_opendde / opendde_root. train.py
and predict.py work unchanged when launched with that env's python, e.g.:

    <env_opendde>/bin/python train.py --config configs/backbone_opendde.yaml

The opendde_abag.pt checkpoint and the CCD asset files it featurizes with live
under an "opendde root" (checkpoint/, common/) resolved from (1) the
OPENDDE_ROOT_DIR environment variable, (2) machine_config.yaml's opendde_root,
(3) opendde's own default ~/.cache/opendde.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# epilora/ must be importable for `from model import EpitopeModel` (repo
# convention: flat imports with epilora/ on sys.path, like data/scripts/*).
_EPILORA_DIR = Path(__file__).resolve().parents[2]
if str(_EPILORA_DIR) not in sys.path:
    sys.path.insert(0, str(_EPILORA_DIR))

from model import EpitopeModel  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_NAME = "opendde_abag.pt"

# Submodules OpenDDE's full forward uses that neither get_pairformer_output
# nor the confidence lane ever touches; deleting them after the checkpoint
# load cuts resident memory (~656M params -> ~420M with the confidence head
# kept).
_NON_TRUNK_MODULES = (
    "diffusion_module",
    "distogram_head",
    "structural_token_expander",
    "structural_token_refiner",
    "inference_noise_scheduler",
)


def resolve_opendde_root() -> Path:
    """The OpenDDE assets root (checkpoint/, common/): $OPENDDE_ROOT_DIR,
    else machine_config.yaml's ``opendde_root``, else opendde's default
    (~/.cache/opendde)."""
    env = os.environ.get("OPENDDE_ROOT_DIR")
    if env:
        return Path(env)
    machine_cfg = _EPILORA_DIR / "machine_config.yaml"
    if machine_cfg.exists():
        import yaml

        cfg = yaml.safe_load(machine_cfg.read_text()) or {}
        root = cfg.get("opendde_root")
        if root:
            return (Path(root) if Path(root).is_absolute()
                    else _EPILORA_DIR / root)
    from opendde.config.data import default_root_dir  # lazy: fair-esm-safe

    return Path(default_root_dir())


def resolve_opendde_checkpoint(checkpoint=None) -> Path:
    """The trunk checkpoint: ``checkpoint`` when given, else
    <opendde root>/checkpoint/opendde_abag.pt."""
    if checkpoint is not None:
        return Path(checkpoint)
    return resolve_opendde_root() / "checkpoint" / DEFAULT_CHECKPOINT_NAME


def load_base_opendde(checkpoint=None, device: str = "cpu"):
    """Load the frozen OpenDDE Pairformer trunk from the ab_ag checkpoint.

    Mirrors runner/inference.py's own load (build_inference_config +
    apply_runtime_compatibility + OpenDDE + strict state-dict load, stripping
    the DDP ``module.`` prefix some checkpoints carry), then prunes every
    module the trunk-only path never touches (see _NON_TRUNK_MODULES).
    """
    ckpt_path = resolve_opendde_checkpoint(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"OpenDDE checkpoint not found: {ckpt_path} -- download "
            f"opendde_abag.pt (see OpenDDE's README) and place it there, or "
            f"point OPENDDE_ROOT_DIR / machine_config.yaml's opendde_root at "
            f"its root")

    # Must happen before any `import opendde...`: opendde evaluates its asset
    # paths at module-import time.
    os.environ.setdefault("OPENDDE_ROOT_DIR", str(resolve_opendde_root()))

    from opendde.config.inference import (
        apply_runtime_compatibility,
        build_inference_config,
    )
    from opendde.model.opendde import OpenDDE
    from opendde.model.triangular.layers import skip_random_init

    dev = torch.device(device)
    configs = apply_runtime_compatibility(build_inference_config(), dev)
    with skip_random_init():  # every weight is overwritten by the checkpoint
        model = OpenDDE(configs)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    if any(k.startswith("module.") for k in state):  # DDP-saved checkpoint
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    for attr in _NON_TRUNK_MODULES:
        if hasattr(model, attr):
            delattr(model, attr)
    for p in model.parameters():
        p.requires_grad = False
    logger.info(f"Loaded OpenDDE trunk from {ckpt_path} "
                f"({sum(p.numel() for p in model.parameters()):,} params kept)")
    return model.to(device).eval()


def _features_for_sequence(seq: str) -> dict:
    """OpenDDE input_feature_dict for one antigen sequence (one protein chain,
    MSA-free, template-free, augmentation-free).

    Same pipeline InferenceDataset.process_one runs for a protein-only input
    JSON with use_msa/use_template off (SampleDictToFeatures -> dummy MSA ->
    data_type_transform), except Featurizer's random ref_pos rigid
    augmentation is disabled so the trunk features are deterministic.
    """
    from opendde.data.core.featurizer import Featurizer
    from opendde.data.inference.json_to_feature import SampleDictToFeatures
    from opendde.data.tokenizer import AtomArrayTokenizer
    from opendde.data.utils import data_type_transform, make_dummy_feature

    class _NoAugSampleDictToFeatures(SampleDictToFeatures):
        """SampleDictToFeatures with ref_pos's random rigid transform off."""

        def get_feature_dict(self):
            # Mirrors SampleDictToFeatures.get_feature_dict (minus the
            # geometry-featurizer branch, unused here) with ref_pos_augment
            # flipped to False in the Featurizer constructor.
            atom_array = self.get_atom_array()
            aa_tokenizer = AtomArrayTokenizer(atom_array)
            token_array = aa_tokenizer.get_token_array()
            featurizer = Featurizer(token_array, atom_array,
                                    include_discont_poly_poly_bonds=True,
                                    ref_pos_augment=False)
            feature_dict = featurizer.get_all_input_features()
            token_array_with_frame = featurizer.get_token_frame(
                token_array=token_array, atom_array=atom_array,
                ref_pos=feature_dict["ref_pos"], ref_mask=feature_dict["ref_mask"],
            )
            feature_dict["has_frame"] = torch.Tensor(
                token_array_with_frame.get_annotation("has_frame")).long()
            feature_dict["frame_atom_index"] = torch.Tensor(
                token_array_with_frame.get_annotation("frame_atom_index")).long()
            return feature_dict, atom_array, token_array

    sample = {"name": "epilora",
              "sequences": [{"proteinChain": {"sequence": seq, "count": 1}}],
              "modelSeeds": [1], "dialect": "openfold"}
    features, atom_array, _ = _NoAugSampleDictToFeatures(sample).get_feature_dict()
    features["distogram_rep_atom_mask"] = torch.Tensor(
        atom_array.distogram_rep_atom_mask).long()
    features = make_dummy_feature(features_dict=features,
                                  dummy_feats=["msa", "template"])
    return data_type_transform(features)


def _ca_coords(coords) -> torch.Tensor:
    """(L, 3) CA coordinates from the (L, 3, 3) N/CA/C arrays data.py loads.
    Raises on missing CA atoms (NaN) rather than silently feeding the
    distance lane wrong geometry."""
    ca = np.asarray(coords, dtype=np.float32)[:, 1, :]
    if np.isnan(ca).any():
        n = int(np.isnan(ca).any(axis=-1).sum())
        raise ValueError(f"{n} residues are missing CA coordinates -- the "
                         f"confidence lane's distance embedding needs them")
    return torch.from_numpy(ca)


def _confidence_lane(trunk, s_inputs, s, z, ca: torch.Tensor) -> torch.Tensor:
    """The checkpoint's own ConfidenceHead recipe, frozen: the per-residue
    single stream that the pair representation z and the CA-CA geometry have
    flowed into (see the module docstring for the exact correspondence with
    ConfidenceHead.forward / memory_efficient_forward)."""
    from opendde.model.utils import one_hot

    conf = trunk.confidence_head
    # z' = z + outer_sum(linear_s1(s_inputs), linear_s2(s_inputs)) -- the
    # confidence head's pair re-initialization (ConfidenceHead.forward's
    # _prepare_confidence_inputs).
    z = z + conf.linear_no_bias_s1(s_inputs)[None, :, :] \
          + conf.linear_no_bias_s2(s_inputs)[:, None, :]
    # z' += the trained distance embedding, over the antigen's true CA-CA
    # distances where OpenDDE feeds its diffusion-predicted ones
    # (memory_efficient_forward's pair-distance embedding).
    dist = torch.cdist(ca, ca)
    z = z + conf.linear_no_bias_d(
        one_hot(x=dist, lower_bins=conf.lower_bins, upper_bins=conf.upper_bins)
        .to(dtype=conf.linear_no_bias_d.weight.dtype))
    z = z + conf.linear_no_bias_d_wo_onehot(
        dist.unsqueeze(dim=-1).to(dtype=conf.linear_no_bias_d_wo_onehot.weight.dtype))
    # 4 trained PairformerStack blocks on (LN(clamp(s)), z'); pair_mask=None
    # matches the model's own single-sequence call site (opendde.py:2040).
    # chunk_size mirrors the model's own inference path (_forward_impl passes
    # configs.infer_setting.chunk_size) -- unchunked, the triangle ops on a
    # ~1300-residue antigen alone need >40 GiB.
    s_ln = conf.input_strunk_ln(torch.clamp(s, min=-512, max=512))
    s_conf, _ = conf.pairformer_stack(
        s_ln, z, pair_mask=None,
        triangle_multiplicative=trunk.configs.triangle_multiplicative,
        triangle_attention=trunk.configs.triangle_attention,
        inplace_safe=True,  # no_grad, and z/s_ln are our own fresh tensors
        chunk_size=trunk.configs.infer_setting.chunk_size)
    return s_conf


@torch.no_grad()
def _trunk_features(trunk, seq: str, coords, cycles: int,
                    use_pair: bool) -> np.ndarray:
    """(len(seq), width) per-residue trunk features -- one row per residue
    (one protein token per residue), float32.

    ``width`` is the trunk's c_s (the single representation ``s``), doubled
    when ``use_pair``: the confidence lane's single stream is appended, so
    the features are [s ; s']. ``coords`` (the (L, 3, 3) N/CA/C array
    data.load_backbone_coords produces) is required exactly when
    ``use_pair`` -- its CA geometry feeds the distance embedding.
    """
    from opendde.model.opendde import update_input_feature_dict

    features = _features_for_sequence(seq)
    dev = next(trunk.parameters()).device
    features = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v)
                for k, v in features.items()}
    features = trunk.relative_position_encoding.generate_relp(features)
    features = update_input_feature_dict(features)
    # chunk_size + inplace_safe as the model's own inference passes them
    # (_forward_impl -> main_inference_loop): we always run no_grad, and
    # unchunked triangle ops are O(L^2) workspace that OOMs the longest
    # antigens even on a 48 GiB card.
    s_inputs, s, z = trunk.get_pairformer_output(
        features, N_cycle=cycles,
        chunk_size=trunk.configs.infer_setting.chunk_size,
        inplace_safe=True)
    if s.shape[0] != len(seq):  # tokenization must stay 1:1 with residues
        raise RuntimeError(f"OpenDDE trunk returned {s.shape[0]} tokens for a "
                           f"{len(seq)}-residue sequence")
    lanes = [s]
    if use_pair:
        if coords is None:
            raise ValueError("use_pair=True needs backbone coords -- the "
                             "confidence lane's distance embedding reads them")
        ca = _ca_coords(coords).to(dev)
        lanes.append(_confidence_lane(trunk, s_inputs, s, z, ca))
    return torch.cat(lanes, dim=-1).detach().float().cpu().numpy()


class OpenDDEPairformerEpitopeModel(EpitopeModel):
    """Frozen OpenDDE (ab_ag) Pairformer trunk + per-residue MLP epitope head.

    The head reads the trunk's single representation ``s`` plus (when
    ``use_pair``, the default) the confidence lane: the checkpoint's own
    ConfidenceHead blocks run frozen over the pair representation and the
    antigen's CA-CA geometry, and their output single stream ``s'`` is
    concatenated onto ``s`` -- so the MLP sees the residue-residue and
    structural information z carries, processed the way OpenDDE itself
    processes it (see the module docstring), not a hand-rolled summary of it.

    ``emb_cache`` (optional) is a directory the frozen trunk's per-sequence
    features are cached in (see the module docstring); it is runtime-only
    state, deliberately not part of config()/checkpoints. The head itself is
    the same MLP head every other epiLoRA backbone uses
    (model.EpitopeModel._init_head), so extra_feats work here too.
    """

    def __init__(self, trunk, cycles: int = 10, dropout: float = 0.1,
                 head_dim: int | None = 128, extra_feats=(), use_pair: bool = True,
                 checkpoint=None, emb_cache=None):
        super().__init__()
        self.opendde = trunk
        self.cycles = int(cycles)
        self.use_pair = bool(use_pair)
        self._cfg = dict(cycles=self.cycles, dropout=dropout, head_dim=head_dim,
                         use_pair=self.use_pair, extra_feats=list(extra_feats),
                         checkpoint=None if checkpoint is None else str(checkpoint))
        self._emb_cache = Path(emb_cache) if emb_cache is not None else None
        for p in self.opendde.parameters():
            p.requires_grad = False
        # Trunk width read off the loaded model (configs.c_s), not hardcoded;
        # the confidence lane doubles it when enabled.
        self._init_head(int(trunk.c_s) * (2 if self.use_pair else 1),
                        dropout, head_dim, extra_feats)

    def _emb_cache_path(self, seq: str, coords) -> Path | None:
        """Cache file for this sample's trunk features -- keyed by (checkpoint,
        cycles, use_pair, chunk size, sequence, CA geometry). The geometry
        term matters: with use_pair the distance embedding makes the features
        a function of the structure, so two records sharing a sequence but
        not a structure must not share a cache entry (and changing the
        checkpoint, recycling depth, or trunk chunking can never silently
        reuse stale or wrong-shaped embeddings either)."""
        if self._emb_cache is None:
            return None
        ckpt = self._cfg.get("checkpoint") or DEFAULT_CHECKPOINT_NAME
        if self.use_pair and coords is not None:
            geo = hashlib.sha1(
                np.asarray(coords, dtype=np.float32)[:, 1, :].tobytes()
            ).hexdigest()
        else:
            geo = "nogeo"
        chunk = self.opendde.configs.infer_setting.chunk_size
        key = hashlib.sha1(
            f"{Path(ckpt).stem}|{self.cycles}|{int(self.use_pair)}|{chunk}|{geo}|{seq}"
            .encode()).hexdigest()
        return self._emb_cache / f"opendde_trunk_{key}.npy"

    def _trunk_features_cached(self, seq: str, coords) -> torch.Tensor:
        """(len(seq), head input width) trunk features, cached if a cache dir
        was given (the trunk + confidence lane are frozen, so their output
        never changes)."""
        cache_path = self._emb_cache_path(seq, coords)
        arr = None
        if cache_path is not None and cache_path.exists():
            try:
                arr = np.load(cache_path)
            except Exception as e:
                logger.warning(f"could not read cache {cache_path}: {e}; "
                               f"recomputing")
        if arr is None:
            arr = _trunk_features(self.opendde, seq, coords, self.cycles,
                                  self.use_pair)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_path.with_name(f"{cache_path.name}.tmp{os.getpid()}")
                with open(tmp, "wb") as f:  # atomic: concurrent runs share this
                    np.save(f, arr)
                os.replace(tmp, cache_path)
            # Long antigens leave multi-GiB fragments behind (the trunk's pair
            # tensors are O(L^2)); return them before the next sample.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    def _encode(self, coords_batch, seq_batch):
        """``coords_batch`` is the (L, 3, 3) N/CA/C array data.load_backbone_coords
        produces -- required when use_pair (its CA geometry feeds the distance
        embedding), ignored otherwise.

        Only batch size 1 is supported (mirrors ESM3/ESMc; train.py only ever
        calls forward() with single-element lists)."""
        if len(seq_batch) != 1:
            raise ValueError("OpenDDEPairformerEpitopeModel only supports "
                             "batch size 1")
        feats = self._trunk_features_cached(seq_batch[0], coords_batch[0])
        pad = feats.new_zeros(1, 1, feats.shape[-1])        # stand-in "begin" token
        return torch.cat([pad, feats.unsqueeze(0)], dim=1)  # (1, 1+L, width)


def build_model_opendde(device: str = "cpu", checkpoint=None, cycles: int = 10,
                        dropout: float = 0.1, head_dim: int | None = 128,
                        extra_feats=(), use_pair: bool = True,
                        emb_cache=None) -> "OpenDDEPairformerEpitopeModel":
    """Build an (untrained) OpenDDE-trunk epiLoRA model on ``device``.

    ``head_dim`` defaults to 128 (an MLP head), unlike the esmif1 champion's
    direct Linear -- that is this backbone's default recipe. ``use_pair``
    adds the confidence lane to the head's input (see the class docstring).
    """
    ckpt = resolve_opendde_checkpoint(checkpoint)
    trunk = load_base_opendde(ckpt, device=device)
    model = OpenDDEPairformerEpitopeModel(trunk, cycles=cycles, dropout=dropout,
                                          head_dim=head_dim, extra_feats=extra_feats,
                                          use_pair=use_pair, checkpoint=ckpt,
                                          emb_cache=emb_cache).to(device)
    return model
