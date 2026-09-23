"""Frozen OpenDDE (ab_ag) Pairformer trunk + per-residue MLP epitope head.

See model.py in this directory for the implementation.
"""

from models.opendde.model import (  # noqa: F401
    OpenDDEPairformerEpitopeModel,
    build_model_opendde,
    load_base_opendde,
    resolve_opendde_checkpoint,
    resolve_opendde_root,
    trunk_cache_path,
    trunk_chunk_size,
)
