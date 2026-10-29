"""Luna-MinimaxH3 — the nodes MiniMax H3 needs and does not ship. Thirteen nodes, category "Luna-MinimaxH3".

Node id = display name:
  Luna-H3-Canvas, Luna-H3-ConcatAVLatent, Luna-H3-ReferenceLoader, Luna-H3-AssetLoader,
  Luna-H3-RefsOut, Luna-H3-ShotList, Luna-H3-LoopStart, Luna-H3-LoopScene, Luna-H3-LoopProfile,
  Luna-H3-GuideApply, Luna-H3-GuideClip, Luna-H3-LoopEnd, Luna-H3-Sampler.

Nine of them (Canvas, ConcatAVLatent, the Loop and Guide nodes, Sampler) are V3 `io.ComfyNode` classes;
the other four (ReferenceLoader, AssetLoader, RefsOut, ShotList) are V1 classes. ComfyUI's loader takes
`NODE_CLASS_MAPPINGS` first and only falls through to `comfy_entrypoint` when it is absent
(`nodes.py`, "V1 node definition" / "V3 Extension Definition"), so a pack is one or the other. This
pack uses the V1 mapping for all thirteen and puts each V3 class in it under its schema's node id, the
way Luna-Media-Tools registers its V3 Luna-Ideogram. The frontend lives in one folder, web/.

Luna-H3-Sampler came from the separate Luna-H3-Sampler pack (2026-10-29). Its recipes route
(GET /luna_sampler/recipes) is registered here at import, because the V1 path has no `on_load`.
"""

from .h3_canvas import LunaMiniMaxH3Canvas
from .h3_latent import LunaH3ConcatAVLatent
from .h3_loop import LunaH3LoopEnd, LunaH3LoopStart
from .h3_loop_guide import LunaH3GuideApply, LunaH3GuideClip
from .h3_loop_scene import LunaH3LoopProfile, LunaH3LoopScene
from .luna_recipes import register_routes as _register_sampler_routes
from .luna_sampler import LunaH3Sampler
from .asset_loader import (
    NODE_CLASS_MAPPINGS as _LOADER_C,
    NODE_DISPLAY_NAME_MAPPINGS as _LOADER_D,
)
from .h3_reference_loader import (
    NODE_CLASS_MAPPINGS as _REFS_C,
    NODE_DISPLAY_NAME_MAPPINGS as _REFS_D,
)
from .h3_refs_out import (
    NODE_CLASS_MAPPINGS as _OUT_C,
    NODE_DISPLAY_NAME_MAPPINGS as _OUT_D,
)
from .shot_list import (
    NODE_CLASS_MAPPINGS as _SHOT_C,
    NODE_DISPLAY_NAME_MAPPINGS as _SHOT_D,
)

_V3_NODES = (LunaMiniMaxH3Canvas, LunaH3ConcatAVLatent, LunaH3LoopStart, LunaH3LoopEnd,
             LunaH3LoopScene, LunaH3LoopProfile, LunaH3GuideApply, LunaH3GuideClip, LunaH3Sampler)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _cls in _V3_NODES:
    _schema = _cls.GET_SCHEMA()
    NODE_CLASS_MAPPINGS[_schema.node_id] = _cls
    NODE_DISPLAY_NAME_MAPPINGS[_schema.node_id] = _schema.display_name

for _c, _d in ((_LOADER_C, _LOADER_D), (_REFS_C, _REFS_D), (_OUT_C, _OUT_D), (_SHOT_C, _SHOT_D)):
    NODE_CLASS_MAPPINGS.update(_c)
    NODE_DISPLAY_NAME_MAPPINGS.update(_d)

_register_sampler_routes()

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
