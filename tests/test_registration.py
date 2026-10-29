"""The pack registers exactly its 13 nodes, node id = display name, one category, and every
frontend file and kit entry names a registered id.

    E:/ComfyUI-Easy-Install/python_embeded/python.exe -m pytest tests
Needs ComfyUI on sys.path (folder_paths, comfy_api, server); the check inserts it.
"""

import filecmp
import importlib.util
import os
import re
import sys
import unittest

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.environ.get("COMFYUI_DIR", r"E:\ComfyUI-Easy-Install\ComfyUI")
MEDIA_TOOLS_WEB = os.environ.get("LUNA_MEDIA_TOOLS_WEB", r"D:\Claude\Luna-Media-Tools\web")

IDS = ["Luna-H3-Canvas", "Luna-H3-ConcatAVLatent", "Luna-H3-ReferenceLoader", "Luna-H3-AssetLoader",
       "Luna-H3-RefsOut", "Luna-H3-ShotList", "Luna-H3-LoopStart", "Luna-H3-LoopScene",
       "Luna-H3-LoopProfile", "Luna-H3-GuideApply", "Luna-H3-GuideClip", "Luna-H3-LoopEnd",
       "Luna-H3-Sampler"]


def load_pack():
    if COMFY not in sys.path:
        sys.path.insert(0, COMFY)
    from aiohttp import web
    import server
    if not hasattr(server.PromptServer, "instance"):   # no running ComfyUI in a test
        server.PromptServer.instance = type("PS", (), {"routes": web.RouteTableDef(),
                                                       "send_sync": lambda *a, **k: None})()
    spec = importlib.util.spec_from_file_location("Luna_MinimaxH3_under_test",
                                                  os.path.join(PACK, "__init__.py"),
                                                  submodule_search_locations=[PACK])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def theme_code():
    with open(os.path.join(PACK, "web", "luna_pack_theme.js"), encoding="utf8") as f:
        return "\n".join(l for l in f.read().splitlines() if not l.lstrip().startswith("//"))


class RegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = load_pack()

    def test_thirteen_ids(self):
        self.assertEqual(sorted(self.pack.NODE_CLASS_MAPPINGS), sorted(IDS))

    def test_id_is_display_name(self):
        self.assertEqual(dict(self.pack.NODE_DISPLAY_NAME_MAPPINGS), {i: i for i in IDS})

    def test_one_category(self):
        for node_id, cls in self.pack.NODE_CLASS_MAPPINGS.items():
            self.assertEqual(getattr(cls, "CATEGORY", None), "Luna-MinimaxH3", node_id)

    def test_web_folder_exists(self):
        self.assertEqual(self.pack.WEB_DIRECTORY, "./web")
        self.assertTrue(os.path.isdir(os.path.join(PACK, "web")))

    def test_every_js_node_name_is_a_registered_id(self):
        wanted = {"luna_asset_loader.js": "Luna-H3-AssetLoader", "luna_shot_list.js": "Luna-H3-ShotList",
                  "luna_h3_canvas.js": "Luna-H3-Canvas", "luna_sampler.js": "Luna-H3-Sampler"}
        for name, node_id in wanted.items():
            with open(os.path.join(PACK, "web", name), encoding="utf8") as f:
                self.assertIn(f'"{node_id}"', f.read(), name)
        listed = set(re.findall(r'"(Luna-[A-Za-z0-9-]+)"', theme_code()))
        self.assertTrue(listed <= set(IDS), listed - set(IDS))

    def test_every_node_has_help_and_collapse(self):
        code = theme_code()
        help_part = code[code.index("registerLunaHelp(app"):code.index("registerLunaCollapse(app")]
        collapse_part = code[code.index("registerLunaCollapse(app"):]
        self.assertEqual(set(re.findall(r'"(Luna-[A-Za-z0-9-]+)"', help_part)), set(IDS))
        self.assertEqual(set(re.findall(r'"(Luna-[A-Za-z0-9-]+)"', collapse_part)), set(IDS))

    def test_every_node_has_a_description(self):
        for node_id, cls in self.pack.NODE_CLASS_MAPPINGS.items():
            desc = getattr(cls, "DESCRIPTION", None)
            if desc is None and hasattr(cls, "GET_SCHEMA"):
                desc = cls.GET_SCHEMA().description
            self.assertTrue(desc and len(str(desc).strip()) > 20, node_id)

    def test_kit_files_match_luna_media_tools(self):
        for name in ("luna_theme.mjs", "luna_help.mjs", "luna_collapse.mjs"):
            other = os.path.join(MEDIA_TOOLS_WEB, name)
            if not os.path.isfile(other):
                self.skipTest("Luna-Media-Tools not beside this pack")
            self.assertTrue(filecmp.cmp(os.path.join(PACK, "web", name), other, shallow=False), name)


if __name__ == "__main__":
    unittest.main()
