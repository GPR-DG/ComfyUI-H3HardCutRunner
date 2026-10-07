"""Build the C Modular canvas from the immutable C baseline.

This keeps the C media/model/prompt inputs and output settings. It does not
execute RunningHub, and it must not be treated as RH deployment validation.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("H3_C_PROJECT_ROOT", REPO_ROOT.parents[2])).resolve()
BASE = PROJECT_ROOT / "work" / "modular_c_baseline_20261001" / "H3_V16_hardcut_runner_RH_SCENE_VLM_C.json"
TARGET = PROJECT_ROOT / "outputs" / "H3_V16_hardcut_runner_RH_SCENE_VLM_C.json"
A2_TARGET = PROJECT_ROOT / "outputs" / "H3_V16_hardcut_runner_RH_SCENE_VLM_C_A2_FORCE_REFS.json"
A2_FORCE_PRESENT = "1,9"


def _ensure_scene_vlm_outputs(node):
    """Append missing sockets only; never move or duplicate existing outputs."""
    names = ("shot_prompts", "empty_shot_indices", "report", "raw_vlm_output",
             "normalised_vlm_output", "scene_fields", "subject_mode",
             "secondary_human_presence", "source_prop_policy", "vlm_runtime", "shot_orientation")
    outputs = node["outputs"]
    if len(outputs) > len(names) or any(
        out.get("name") != names[i] or out.get("type") != "STRING" or
        out.get("slot_index", i) != i for i, out in enumerate(outputs)
    ):
        raise ValueError("SceneVLM output sockets must preserve the append-only schema")
    for i in range(len(outputs), len(names)):
        outputs.append({"name": names[i], "type": "STRING", "links": None, "slot_index": i, "shape": 3})


def make_node(node_id, node_type, title, inputs, outputs, pos, size=(310, 180), widgets=None):
    return {
        "id": node_id, "type": node_type, "pos": list(pos), "size": list(size),
        "flags": {}, "order": 0, "mode": 0,
        "inputs": [{"name": name, "type": kind, "link": None} for name, kind in inputs],
        "outputs": [{"name": name, "type": kind, "links": None, "slot_index": i, "shape": 3}
                    for i, (name, kind) in enumerate(outputs)],
        "title": title,
        "properties": {"Node name for S&R": node_type},
        "widgets_values": [] if widgets is None else widgets,
    }


def build():
    data = json.loads(BASE.read_text(encoding="utf-8"))
    source = {n["id"]: n for n in data["nodes"]}
    # The only retained old nodes are C's real input/model/analysis/output path
    # and the explicitly inspected RH/Comfy worker nodes from its old split chain.
    keep = {
        8, 9, 12, 17, 22, 33, 600, 601, 619, 620, 622, 623, 626, 627,
        631, 632, 633, 634, 635, 641, 643, 645, 700, 702, 703, 704, 705,
        706, 707, 712, 713,
        624, 625, 628, 629, 636, 637, 638, 639, 642, 646, 647,
    }
    nodes = {i: copy.deepcopy(source[i]) for i in keep}
    for n in nodes.values():
        for inp in n.get("inputs", []):
            inp["link"] = None
        for out in n.get("outputs", []):
            out["links"] = None
        n["mode"] = 0

    _ensure_scene_vlm_outputs(nodes[700])

    for i in (629, 646):
        ref3 = {"name": "ref_images.ref_image_2", "type": "IMAGE", "link": None,
                "shape": 7}
        insert_at = next(j for j, inp in enumerate(nodes[i]["inputs"])
                         if inp["name"] == "ref_images.ref_image_1") + 1
        nodes[i]["inputs"].insert(insert_at, ref3)
        nodes[i]["widgets_values"] = ["", 672, 1184, 39, "match"]

    specs = [
        (714, "H3CShotPlanner", "01  Shot Planner / Manifest",
         [("rgb", "IMAGE"), ("depth", "IMAGE"), ("seed", "INT"), ("cut_threshold", "FLOAT"), ("min_shot_frames", "INT")],
         [("manifest", "STRING"), ("shot_count", "INT"), ("total_frames", "INT"), ("report", "STRING")], (1460, 80), (350, 210), [999, 0.18, 8]),
        (715, "H3CShotSelectPad", "02  Shot Select / Depth Pad",
         [("rgb", "IMAGE"), ("depth", "IMAGE"), ("manifest", "STRING"), ("index", "INT")],
         [("shot_rgb", "IMAGE"), ("depth_ref_video", "IMAGE"), ("original_f", "INT"), ("work_l", "INT"), ("shot_seed", "INT"), ("shot_number", "INT"), ("range", "STRING")], (1840, 80), (360, 260), []),
        (716, "H3CShotPolicy", "03  Shot / Reference Policy",
         [("shot_prompts", "STRING"), ("empty_shot_indices", "STRING"), ("shot_number", "INT"), ("shot_count", "INT"), ("seed", "INT"),
          ("picture1", "IMAGE"), ("picture2", "IMAGE"), ("picture3", "IMAGE"),
          ("manual_force_empty_indices", "STRING"), ("manual_force_present_indices", "STRING")],
         [("picture1", "IMAGE"), ("picture2", "IMAGE"), ("picture3", "IMAGE"), ("strict_empty", "BOOLEAN"),
          ("allow_secondary_humans", "BOOLEAN"), ("inject_target_references", "BOOLEAN"),
          ("subject_mode", "STRING"), ("secondary_human_presence", "STRING"), ("source_prop_policy", "STRING"),
          ("reason", "STRING"), ("report", "STRING"), ("picture3_present", "BOOLEAN"),
          ("picture1_used", "BOOLEAN"), ("picture2_used", "BOOLEAN"), ("picture3_used", "BOOLEAN"), ("seed", "INT")],
         (2220, 80), (380, 590), ["", ""]),
        (717, "H3CPromptCompiler", "04  First / Second Prompt Compiler",
         [("shot_prompts", "STRING"), ("shot_number", "INT"), ("prompt_first", "STRING"),
          ("prompt_second", "STRING"), ("strict_empty", "BOOLEAN"), ("allow_secondary_humans", "BOOLEAN"),
          ("picture3_present", "BOOLEAN")],
         [("first_pass_prompt", "STRING"), ("second_pass_prompt", "STRING"), ("scene_fields", "STRING"),
          ("filtered_fields", "STRING"), ("picture3_contract", "STRING"), ("reference_policy", "STRING"),
          ("report", "STRING")], (2620, 80), (400, 330), []),
        (718, "H3CShotTrim", "08  Trim to Original Shot Frames",
         [("decoded", "IMAGE"), ("original_f", "INT"), ("work_l", "INT"), ("shot_number", "INT")],
         [("trimmed", "IMAGE"), ("frame_count", "INT"), ("report", "STRING")], (4970, 80), (300, 155), []),
        (719, "H3COrderedMergeStep", "09  Ordered Merge Step",
         [("previous", "*"), ("shot_frames", "IMAGE"), ("shot_number", "INT"),
          ("total_frames", "INT"), ("shot_range", "STRING"), ("policy_report", "STRING"),
          ("prompt_report", "STRING"), ("shot_report", "STRING")],
         [("state", "*"), ("report", "STRING")], (5300, 80), (300, 250), []),
        (720, "H3COrderedMergeFinish", "10  Ordered Timeline / FPS Check",
         [("state", "*"), ("fps", "FLOAT"), ("total_frames", "INT")],
         [("images", "IMAGE"), ("fps", "FLOAT"), ("frame_count", "INT"), ("report", "STRING")],
         (5930, 80), (330, 170), [24.0]),
    ]
    for spec in specs:
        n = make_node(*spec)
        nodes[n["id"]] = n

    # Easy-Use's current public source defines flow/index/value1 and a rawLink
    # flow + initial_value1 input. RH's installed version still needs live check.
    loop_outputs = [("flow", "FLOW_CONTROL"), ("index", "INT")]
    loop_outputs.extend((f"value{i}", "*") for i in range(1, 20))
    nodes[721] = make_node(721, "easy forLoopStart", "C Dynamic N-Shot Loop Start",
                           [("total", "INT"), ("initial_value1", "*")], loop_outputs,
                           (1470, 430), (340, 510), [1])
    nodes[722] = make_node(722, "easy forLoopEnd", "C Dynamic N-Shot Loop End",
                           [("flow", "FLOW_CONTROL"), ("initial_value1", "*")],
                           [(f"value{i}", "*") for i in range(1, 20)],
                           (5620, 80), (280, 125), [])
    nodes[723] = copy.deepcopy(nodes[628])
    nodes[723]["id"] = 723
    nodes[723]["title"] = "Second Pass Noise / Same Shot Seed"
    nodes[724] = copy.deepcopy(nodes[639])
    nodes[724]["id"] = 724
    nodes[724]["title"] = "First Pass Debug Decode (denoised)"
    nodes[725] = make_node(725, "PreviewImage", "FIRST PASS RESULT / DEBUG",
                           [("images", "IMAGE")], [], (4360, 440), (310, 320))
    nodes[726] = make_node(726, "PreviewImage", "SHOT RGB / DEBUG",
                           [("images", "IMAGE")], [], (1850, 390), (280, 260))
    nodes[727] = make_node(727, "PreviewImage", "DEPTH ref_video_0 / DEBUG",
                           [("images", "IMAGE")], [], (1850, 675), (280, 260))
    for i, title, pos in ((728, "RAW VLM OUTPUT", (1120, 620)),
                          (729, "NORMALISED SCENE RECORD", (1450, 620)),
                          (730, "SHOT REFERENCE POLICY", (2230, 490)),
                          (731, "FIRST + SECOND FINAL PROMPTS", (2630, 390)),
                          (732, "SHOT MANIFEST", (1470, 300))):
        nodes[i] = make_node(i, "PreviewAny", title, [("source", "*")], [("STRING", "STRING")],
                             pos, (300, 180))

    # Keep the original active source/model/sigma/VLM wiring that does not touch #701.
    links = []
    for link in data["links"]:
        if link[1] in nodes and link[3] in nodes and link[1] < 714 and link[3] < 714:
            if link[1] in {624, 625, 628, 629, 636, 637, 638, 639, 642, 646, 647}:
                continue
            if link[3] in {624, 625, 628, 629, 636, 637, 638, 639, 642, 646, 647}:
                continue
            if link[3] in {33, 704, 706}:
                continue
            links.append(list(link))

    def port(node_id, field, output):
        items = nodes[node_id]["outputs" if output else "inputs"]
        for index, item in enumerate(items):
            if item["name"] == field:
                return index
        raise KeyError(f"{node_id}.{field} {'output' if output else 'input'}")

    def wire(src, src_name, dst, dst_name):
        source_index = port(src, src_name, True)
        target_index = port(dst, dst_name, False)
        kind = nodes[src]["outputs"][source_index]["type"]
        links.append([0, src, source_index, dst, target_index, kind])

    # Existing source and analysis.
    for dst, field in ((714, "rgb"), (715, "rgb")):
        wire(22, "IMAGE", dst, field)
    for dst, field in ((714, "depth"), (715, "depth")):
        wire(601, "depth_maps", dst, field)
    wire(702, "FLOAT", 714, "cut_threshold")
    wire(703, "INT", 714, "min_shot_frames")
    wire(714, "manifest", 715, "manifest")
    wire(714, "manifest", 732, "source")
    wire(714, "shot_count", 721, "total")
    wire(721, "index", 715, "index")
    wire(715, "shot_rgb", 726, "images")
    wire(715, "depth_ref_video", 727, "images")
    wire(700, "raw_vlm_output", 728, "source")
    wire(700, "normalised_vlm_output", 729, "source")
    wire(700, "shot_prompts", 716, "shot_prompts")
    wire(700, "empty_shot_indices", 716, "empty_shot_indices")
    wire(700, "shot_prompts", 717, "shot_prompts")
    wire(715, "shot_number", 716, "shot_number")
    wire(714, "shot_count", 716, "shot_count")
    wire(715, "shot_seed", 716, "seed")
    wire(715, "shot_number", 717, "shot_number")
    wire(17, "IMAGE", 716, "picture1")
    wire(643, "IMAGE", 716, "picture2")
    wire(713, "image", 716, "picture3")
    wire(716, "report", 730, "source")
    wire(716, "strict_empty", 717, "strict_empty")
    wire(716, "allow_secondary_humans", 717, "allow_secondary_humans")
    wire(716, "picture3_present", 717, "picture3_present")
    wire(712, "STRING", 717, "prompt_first")
    wire(707, "STRING", 717, "prompt_second")
    wire(717, "report", 731, "source")

    # Two visible H3 conditioning and sampling passes; first pass's denoised
    # output alone feeds split/upscale. Second pass's output alone feeds decode.
    for h3, prompt in ((629, "first_pass_prompt"), (646, "second_pass_prompt")):
        wire(627, "CLIP", h3, "clip")
        wire(620, "VAE", h3, "vae")
        wire(717, prompt, h3, "prompt")
        wire(619, "width", h3, "width")
        wire(619, "height", h3, "height")
        wire(715, "work_l", h3, "length")
        for picture in (1, 2, 3):
            wire(716, f"picture{picture}", h3, f"ref_images.ref_image_{picture - 1}")
    wire(715, "depth_ref_video", 629, "ref_videos.ref_video_0")
    for noise in (628, 723):
        wire(715, "shot_seed", noise, "noise_seed")
    for guider, h3 in ((625, 629), (647, 646)):
        wire(645, "MODEL", guider, "model")
        wire(h3, "positive", guider, "conditioning")
    for sampler, noise, guider, sigma in ((624, 628, 625, "high_sigmas"),
                                         (636, 723, 647, "low_sigmas")):
        wire(noise, "NOISE", sampler, "noise")
        wire(guider, "GUIDER", sampler, "guider")
        wire(622, "SAMPLER", sampler, "sampler")
        wire(635, sigma, sampler, "sigmas")
    wire(629, "LATENT", 624, "latent_image")
    wire(624, "denoised_output", 637, "av_latent")
    wire(624, "denoised_output", 724, "samples")
    wire(620, "VAE", 724, "vae")
    wire(724, "IMAGE", 725, "images")
    wire(637, "video_latent", 642, "latent")
    wire(642, "latent", 638, "video_latent")
    wire(637, "audio_latent", 638, "audio_latent")
    wire(638, "latent", 636, "latent_image")
    wire(636, "output", 639, "samples")
    wire(620, "VAE", 639, "vae")
    wire(639, "IMAGE", 718, "decoded")
    wire(715, "original_f", 718, "original_f")
    wire(715, "work_l", 718, "work_l")
    wire(715, "shot_number", 718, "shot_number")
    wire(721, "value1", 719, "previous")
    wire(718, "trimmed", 719, "shot_frames")
    wire(715, "shot_number", 719, "shot_number")
    wire(714, "total_frames", 719, "total_frames")
    wire(715, "range", 719, "shot_range")
    wire(716, "report", 719, "policy_report")
    wire(717, "report", 719, "prompt_report")
    wire(718, "report", 719, "shot_report")
    wire(721, "flow", 722, "flow")
    wire(719, "state", 722, "initial_value1")
    wire(722, "value1", 720, "state")
    wire(714, "total_frames", 720, "total_frames")
    wire(720, "images", 704, "images")
    wire(720, "fps", 704, "fps")
    wire(704, "VIDEO", 33, "video")
    wire(720, "report", 706, "source")

    # Reassign link IDs and rebuild both endpoint references from one source of truth.
    for n in nodes.values():
        for inp in n["inputs"]:
            inp["link"] = None
        for out in n["outputs"]:
            out["links"] = None
    for link_id, item in enumerate(links, 1):
        _, src, source_index, dst, target_index, kind = item
        item[0] = link_id
        inp = nodes[dst]["inputs"][target_index]
        if inp["link"] is not None:
            raise ValueError(f"duplicate input link {dst}.{inp['name']}")
        inp["link"] = link_id
        out = nodes[src]["outputs"][source_index]
        if out["links"] is None:
            out["links"] = []
        out["links"].append(link_id)

    # A compact readable canvas; existing node sizes remain intact, except the
    # VLM and two H3 nodes whose new visible ports need additional height.
    positions = {
        22: (80, 80), 8: (365, 80), 643: (670, 80), 713: (1230, 80),
        17: (365, 680), 712: (800, 640), 707: (800, 920),
        600: (40, 1510), 601: (40, 1650), 9: (40, 1770), 12: (40, 1870),
        626: (450, 1510), 631: (720, 1510), 632: (1100, 1510),
        633: (1350, 1510), 634: (1600, 1510), 645: (1840, 1510),
        620: (400, 1660), 627: (650, 1660), 622: (900, 1660),
        623: (1130, 1680), 641: (1360, 1680), 635: (1590, 1680),
        619: (400, 1790),
        700: (1650, 80), 728: (1650, 870), 729: (1970, 870),
        705: (1650, 1080), 714: (2180, 80), 732: (2180, 320),
        715: (2180, 530), 702: (2280, 820), 703: (2280, 920),
        716: (2570, 80), 730: (2570, 690), 717: (2570, 890),
        731: (2570, 1240), 721: (2990, 80), 726: (2990, 620),
        727: (2990, 900),
        629: (3360, 80), 628: (3360, 490), 625: (3360, 600),
        624: (3710, 80), 637: (3960, 80), 642: (3960, 170),
        724: (3960, 420), 725: (3960, 510), 638: (4310, 80),
        646: (4310, 200), 647: (4310, 610), 723: (4310, 720),
        636: (4660, 80), 639: (4930, 80),
        718: (5220, 80), 719: (5550, 80), 722: (5880, 80),
        720: (6190, 80), 704: (6550, 80), 706: (6550, 340),
        33: (6840, 80),
    }
    if set(positions) != set(nodes):
        raise ValueError(f"layout missing IDs: {sorted(set(nodes) - set(positions))}")
    for node_id, point in positions.items():
        nodes[node_id]["pos"] = list(point)
    nodes[700]["size"] = [500, 760]
    for node_id in (629, 646):
        nodes[node_id]["size"] = [330, 380]
    rectangles = []
    for n in nodes.values():
        x, y = n["pos"][:2]
        w, h = n["size"][:2]
        for other_id, ox, oy, ow, oh in rectangles:
            if x < ox + ow and ox < x + w and y < oy + oh and oy < y + h:
                raise ValueError(f"layout overlap: {n['id']} and {other_id}")
        rectangles.append((n["id"], x, y, w, h))

    # Dense, separate overview groups. Group geometry has no execution effect.
    data["groups"] = [
        {"id": 1, "title": "01  INPUT / REFERENCE MEDIA", "bounding": [40, 40, 1590, 1140], "color": "#304650", "font_size": 24, "flags": {}},
        {"id": 2, "title": "02  SHOT ANALYSIS / POLICY", "bounding": [1620, 40, 1710, 1410], "color": "#394756", "font_size": 24, "flags": {}},
        {"id": 3, "title": "03  SHARED H3 RESOURCES", "bounding": [10, 1470, 2540, 820], "color": "#353e49", "font_size": 24, "flags": {}},
        {"id": 4, "title": "04  TWO-PASS NATIVE H3 CHAIN", "bounding": [3320, 40, 1880, 850], "color": "#443e55", "font_size": 24, "flags": {}},
        {"id": 5, "title": "05  TRIM / ORDERED OUTPUT", "bounding": [5190, 40, 2080, 600], "color": "#314252", "font_size": 24, "flags": {}},
    ]
    data["nodes"] = sorted(nodes.values(), key=lambda n: n["id"])
    data["links"] = links
    data["last_node_id"] = max(nodes)
    data["last_link_id"] = len(links)
    extra = data.setdefault("extra", {})
    extra["candidate_variant"] = "C_MODULAR_RH_AB_PENDING"
    extra["candidate_notes"] = (
        "Observable per-shot C Modular chain; DepthCrafter feeds first-pass H3 "
        "ref_video_0; Picture3 is optional; RH live A/B remains pending."
    )
    extra["h3_scene_vlm_candidate"]["active_nodes"] = [
        "H3ShotSceneVLM", "H3OptionalPicture3", "H3CShotPlanner",
        "H3CShotSelectPad", "H3CShotPolicy", "H3CPromptCompiler",
        "H3CShotTrim", "H3COrderedMergeStep", "H3COrderedMergeFinish",
    ]
    extra["h3_scene_vlm_candidate"]["picture3"] = (
        "optional upload; connected to policy; None when no image is selected"
    )
    extra["h3_scene_vlm_candidate"]["detector_contract"] = (
        "Planner N = VLM N = Policy N = Easy-Use loop total N"
    )
    TARGET.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(nodes), len(links)


def build_a2_force_refs():
    """Copy C; append missing VLM sockets and set the sample Policy override."""
    data = json.loads(TARGET.read_text(encoding="utf-8"))
    vlm = next(n for n in data["nodes"] if n["id"] == 700 and n["type"] == "H3ShotSceneVLM")
    _ensure_scene_vlm_outputs(vlm)
    policy = next(n for n in data["nodes"] if n["id"] == 716 and n["type"] == "H3CShotPolicy")
    if [x["name"] for x in policy["inputs"]][-2:] != [
        "manual_force_empty_indices", "manual_force_present_indices"
    ]:
        raise ValueError("formal C Policy input contract changed")
    if policy.get("widgets_values") != ["", ""]:
        raise ValueError("formal C Policy widget contract changed")
    policy["widgets_values"][1] = A2_FORCE_PRESENT
    data.setdefault("extra", {})["candidate_variant"] = "C_MODULAR_A2_FORCE_REFS_RH_AB_PENDING"
    data["extra"]["a2_force_present_indices"] = A2_FORCE_PRESENT
    A2_TARGET.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(data["nodes"]), len(data["links"])


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--a2-force-refs"]:
        print("A2 nodes, links =", build_a2_force_refs())
    elif not sys.argv[1:]:
        print("nodes, links =", build())
    else:
        raise SystemExit("usage: build_h3_modular_c.py [--a2-force-refs]")
