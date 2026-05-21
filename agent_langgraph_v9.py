import time
import json
import os
from turtle import st
from matplotlib import colors
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"
import hashlib
import threading
import shutil
import re
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'Tools', 'nnUNet'))
import time
import base64
from pathlib import Path
import gradio as gr
import numpy as np
import plotly.graph_objects as go
import SimpleITK as sitk
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from dotenv import load_dotenv
from openai import OpenAI
from skimage import measure
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors as RL
from reportlab.platypus import ListFlowable, ListItem
from Tools.diameter_v4 import AortaAnalysis
from Tools.segmentation import AortaSegmentation
from Tools.super_resolution_v2 import SuperResolution
from disk_ttl import DiskTTLCache

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from langchain_google_community import GoogleSearchAPIWrapper

from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import TypedDict, Literal, Any, Optional, List, Dict

from langgraph.graph import StateGraph, END
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

@dataclass
class StudyState:
    """
    Container for all data and intermediate results associated with a single CT study.

    Each study (e.g., A or B in multi-study mode) maintains its own independent state,
    allowing parallel processing and comparison without interference.

    This includes:
    - Raw and processed image data
    - Segmentation outputs
    - Diameter measurements
    - Cached visualizations (meshes)
    - Pipeline execution status flags

    The goal is to make the pipeline:
    - Stateful (no recomputation)
    - Modular (each step depends on this state)
    - Reusable (supports iterative user interaction)
    """

    # image + geometry
    original_path: Optional[str] = None
    image_path: Optional[str] = None
    sitk_image: Optional[sitk.Image] = None
    image_data: Optional[np.ndarray] = None
    spacing: Optional[Tuple[float, float, float]] = None

    # original vs SR
    orig_image: Optional[sitk.Image] = None
    orig_spacing: Optional[Tuple[float, float, float]] = None
    sr_image: Optional[sitk.Image] = None
    sr_spacing: Optional[Tuple[float, float, float]] = None

    # outputs
    segmented_mask: Optional[np.ndarray] = None
    segmented_mask_orig: Optional[np.ndarray] = None
    diameter_result_dict: Optional[Dict[int, float]] = None

    # misc
    last_selected_class_ids: Optional[List[int]] = None
    last_report_path: Optional[str] = None
    last_impression: Optional[str] = None

    mesh_cache: Dict[Any, Any] = field(default_factory=dict)
    seg_version: int = 0

    pipeline_state: Dict[str, bool] = field(default_factory=lambda: {
        "sr_done": False,
        "seg_done": False,
        "diam_done": False,
        "abn_done": False,
        "report_done": False,
    })



class AortaGraphState(TypedDict, total=False):
    user_text: str
    route: Literal["web", "imaging"]

    tool_type: str
    class_ids: List[int]

    assistant_text: str
    fig: Any
    needs_clarification: bool
    clarification_question: str
    pending_clarification: dict


def _is_iliac_ambiguous(text: str) -> tuple[bool, dict]:
    """
    Detect whether a user query involving 'iliac' arteries is underspecified.

    Why this matters:
    - Iliac arteries have multiple branches (common, external)
    - They are also bilateral (left/right)
    - The segmentation/measurement tools require precise mapping to class IDs

    This function checks:
    - Whether side (left/right/both) is specified
    - Whether branch (common/external) is specified
    - Whether unsupported anatomy (e.g., internal iliac) is requested

    Returns:
        (needs_clarification, info_dict)

    info_dict may contain:
        - missing_side: bool
        - missing_branch: bool
        - internal_mentioned: bool

    This is used to trigger a clarification step before executing the pipeline.
    """
    t = (text or "").lower()

    if "iliac" not in t:
        return False, {}

    # If user is NOT asking for seg/diam, don’t gate
    triggers = ("segment", "segmentation", "mask", "diameter", "measure")
    if not any(w in t for w in triggers):
        return False, {}

    side_present = any(w in t for w in ("left", "right", "both", "bilateral"))
    branch_present = any(w in t for w in ("common", "external"))

    # If they mention "internal iliac", we still clarify because your tool set doesn't include it
    internal_mentioned = "internal" in t

    missing_side = not side_present
    missing_branch = (not branch_present) or internal_mentioned

    if missing_side or missing_branch:
        return True, {
            "missing_side": missing_side,
            "missing_branch": missing_branch,
            "internal_mentioned": internal_mentioned,
        }

    return False, {}

def _clarify_gate_node(agent, state: AortaGraphState) -> AortaGraphState:
    """
    Intercepts user input before execution to ensure anatomical specificity.

    If the request is ambiguous (e.g., "show iliac diameters"):
        → the system pauses execution
        → asks a targeted clarification question
        → stores the pending request for later resolution

    This prevents:
    - incorrect segmentation/measurement
    - mapping errors between text and class IDs

    If no ambiguity:
        → passes state forward unchanged

    Output:
        Updates state with:
        - needs_clarification (bool)
        - clarification_question (str)
        - assistant_text (response to user)
    """    
    msg = (state.get("user_text") or "").strip()

    needs, info = _is_iliac_ambiguous(msg)
    if not needs:
        state["needs_clarification"] = False
        return state

    # Build question
    parts = []
    if info.get("missing_side"):
        parts.append("side (left, right, or both)")
    if info.get("missing_branch"):
        parts.append("branch (common or external)")

    question = "I can do that — quick clarification: do you mean " + " and ".join(parts) + " iliac?"

    state["needs_clarification"] = True
    state["clarification_question"] = question
    state["pending_clarification"] = {
        "original_user_text": msg,
        "info": info,
    }
    state["assistant_text"] = question
    state["fig"] = None
    return state


def _route_node(agent, state: AortaGraphState) -> AortaGraphState:
    """
    Determines whether the user query should be handled by:
        1) Imaging pipeline (segmentation, diameters, etc.)
        2) Web / knowledge pipeline (guidelines, papers, etc.)

    Logic:
    - Uses heuristic classifier (_looks_like_web_query)
    - If query is knowledge-based → route = 'web'
    - Otherwise → route = 'imaging'

    This is critical for enabling:
    - Hybrid system behavior (analysis + knowledge)
    - Proper separation of responsibilities

    Output:
        state["route"] ∈ {"web", "imaging"}
    """    
    msg = (state.get("user_text") or "").strip()
    if msg and agent.auto_web_search and agent._looks_like_web_query(msg):
        state["route"] = "web"
    else:
        state["route"] = "imaging"
    return state

def _web_node(agent, state: AortaGraphState) -> AortaGraphState:
    """
    Handles knowledge-based queries using external search + LLM reasoning.

    Steps:
    - Perform web search (Google / LangChain)
    - Generate grounded response with citations
    - Return textual answer (no visualization)

    Used for:
    - Guidelines (e.g., AAA thresholds)
    - Research questions
    - Definitions / background knowledge

    Output:
        state["assistant_text"] = answer
        state["fig"] = None
    """    
    state["assistant_text"] = agent.web_search_answer_langchain(state.get("user_text", ""))
    state["fig"] = None
    return state

def _imaging_intent_node(agent, state: AortaGraphState) -> AortaGraphState:
    """
    Converts natural language input into a structured tool call.

    Uses LLM (process_command) to extract:
        - tool_type (e.g., 'seg', 'diameter', 'super_resolution')
        - class_ids (target anatomical regions)

    This is the bridge between:
        unstructured text → deterministic pipeline execution

    Output:
        state["tool_type"]
        state["class_ids"]
        state["tool_call"]
    """    
    msg = state.get("user_text", "")
    # history handling: for now, keep it minimal (empty history)
    resp = agent.process_command(msg, history=[])

    tool_call = agent.safe_json_loads(resp) or {}
    tool_type = tool_call.get("tools", "")
    class_ids = tool_call.get("params", []) or []

    state["tool_call"] = tool_call
    state["tool_type"] = tool_type
    state["class_ids"] = class_ids
    state["assistant_text"] = ""   # will be filled by SR/SEG/DIAM nodes or fallback
    state["fig"] = None
    return state

def _sr_node(agent, state):
    """
    Executes (or ensures) super-resolution preprocessing.

    Triggered when:
    - Input CT has anisotropic spacing (e.g., thick slices)

    Behavior:
    - Runs SR only if needed
    - Updates pipeline state
    - Returns status message

    Note:
    This step improves downstream segmentation accuracy.
    """
    sid = agent.active_sid
    msg = agent.ensure_super_resolved(sid)
    state["assistant_text"] = msg
    state["fig"] = None
    return state

def _seg_node(agent, state):
    """
    Executes (or ensures) aortic segmentation.

    Behavior:
    - Runs segmentation if not already computed
    - Optionally generates 3D mesh visualization for selected regions

    Output:
        - segmentation mask stored in state
        - optional Plotly 3D figure

    This is the foundational step for all downstream analysis.
    """    

    sid = agent.active_sid
    msg = agent.ensure_segmentation(sid)

    class_ids = state.get("class_ids", []) or []
    fig = agent.create_3d_mesh(class_ids, sid=sid) if class_ids else None
    state["assistant_text"] = f"{msg}\nSegmented regions: {class_ids}" if class_ids else msg
    state["fig"] = fig
    return state

def _diam_node(agent, state):
    """
    Computes cross-sectional diameters for segmented regions.

    Requirements:
    - Segmentation must already exist

    Behavior:
    - Runs diameter calculation if not cached
    - Generates structured HTML table
    - Optionally generates 3D mesh visualization

    Output:
        - diameter_result_dict
        - formatted table for UI
    """    
    sid = agent.active_sid
    msg = agent.ensure_diameters(sid)

    class_ids = state.get("class_ids", []) or []
    agent.last_selected_class_ids = class_ids[:]  # ok

    fig = agent.create_3d_mesh(class_ids, sid=sid) if class_ids else None
    table_html = agent.render_diameter_table_html(class_ids)

    state["assistant_text"] = f"{msg}\n{table_html}"
    state["fig"] = fig
    return state




def _fallback_node(agent, state: AortaGraphState) -> AortaGraphState:
    state["assistant_text"] = "__IMAGING__"
    state["fig"] = None
    return state


def _imaging_next(agent, state) -> str:
    """
    Determines the next execution step based on:
        - requested tool (seg, diameter, SR)
        - current pipeline state

    This function encodes dependency logic:
        - segmentation may require SR
        - diameter requires segmentation
        - avoids redundant computation

    Returns:
        Next node name in LangGraph
        (e.g., 'seg', 'sr_then_diam', 'seg_then_diam')
    """    
    tool_type = (state.get("tool_type") or "").strip()

    if tool_type == "super_resolution":
        return "sr"

    if tool_type == "seg":
        return "sr_then_seg" if _needs_sr(agent) else "seg"

    if tool_type == "diameter":
        if _needs_sr(agent):
            return "sr_then_diam"
        st = agent.S(agent.active_sid)
        if st.segmented_mask is None:
            return "seg_then_diam"

        return "diam"

    return "fallback"

def _needs_sr(agent) -> bool:
    try:
        st = agent.S(agent.active_sid)
        return st.image_path is not None and st.sr_image is None and st.spacing and st.spacing[2] != 1
    except Exception:
        return False


def _sr_then_seg(agent, state):
    sid = agent.active_sid
    agent.ensure_super_resolved(sid)
    return _seg_node(agent, state)

def _seg_then_diam(agent, state):
    sid = agent.active_sid
    agent.ensure_segmentation(sid)
    return _diam_node(agent, state)

def _sr_then_diam(agent, state):
    sid = agent.active_sid
    agent.ensure_super_resolved(sid)
    agent.ensure_segmentation(sid)
    return _diam_node(agent, state)



def build_aorta_graph_step3(agent):
    """
    Constructs the LangGraph execution pipeline.

    The graph consists of:
        - routing node (web vs imaging)
        - clarification gate
        - intent parsing
        - execution nodes (SR, SEG, DIAM)
        - fallback handling

    Key properties:
    - Directed graph (not linear pipeline)
    - Conditional edges based on state
    - Ensures correct execution order

    This is the central orchestration mechanism of AortaGPT.
    """    
    g = StateGraph(AortaGraphState)

    g.add_node("route", lambda s: _route_node(agent, s))
    g.add_node("web", lambda s: _web_node(agent, s))

    g.add_node("imaging_intent", lambda s: _imaging_intent_node(agent, s))

    g.add_node("sr", lambda s: _sr_node(agent, s))
    g.add_node("seg", lambda s: _seg_node(agent, s))
    g.add_node("diam", lambda s: _diam_node(agent, s))
    g.add_node("sr_then_seg", lambda s: _sr_then_seg(agent, s))
    g.add_node("seg_then_diam", lambda s: _seg_then_diam(agent, s))
    g.add_node("sr_then_diam", lambda s: _sr_then_diam(agent, s))

    g.add_node("fallback", lambda s: _fallback_node(agent, s))

    g.set_entry_point("route")

    g.add_node("clarify_gate", lambda s: _clarify_gate_node(agent, s))

    g.add_conditional_edges(
        "route",
        lambda s: s.get("route", "imaging"),
        {"web": "web", "imaging": "clarify_gate"},
    )

    g.add_conditional_edges(
        "clarify_gate",
        lambda s: "clarify" if s.get("needs_clarification") else "imaging",
        {"clarify": END, "imaging": "imaging_intent"},
    )


    g.add_conditional_edges(
        "imaging_intent",
        lambda s: _imaging_next(agent, s),
        {
            "sr": "sr",
            "seg": "seg",
            "diam": "diam",
            "sr_then_seg": "sr_then_seg",
            "seg_then_diam": "seg_then_diam",
            "sr_then_diam": "sr_then_diam",
            "fallback": "fallback",
        },
    )

    g.add_edge("web", END)
    g.add_edge("sr", END)
    g.add_edge("seg", END)
    g.add_edge("diam", END)
    g.add_edge("sr_then_seg", END)
    g.add_edge("seg_then_diam", END)
    g.add_edge("sr_then_diam", END)
    g.add_edge("fallback", END)

    return g.compile()



def _extract_urls(text: str):
    if not text:
        return []
    # crude but effective URL finder
    urls = re.findall(r"https?://[^\s\)>\]]+", text)
    # dedupe while preserving order
    seen = set()
    out = []
    for u in urls:
        u2 = u.strip().rstrip(".,;)")
        if u2 not in seen:
            seen.add(u2)
            out.append(u2)
    return out

def _format_citations_block(urls):
    if not urls:
        return ""
    # simple numbered citations
    lines = ["\n\n**Sources**"]
    for i, u in enumerate(urls[:8], 1):  # cap to 8
        dom = urlparse(u).netloc.replace("www.", "")
        lines.append(f"[{i}] {dom} — {u}")
    return "\n".join(lines)

load_dotenv()

web_cache = DiskTTLCache(cache_path="outputs/web_cache.json", ttl_seconds=30 * 60)





def google_search_cached(query: str) -> str:
    # 1) cache hit
    cached = web_cache.get(query)
    if cached is not None:
        return cached

    # 2) fetch with retry
    last_err = None
    for attempt in range(2):
        try:
            result = google_search.run(query)
            web_cache.set(query, result)
            return result
        except Exception as e:
            last_err = e
            time.sleep(0.4 * (attempt + 1))

    # 3) stale fallback (helps with "Response ended prematurely")
    stale = web_cache.get_stale(query)
    if stale is not None:
        return stale + "\n\n[Note: served cached result due to temporary web error]"

    raise RuntimeError(f"Error while using web search: {last_err}")


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
pretrained_model_path = os.getenv("PRETRAINED_MODEL_PATH")
highres_model_path = os.getenv("HIGHRES_MODEL_PATH")

# --------- Google / LangChain web-search agent wiring ---------


# --------- Google / Modern LangChain tool calling wiring ---------
google_llm = None  # global bound model with tools

if GOOGLE_API_KEY and GOOGLE_CSE_ID and OPENAI_API_KEY:
    try:
        google_search = GoogleSearchAPIWrapper(
            google_api_key=GOOGLE_API_KEY,
            google_cse_id=GOOGLE_CSE_ID,
        )

        @tool
        def google_search_tool(query: str) -> str:
            """Search Google for factual info and return snippets/links."""
            return google_search_cached(query)

        google_llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0.2,
            api_key=OPENAI_API_KEY,
        ).bind_tools([google_search_tool])

    except Exception as e:
        print(f"[WARN] Failed to initialize modern Google tool calling: {e}")
        google_llm = None
else:
    print("[INFO] Google search not configured (missing GOOGLE_API_KEY / GOOGLE_CSE_ID / OPENAI_API_KEY)")
# --------------------------------------------------------------

def data_url(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


class AortaAgent:

    """
    Core system controller that manages:
        - Image data and processing state
        - Model execution (SR, segmentation, diameter)
        - Visualization generation
        - Web search integration
        - Report generation

    Responsibilities:
        - Maintain per-study state (A/B)
        - Execute pipeline steps safely (GPU locking)
        - Provide user-facing outputs (text + figures)
        - Support iterative and interactive workflows

    Acts as the main interface between:
        UI (Gradio) ↔ Models ↔ LangGraph orchestration
    """    
    def __init__(self):
        self.orig_image = None
        self.orig_spacing = None
        self.sr_image = None
        self.sr_spacing = None      
        self.image_data = None
        self.image_path = None
        self.sitk_image = None
        self.segmented_mask = None
        self.segmented_mask_orig = None  # mask in original-resolution space
        self.last_selected_class_ids = None
        self.last_report_path = None
        self.last_impression = None  
        self.mesh_cache = {}      # key -> plotly.Figure
        self.seg_version = 0      # bump whenever segmentation updates
        self.classes = ['1', '3','5', '7', '8', '9', '10', '12', '14', '17', '18', '19', '22', '23']
        self.classes = [int(c) for c in self.classes]  # Convert class IDs to integers
        # self.classes = ['1', '3','5', '7', '8', '9', '10', '12', '14', '17', '18', '19', '22', '23']
        self.auto_web_search = True   # set False if you want manual only
        self.guidelines_only = True
        self.web_search_enabled = True   # master on/off switch
        self.last_web_sources = []       # stores last URLs for citations/debug
        self.google_search = google_search  # use the global GoogleSearchAPIWrapper (if configured)
        self.pipeline_state = {
            "sr_done": False,
            "seg_done": False,
            "diam_done": False,
            "abn_done": False,      # NEW: abnormality analysis completed
            "report_done": False,   # NEW: PDF report generated
        }
        self.gpu_lock = threading.RLock()

        self.MAIN_AORTA_ZONES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        
        self.ZONE_COLORS = {
            0:  "rgba(0,0,0,0)",   # transparent
            1:  "#0000FF",         # blue
            2:  "#90EE90",         # light green
            3:  "#800080",         # purple
            4:  "#A9A9A9",         # dark gray
            5:  "#006400",         # dark green
            6:  "#00FFFF",         # cyan
            7:  "#F5F5DC",         # beige
            8:  "#FFC0CB",         # pink
            9:  "#FF0000",         # red
            10: "#8B0000",         # dark red
            11: "#FFA500",         # orange
            12: "#FF8C00",         # dark orange
            13: "#5C4033",         # brown
            14: "#FFFF00",         # yellow
            15: "#8b5bde",         # violet/lavender
            16: "#D2B48C",         # tan
            17: "#FF00FF",         # magenta
            18: "#D3D3D3",         # light gray
            19: "#808080",         # gray
            20: "#00008B",         # dark blue
            21: "#52a8ff",         # light blue
            22: "#F0E68C",         # khaki
            23: "#50C878",         # emerald (extra if needed)
}
        self.AORTIC_GROUPS = {
            "Ascending/arch":  [1, 3, 5],
            "Descending": [7, 8, 9],
            "Visceral":   [10, 12, 14],
            "Infrarenal": [17],
            "Iliacs":     [18, 19, 22, 23],
        }

        self.class_mapping = {
            "1": "Zone 0 (Main Aorta)",
            "2": "Innominate",
            "3": "Zone 1 (Main Aorta)",
            "4": "Left Common Carotid",
            "5": "Zone 2 (Main Aorta)",
            "6": "Left Subclavian Artery",
            "7": "Zone 3 (Main Aorta)",
            "8": "Zone 4 (Main Aorta)",
            "9": "Zone 5 (Main Aorta)",
            "10": "Zone 6 (Main Aorta)",
            "11": "Celiac Artery",
            "12": "Zone 7 (Main Aorta)",
            "13": "SMA",
            "14": "Zone 8 (Main Aorta)",
            "15": "Right Renal Artery",
            "16": "Left Renal Artery",
            "17": "Zone 9 (Main Aorta)",
            "18": "Zone 10 R (Right Common Iliac Artery)",
            "19": "Zone 10 L (Left Common Iliac Artery)",
            "20": "Right Internal Iliac Artery",
            "21": "Left Internal Iliac Artery",
            "22": "Zone 11 R (Right External Iliac Artery)",
            "23": "Zone 11 L (Left External Iliac Artery)",
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.super_res_tool = SuperResolution(highres_model_path, device=self.device)
        self.seg_tool = AortaSegmentation(pretrained_model_path, device=self.device)
        self.diameter_tool = AortaAnalysis()
        self.studies: Dict[str, StudyState] = {
            "A": StudyState(),
            "B": StudyState(),
        }
        self.active_sid: str = "A"  # default target


    def S(self, sid: str | None = None) -> StudyState:
        sid = sid or self.active_sid
        return self.studies[sid]

    def set_active(self, sid: str):
        if sid in self.studies:
            self.active_sid = sid

    def render_diameter_table_html(self, class_ids: list[int], sid: str | None = None) -> str:

        """
        Render diameter measurements as an HTML table for the Gradio interface.

        The table is grouped by anatomical region:
            - Ascending / arch
            - Descending
            - Visceral
            - Infrarenal
            - Iliacs

        For each selected class ID, the function displays:
            - anatomical group
            - segment name
            - measured diameter in millimeters
            - guideline-context note

        Inputs:
            class_ids:
                List of anatomical class IDs selected by the user or command parser.
            sid:
                Study slot identifier ("A" or "B"). Defaults to the active study.

        Output:
            HTML string rendered directly in the UI.

        Notes:
            This function does not compute diameters. It only formats already-computed
            values stored in st.diameter_result_dict.
        """

        st = self.S(sid)
        if not st.diameter_result_dict:
            return "<em>No diameters available.</em>"


        rows = []
        rows.append(
            '<table style="border-collapse:collapse;width:100%;background:#111;'
            'border:1px solid #2a2a2a;border-radius:10px;overflow:hidden;">'
            '<thead><tr>'
            '<th style="text-align:center;padding:8px 10px;border-bottom:1px solid #2a2a2a;color:white;">Group</th>'
            '<th style="text-align:center;padding:8px 10px;border-bottom:1px solid #2a2a2a;color:white;">Segment</th>'
            '<th style="text-align:center;padding:8px 10px;border-bottom:1px solid #2a2a2a;color:white;">Diameter (mm)</th>'
            '<th style="text-align:center;padding:8px 10px;border-bottom:1px solid #2a2a2a;color:white;">'
            'Guideline context<br>'
            '<span style="font-size:10px;opacity:0.6;">(diameter-only perspective; other clinical factors may still matter)</span>'
            '</th>'
            '</tr></thead><tbody>'
        )

        any_row = False
        for group_name, group_cids in self.AORTIC_GROUPS.items():
            group_rows = []
            for cid in group_cids:
                if cid not in class_ids:
                    continue
                if cid not in st.diameter_result_dict:
                    continue

                diameter = float(st.diameter_result_dict[cid])
                region_name = self.class_mapping.get(str(cid), f"Class {cid}")
                color = self.color_for_class_id(cid)
                note = self._guideline_note_for_diameter(cid, diameter)

                group_rows.append(
                    (
                        f'<td style="padding:6px 10px;text-align:center;">'
                        f'<span style="color:{color};font-weight:600">{region_name}</span>'
                        f"</td>",
                        f'<td style="padding:6px 10px;text-align:center;">'
                        f'<span style="color:{color}">{diameter:.2f}</span></td>',
                        f'<td style="padding:6px 10px;text-align:left;font-size:11px;color:#cccccc;">'
                        f'{note}</td>'
                    )
                )

            if group_rows:
                any_row = True
                rowspan = len(group_rows)
                first = True
                for seg_cell, dia_cell, note_cell in group_rows:
                    rows.append("<tr>")
                    if first:
                        rows.append(
                            f'<td rowspan="{rowspan}" style="padding:6px 10px;text-align:center;'
                            f'font-weight:700;vertical-align:middle;color:white;">{group_name}</td>'
                        )
                        first = False
                    rows.append(seg_cell)
                    rows.append(dia_cell)
                    rows.append(note_cell)
                    rows.append("</tr>")

        rows.append("</tbody></table>")
        if not any_row:
            return "<em>No diameters available for the selected classes.</em>"
        return "".join(rows)


    def ensure_super_resolved(self, sid: str = "A") -> str:
        st = self.S(sid)

        if st.image_path is None:
            return f"[Image {sid}] Please upload a NIfTI image first."

        try:
            if st.sr_image is not None:
                st.pipeline_state["sr_done"] = True
                return f"[Image {sid}] Super-resolution already available."
            if st.spacing and st.spacing[2] == 1:
                st.pipeline_state["sr_done"] = True
                return f"[Image {sid}] Super-resolution not needed (isotropic z-spacing)."
        except Exception:
            pass

        _, _, _, msg = self.super_resolve(sid)
        return f"[Image {sid}] {msg}"


    def ensure_segmentation(self, sid: str = "A") -> str:
        """
        Ensures segmentation exists for the given study.

        Behavior:
        - Checks if segmentation is already computed
        - If not:
            → optionally runs SR first
            → runs segmentation model
            → caches result
        - If yes:
            → reuses existing result

        This pattern avoids redundant computation and ensures
        consistent pipeline behavior.
        """        
        st = self.S(sid)

        if st.image_path is None:
            return f"[Image {sid}] Please upload a NIfTI image first."

        with self.gpu_lock:
            try:
                if st.sr_image is None and st.spacing and st.spacing[2] != 1:
                    self.ensure_super_resolved(sid)

                if st.segmented_mask is None:
                    st.segmented_mask = self.seg_tool.segment_image(st.image_path)
                    st.pipeline_state["seg_done"] = True
                    st.seg_version += 1
                    st.mesh_cache.clear()

                    self._sync_original_mask(sid)

                    # free SR memory optional
                    if st.sr_image is not None:
                        st.sr_image = None
                        st.sr_spacing = None
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    return f"[Image {sid}] Segmentation completed."

                return f"[Image {sid}] Segmentation already available."
            finally:
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass



    def ensure_diameters(self, sid: str = "A") -> str:
        """
        Ensures diameter measurements are available.

        Behavior:
        - Ensures segmentation exists first
        - Computes diameters only if not cached
        - Converts voxel measurements to millimeters

        Output:
            Updates st.diameter_result_dict
        """        
        st = self.S(sid)

        if st.image_path is None:
            return f"[Image {sid}] Please upload a NIfTI image first."

        self.ensure_segmentation(sid)

        if st.diameter_result_dict is None:
            diameter_result, *_ = self.diameter_tool.calculate_all_diameters(st.segmented_mask)
            diameter_result_mm = [i * st.spacing[0] for i in diameter_result]
            st.diameter_result_dict = dict(zip(self.classes, diameter_result_mm))
            st.pipeline_state["diam_done"] = True
            return f"[Image {sid}] Diameter calculation completed."

        return f"[Image {sid}] Diameters already available."



    def run_segmentation(self, sid: str) -> str:
        """Ensure segmentation exists for the given slot (A/B)."""
        st = self.S(sid)

        # 1) Must have either a path or an in-memory volume for this slot
        if getattr(st, "image_path", None) is None and getattr(st, "volume", None) is None:
            return "To perform this action, please upload a NIfTI image first."

        # 2) Super-resolution per slot (if you use SR)
        #    Use st.spacing/st.sr_image, NOT self.spacing/self.sr_image
        if getattr(st, "sr_image", None) is None and getattr(st, "spacing", None) is not None:
            if st.spacing[2] != 1:
                # IMPORTANT: make ensure_super_resolved accept sid and write to st
                self.ensure_super_resolved(sid=sid)

        # 3) Segmentation per slot
        if getattr(st, "segmented_mask", None) is None:
            # If your seg tool needs a path:
            # mask = self.seg_tool.segment_image(st.image_path)
            #
            # If your seg tool can take a volume (better):
            # mask = self.seg_tool.segment_volume(st.volume)

            mask = self.seg_tool.segment_image(st.image_path)
            st.segmented_mask = mask

            st.pipeline_state["seg_done"] = True
            st.seg_version = getattr(st, "seg_version", 0) + 1

            # Per-slot cache
            st.mesh_cache = getattr(st, "mesh_cache", {})
            st.mesh_cache.clear()

            # IMPORTANT: make _sync_original_mask accept sid and use st
            self._sync_original_mask(sid=sid)

            return f"Segmentation completed for Image {sid}."

        return f"Segmentation already available for Image {sid}."


    def run_diameters(self, sid: str) -> str:
        """Ensure diameter_result_dict exists for this slot."""
        st = self.S(sid)

        if getattr(st, "image_path", None) is None and getattr(st, "volume", None) is None:
            return "To perform this action, please upload a NIfTI image first."

        # Ensure segmentation for THIS sid (and SR if your seg step depends on it)
        self.run_segmentation(sid)

        # Make sure segmentation actually exists in st
        if getattr(st, "segmented_mask", None) is None:
            return f"Segmentation is required before diameters for Image {sid}."

        if getattr(st, "diameter_result_dict", None) is None:
            diameter_result, *_ = self.diameter_tool.calculate_all_diameters(st.segmented_mask)

            # spacing must also be per-slot
            if getattr(st, "spacing", None) is None:
                return f"Missing spacing metadata for Image {sid}."

            diameter_result_mm = [d * st.spacing[0] for d in diameter_result]

            st.diameter_result_dict = dict(zip(self.classes, diameter_result_mm))
            st.pipeline_state["diam_done"] = True

            return f"Diameter calculation completed for Image {sid}."

        return f"Diameters already available for Image {sid}."


    def _confidence_tag_from_search(self, raw: str, urls: list[str]) -> tuple[str, str]:
        if not raw or len(raw.strip()) < 200:
            return "INSUFFICIENT", "Search results were too sparse to support a grounded answer."

        if not urls:
            return "INSUFFICIENT", "No source URLs were present in the search results."

        preferred = (
            "vascular.org", "esvs.org", "ahajournals.org", "acc.org", "jacc.org",
            "escardio.org", "nice.org.uk", "cdc.gov", "nih.gov",
            "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
            "nejm.org", "jamanetwork.com", "thelancet.com", "bmj.com",
            "sciencedirect.com", "springer.com", "nature.com", "oup.com", "wiley.com",
        )

        def dom(u: str) -> str:
            try:
                return urlparse(u).netloc.replace("www.", "")
            except Exception:
                return ""

        domains = [dom(u) for u in urls]
        pref_hits = sum(1 for d in domains if any(d == p or d.endswith("." + p) for p in preferred))

        if pref_hits >= 2:
            return "HIGH", "Multiple guideline/journal sources were found."
        if pref_hits == 1 and len(raw) > 800:
            return "HIGH", "At least one guideline/journal source with sufficient supporting text."
        if pref_hits == 1 or len(urls) >= 3:
            return "MODERATE", "Some supporting sources were found, but evidence is limited or not fully consistent."

        return "INSUFFICIENT", "Not enough high-quality supporting sources were found."


    def web_search_answer_langchain(self, query: str) -> str:
        global google_llm, google_search_tool

        if google_llm is None:
            return "Web search is not configured."

        q = query
        if getattr(self, "guidelines_only", False):
            q = self._apply_guideline_filters(query)

        system = (
            "You are a factual assistant. Use the Google search tool when needed. "
            "Base your answer only on tool results. If results are insufficient, say so. "
            "Keep it concise."
        )

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=q),
        ]

        tool_map = {"google_search_tool": google_search_tool}

        # We'll keep all raw tool outputs here for scoring/citations
        tool_raw_chunks: list[str] = []

        for _ in range(2):
            ai = google_llm.invoke(messages)
            messages.append(ai)

            tool_calls = getattr(ai, "tool_calls", None) or []
            if not tool_calls:
                # No tool calls → answer came directly (often means it didn't search)
                # Treat as MODERATE at best unless query was trivial
                tag = "MODERATE"
                rationale = "No web tool was used in this response."
                ans = (ai.content or "").strip() or "I couldn't produce an answer."
                return f"**Confidence: {tag}** — {rationale}\n\n{ans}"

            for tc in tool_calls:
                name = tc.get("name")
                args = tc.get("args") or {}
                call_id = tc.get("id")

                fn = tool_map.get(name)
                if fn is None:
                    messages.append(ToolMessage(content=f"Tool not found: {name}", tool_call_id=call_id))
                    continue

                try:
                    result = fn.invoke(args)
                except Exception as e:
                    result = f"Tool error: {e}"

                tool_raw_chunks.append(str(result))
                messages.append(ToolMessage(content=str(result), tool_call_id=call_id))

        # Final answer
        messages.append(HumanMessage(content="Now answer using the tool results you already have."))
        ai = google_llm.invoke(messages)
        answer = (ai.content or "").strip() or "I couldn't produce an answer."

        raw = "\n\n".join(tool_raw_chunks)
        urls = _extract_urls(raw)

        tag, rationale = self._confidence_tag_from_search(raw, urls)

        return f"**Confidence: {tag}** — {rationale}\n\n{answer}"
    

    def answer_with_google_citations(self, question: str) -> str:
        """
        Uses GoogleSearchAPIWrapper to fetch results, then asks your OpenAI model
        to synthesize an answer that includes citations.
        """
        if not self.web_search_enabled:
            return "Web search is currently disabled."

        try:
            raw = self.google_search.run(question)  # string of results/snippets
        except Exception as e:
            return f"Error while using web search: {e}"

        urls = _extract_urls(raw)
        self.last_web_sources = urls

        synthesis_system = (
            "You are a helpful assistant. Use the provided web search snippets to answer.\n"
            "Rules:\n"
            "- Be concise.\n"
            "- If you claim a fact, it must be supported by the snippets.\n"
            "- Include citations as [1], [2], ... referencing the Sources list you will provide.\n"
            "- Do not invent URLs.\n"
        )

        synthesis_user = (
            f"Question: {question}\n\n"
            f"Web snippets:\n{raw}\n\n"
            "Write the answer with inline citations like [1] and then a Sources list."
        )

        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                temperature=0.2,
                max_tokens=450,
                messages=[
                    {"role": "system", "content": synthesis_system},
                    {"role": "user", "content": synthesis_user},
                ],
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            # fallback: show raw search + sources
            return f"Search results:\n{raw}\n{_format_citations_block(urls)}\n\n(LLM synthesis failed: {e})"

        # If the model forgot to include a sources list, append one.
        if "Sources" not in answer and "sources" not in answer.lower():
            answer += _format_citations_block(urls)

        return answer


    def _looks_like_web_query(self, text: str) -> bool:
            """
            Heuristic: returns True if the user is asking for external facts / papers / guidelines / latest info,
            and NOT asking to run a tool on the uploaded CT (seg/diam/sr/report/slice UI, etc.).
            """
            if not text or not isinstance(text, str):
                return False

            t = text.strip().lower()

            # If user explicitly wants web/search, always do it
            if t.startswith(("web:", "google:", "search:")):
                return True

            # If user is clearly asking to use AortaGPT imaging tools, do NOT web-search
            tool_keywords = [
                "segment", "segmentation", "mask", "overlay", "mesh",
                "diameter", "measure", "centerline",
                "super resolution", "super-resolution", "enhance resolution", "sr",
                "export", "pdf", "report",
                "slice", "axial", "sagittal", "coronal",
                "ct view", "nii", "nifti", "mha", "upload"
            ]
            if any(k in t for k in tool_keywords):
                return False

            # If no image is loaded, most "analysis" questions are probably knowledge questions
            # (you can still chat about your system without web-search though).
            # We'll use additional cues below.

            # Strong web-ish cues: guidelines, papers, "latest", policy, "what is", etc.
            web_cues = [
                "guideline", "svs", "esvs", "acc", "aha",
                "paper", "publication", "doi", "pmid", "pubmed",
                "trial", "study", "meta-analysis", "systematic review",
                "threshold", "criteria", "recommendation",
                "most recent", "latest", "new", "updated", "as of",
                "who is", "when did", "what is", "definition of", "difference between",
                "incidence", "prevalence", "statistics",
                "icd", "cpt",
            ]
            
            if any(k in t for k in web_cues):
                return True

            # Question-mark style factual Qs that are not about the uploaded data
            if "?" in t:
                # common “knowledge question” starters
                starters = ("what", "why", "how", "who", "when", "where", "which")
                if t.lstrip().startswith(starters):
                    return True

            return False    


    def _maybe_auto_web_search(self, command: str) -> str | None:
        """
        Returns a web-search answer if we decide to route automatically; otherwise returns None.
        """
        if not self.auto_web_search:
            return None

        if self._looks_like_web_query(command):
            return self.web_search_answer_langchain(command)


        return None



    def color_for_class_id(self, class_id: int) -> str:
        """
        Return the color hex for a given segmentation class_id.
        Falls back to gray if not found.
        """
        return self.ZONE_COLORS.get(class_id, "#bbbbbb")

    def render_colored_diameter_message(self, diameter_result, category_func):
        lines = ["Sure, here is the diameter calculation result."]
        for zone_label, value_mm in diameter_result:  # iterable of (label, value)
            color = self.color_for_class_id(zone_label)
            cat = category_func(zone_label)  # your “— Main Aorta …” suffix if any
            lines.append(
                f'<span style="color:{color};font-weight:600">{zone_label}</span>: '
                f'<span style="color:{color}">{value_mm:.2f} mm</span>{cat}'
            )
        return "\n".join(lines)

    def zone_category_suffix(self, zone_label: str) -> str:
        """
        Return a suffix for the zone label, e.g. ' — Main Aorta (box)'.
        Handles 'Zone 10', 'Zone 10 L', 'Zone 10 R', etc.
        """
        m = re.search(r"Zone\s+(\d+)", zone_label)
        if not m:
            return ""
        zid = int(m.group(1))
        if zid in self.MAIN_AORTA_ZONES:
            return " (**Main Aorta**)"
        return ""


    def load_nifti(self, file_path: str, sid: str = "A") -> str:
        """
    Load a CT volume from a NIfTI or MHA file into the selected study slot.

    Supported formats:
        - .nii
        - .nii.gz
        - .mha

    Steps:
        1. Validate file type.
        2. Read image using SimpleITK.
        3. Convert image to NumPy array for processing.
        4. Extract voxel spacing.
        5. Reset the selected StudyState slot with the new image.

    Inputs:
        file_path:
            Path to the uploaded CT volume.
        sid:
            Study slot identifier ("A" or "B").

    Output:
        Status message describing the loaded image size and spacing.

    Important:
        Loading a new image resets all previous outputs for that slot, including
        segmentation masks, diameter results, meshes, and reports.
    """        
        st = self.S(sid)

        if file_path is None:
            return "Error: No file provided."
        if not file_path.lower().endswith((".nii", ".nii.gz", ".mha")):
            return "Error: Unsupported file format. Please upload a .nii, .nii.gz, or .mha file."

        try:
            img = sitk.ReadImage(file_path)
            vol = sitk.GetArrayFromImage(img)
            sp = img.GetSpacing()

            # Reset this study slot fully
            self.studies[sid] = StudyState(
                original_path=file_path,
                image_path=file_path,
                sitk_image=img,
                image_data=vol,
                spacing=sp,
                orig_image=img,
                orig_spacing=sp,
            )

            st = self.S(sid)
            return (
                f"[Image {sid}] Loaded NIfTI with shape {st.sitk_image.GetSize()} "
                f"and spacing ({st.spacing[0]}mm, {st.spacing[1]}mm, {st.spacing[2]}mm)."
            )
        except Exception as e:
            return f"[Image {sid}] Error loading NIfTI file: {str(e)}"


    def create_3d_mesh(self, class_ids, sid: str | None = None):
        """
        Generate an interactive 3D mesh visualization for selected segmented regions.

        This function converts a multi-class segmentation mask into a Plotly 3D mesh
        using marching cubes.

        Steps:
            1. Select requested anatomical class IDs.
            2. Build a union mask to find the tight bounding box.
            3. Crop the segmentation mask for efficient rendering.
            4. Optionally downsample very large crops.
            5. Run marching cubes for each selected class.
            6. Convert coordinates from NumPy order (Z, Y, X) to Plotly order (X, Y, Z).
            7. Add color-coded mesh traces to a Plotly figure.
            8. Cache the resulting figure for reuse.

        Inputs:
            class_ids:
                Anatomical segmentation IDs to visualize.
            sid:
                Study slot identifier.

        Output:
            Plotly Figure object, or None if no valid segmentation/class is available.

        Notes:
            - Meshes are cached using image path, segmentation version, and requested IDs.
            - The segmentation version is incremented whenever a new mask is produced.
            - This prevents stale visualizations after segmentation updates.
        """        
        st = self.S(sid)
        if st.segmented_mask is None or st.sitk_image is None:
            return None

        req_ids = tuple(sorted(int(c) for c in class_ids))
        cache_key = (st.image_path, st.seg_version, req_ids)
        if cache_key in st.mesh_cache:
            return st.mesh_cache[cache_key]

        sx, sy, sz = st.spacing
        mask_all = st.segmented_mask
        # 1) Build a union mask to find tight bbox
        req_mask = np.isin(mask_all, req_ids)
        if not np.any(req_mask):
            # nothing to show
            return None

        z_idx, y_idx, x_idx = np.where(req_mask)
        z0, z1 = int(z_idx.min()), int(z_idx.max()) + 1
        y0, y1 = int(y_idx.min()), int(y_idx.max()) + 1
        x0, x1 = int(x_idx.min()), int(x_idx.max()) + 1

        # pad a little for nicer framing
        pad = 4
        z0 = max(0, z0 - pad); y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
        z1 = min(mask_all.shape[0], z1 + pad)
        y1 = min(mask_all.shape[1], y1 + pad)
        x1 = min(mask_all.shape[2], x1 + pad)

        sub = mask_all[z0:z1, y0:y1, x0:x1]
        # Optional decimation for huge crops (keeps scale correct by scaling spacing)
        step = 1
        max_dim = max(sub.shape)
        if max_dim > 320:
            step = 2
            sub = sub[::step, ::step, ::step]
            sx, sy, sz = st.spacing  # SITK order (x,y,z)
            spacing_for_marching_cubes = (sz, sy, sx)  # because vol is (Z,Y,X)

            # later, if step=2:
            spacing_for_marching_cubes = tuple(s * step for s in spacing_for_marching_cubes)

        fig = go.Figure()
        palette = [
            "rgba(0,0,0,0)", "#0000FF", "#90EE90", "#800080", "#A9A9A9", "#006400",
            "#00FFFF", "#F5F5DC", "#FFC0CB", "#FF0000", "#8B0000", "#FFA500",
            "#FF8C00", "#5C4033", "#FFFF00", "#8b5bde", "#D2B48C", "#FF00FF",
            "#D3D3D3", "#808080", "#00008B", "#52a8ff", "#F0E68C", "#50C878",
        ]

        for cid in req_ids:
            if cid >= len(palette): 
                continue
            sub_c = (sub == cid)
            if not np.any(sub_c):
                continue

            verts, faces, _, _ = measure.marching_cubes(sub_c, level=0, spacing=spacing_for_marching_cubes)

            # crop origin in physical coordinates (Z,Y,X)
            origin_zyx_mm = np.array([z0, y0, x0]) * np.array(spacing_for_marching_cubes)

            # convert verts (Z,Y,X) -> (X,Y,Z) for plotly
            vz, vy, vx = verts[:, 0], verts[:, 1], verts[:, 2]
            vx += origin_zyx_mm[2]
            vy += origin_zyx_mm[1]
            vz += origin_zyx_mm[0]

            x, y, z = vx, vy, vz


            i, j, k = faces.T
            region_name = self.class_mapping.get(str(cid), f"Class {cid}")
            fig.add_trace(
                go.Mesh3d(
                    x=x, y=y, z=z, i=i, j=j, k=k,
                    color=palette[cid], opacity=0.65,
                    hovertext=region_name, hoverinfo="text"
                )
            )

        # Brighter dark theme & good default camera
        fig.update_layout(
            title="3D Segmentation",
            scene=dict(
                aspectmode="data",
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                zaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                bgcolor="#111418",
            ),
            margin=dict(l=0, r=0, t=36, b=0),
            paper_bgcolor="#111418",
        )
        fig.update_scenes(camera={"eye": {"x": 2.0, "y": -2.2, "z": 1.2}})

        # Cache & return
        self.mesh_cache[cache_key] = fig
        return fig

    def generate_pdf_report(self, selected_class_ids=None, sid: str | None = None):
        """
        Generate a structured PDF report for the selected study.

        The report includes:
            - AortaGPT report title
            - scan metadata
            - voxel spacing
            - study slot identifier
            - zone-based diameter table
            - guideline-context notes
            - abnormality analysis, if available
            - clinical disclaimer

        Inputs:
            selected_class_ids:
                Optional list of class IDs to include in the report.
                If None, the function uses the last selected regions or all available
                diameter results.
            sid:
                Study slot identifier ("A" or "B").

        Output:
            Absolute path to the generated PDF file.

        Behavior:
            - If no diameters exist, a minimal report is still generated.
            - If abnormality analysis has not been run, the report explains that
            this section is unavailable.
            - Report files are saved under the outputs directory with a timestamp.

        Important:
            This function is report-generation only. It does not run segmentation,
            diameter calculation, or abnormality analysis automatically.
        """
        sid = sid or getattr(self, "active_sid", "A")
        st = self.S(sid)

        # ----------------------------
        # Nothing computed yet
        # ----------------------------
        if st.diameter_result_dict is None or len(st.diameter_result_dict) == 0:
            out_dir = Path(getattr(self, "output_dir", "outputs"))
            out_dir.mkdir(parents=True, exist_ok=True)

            pdf_path = out_dir / f"report_{sid}_{time.strftime('%Y%m%d-%H%M%S')}.pdf"
            doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
            styles = getSampleStyleSheet()

            story = [
                Paragraph("<b>AortaGPT Report</b>", styles["Title"]),
                Spacer(1, 8),
                Paragraph("No diameters have been computed yet.", styles["Normal"]),
            ]
            doc.build(story)

            st.last_report_path = str(pdf_path)
            if hasattr(st, "pipeline_state"):
                st.pipeline_state["report_done"] = True
            return str(pdf_path)

        # ----------------------------
        # Filter which classes to include
        # ----------------------------
        if selected_class_ids is None:
            selected_class_ids = (
                st.last_selected_class_ids
                if st.last_selected_class_ids
                else list(st.diameter_result_dict.keys())
            )

        # normalize to ints + keep only those present in this slot
        selected_class_ids = [
            int(c) for c in selected_class_ids
            if int(c) in st.diameter_result_dict
        ]

        out_dir = Path(getattr(self, "output_dir", "outputs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / f"report_{sid}_{time.strftime('%Y%m%d-%H%M%S')}.pdf"

        # ----------------------------
        # Build content
        # ----------------------------
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )
        styles = getSampleStyleSheet()

        title = Paragraph("<b>AortaGPT Diameter Report</b>", styles["Title"])

        meta_lines = []
        # patient/study name
        if getattr(st, "original_path", None):
            meta_lines.append(f"Patient: {os.path.basename(st.original_path)}")

        # voxel spacing
        if getattr(st, "spacing", None) is not None:
            sp = st.spacing
            meta_lines.append(
                f"Voxel spacing (x,y,z): {sp[0]:.3f} / {sp[1]:.3f} / {sp[2]:.3f} mm"
            )

        meta_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        meta_lines.append(f"Study slot: {sid}")

        story = [title, Spacer(1, 6)]
        for line in meta_lines:
            story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 12))

        data = [[
            Paragraph("<b>Group</b>", styles["BodyText"]),
            Paragraph("<b>Segment</b>", styles["BodyText"]),
            Paragraph("<b>Diameter (mm)</b>", styles["BodyText"]),
            Paragraph(
                "<b>Guideline context</b><br/><font size='8' color='gray'>"
                "(diameter-only perspective; other clinical factors may still matter)"
                "</font>",
                styles["BodyText"],
            ),
        ]]

        # (kept — in case you want to use it later)
        def _para_colored(text, hex_color, bold=False):
            txt = f'<font color="{hex_color}">{text}</font>'
            if bold:
                txt = f"<b>{txt}</b>"
            return Paragraph(txt, styles["BodyText"])

        any_row = False

        # Build grouped rows consistent with UI table
        for group_name, group_cids in self.AORTIC_GROUPS.items():
            group_rows = []
            for cid in group_cids:
                if cid not in selected_class_ids:
                    continue
                if cid not in st.diameter_result_dict:
                    continue

                dval = float(st.diameter_result_dict[cid])
                region_name = self.class_mapping.get(str(cid), f"Class {cid}")
                note = self._guideline_note_for_diameter(cid, dval)

                group_rows.append([
                    Paragraph(group_name, styles["BodyText"]),
                    Paragraph(region_name, styles["BodyText"]),
                    Paragraph(f"{dval:.2f}", styles["BodyText"]),
                    Paragraph(note, styles["BodyText"]),
                ])

            if group_rows:
                any_row = True
                data.extend(group_rows)

        if not any_row:
            story.append(Paragraph("No diameters available for the selected classes.", styles["Italic"]))
            doc.build(story)
            st.last_report_path = str(pdf_path)
            if hasattr(st, "pipeline_state"):
                st.pipeline_state["report_done"] = True
            return str(pdf_path)

        # ----------------------------
        # Create table
        # ----------------------------
        t = Table(data, colWidths=[100, None, 70, None])

        ts = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), RL.black),
            ("TEXTCOLOR", (0, 0), (-1, 0), RL.whitesmoke),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("LINEABOVE", (0, 0), (-1, 0), 1, RL.gray),
            ("LINEBELOW", (0, 0), (-1, 0), 1, RL.gray),

            ("ALIGN", (0, 1), (-1, -1), "CENTER"),
            ("VALIGN", (0, 1), (-1, -1), "MIDDLE"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.25, RL.grey),
        ])

        # Rowspans for Group column
        r = 1
        while r < len(data):
            group_text = data[r][0].getPlainText()
            start = r
            while r + 1 < len(data) and data[r + 1][0].getPlainText() == group_text:
                r += 1
            end = r
            if end > start:
                ts.add("SPAN", (0, start), (0, end))
                ts.add("VALIGN", (0, start), (0, end), "MIDDLE")
            r += 1

        t.setStyle(ts)
        story.append(t)

        # ----------------------------
        # Abnormality Analysis section
        # ----------------------------
        story.append(Spacer(1, 14))
        story.append(Paragraph("<b>Abnormality Analysis</b>", styles["Heading2"]))

        if getattr(st, "last_impression", None):
            story.extend(self._markdown_to_flowables(st.last_impression, styles))
        else:
            story.append(
                Paragraph(
                    "No abnormality analysis available. "
                    "Use the <i>🧠 Abnormality Analysis</i> quick action first, then export the report again.",
                    styles["Italic"],
                )
            )

        # ----------------------------
        # Disclaimer
        # ----------------------------
        story.append(Spacer(1, 20))
        story.append(Paragraph("<b>Disclaimer</b>", styles["Heading2"]))
        story.append(Paragraph(
            "This document is not a formal medical report. "
            "It is an AI-generated summary intended to assist clinicians and researchers. "
            "Do not solely rely on it for diagnosis or treatment decisions. "
            "All findings must be verified and interpreted by qualified medical professionals.",
            styles["Italic"],
        ))

        doc.build(story)

        st.last_report_path = str(pdf_path)
        if hasattr(st, "pipeline_state"):
            st.pipeline_state["report_done"] = True

        return str(pdf_path)
    

    def _clean_impression_text(self, text: str) -> str:
        """
        Clean the LLM-generated abnormality impression before saving or exporting.

        Removes:
            - assistant-style closing lines
            - call-to-action language
            - unnecessary trailing sections
            - divider-separated extra content

        Input:
            text:
                Raw abnormality analysis generated by the language model.

        Output:
            Cleaned clinical-style impression text.

        Purpose:
            Keeps exported reports focused, professional, and free from conversational
            filler that is appropriate for chat but not for documentation.
        """
        if not text:
            return ""
        t = text

        # Hard cut after common dividers
        t = re.split(r"\n-{2,}\n|\n_{2,}\n|\n\*{2,}\n", t, maxsplit=1)[0]

        # Drop common CTA/help lines (line-based)
        lines = []
        for line in t.splitlines():
            s = line.strip()
            if re.match(r"^(if you want|i can help|let me know|click|use the quick action|happy to).*", s, flags=re.I):
                continue
            if re.match(r"^(need anything else|anything else|do you want me).*", s, flags=re.I):
                continue
            lines.append(line)
        t = "\n".join(lines).strip()

        return t


    def _markdown_to_flowables(self, text: str, styles):
        """
        Convert simple markdown-style text into ReportLab flowables.

        Supports:
            - paragraphs separated by blank lines
            - bullet lists beginning with '-' or '*'
            - preserved line breaks within paragraphs

        Inputs:
            text:
                Markdown-like abnormality analysis text.
            styles:
                ReportLab stylesheet object.

        Output:
            List of ReportLab flowables that can be inserted into a PDF story.

        Purpose:
            Allows LLM-generated markdown output to be placed cleanly inside
            the exported PDF report.
        """
        flow = []
        paras = re.split(r"\n\s*\n", text.strip())  # blank-line split

        for para in paras:
            # detect bullet block (all or most lines starting with -/*)
            lines = para.splitlines()
            bullet_lines = [re.match(r"^\s*[-*]\s+(.*)$", ln) for ln in lines]
            if bullet_lines and sum(1 for m in bullet_lines if m) >= max(2, len(lines) // 2):
                items = []
                for m in bullet_lines:
                    if not m:
                        continue
                    item_txt = m.group(1)
                    # basic HTML escapings already fine in Paragraph
                    items.append(ListItem(Paragraph(item_txt, styles["BodyText"]), leftIndent=12))
                flow.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=6))
                flow.append(Spacer(1, 6))
            else:
                # plain paragraph; preserve single line breaks with <br/>
                para_html = "<br/>".join([ln for ln in lines])
                flow.append(Paragraph(para_html, styles["BodyText"]))
                flow.append(Spacer(1, 6))

        return flow
    @staticmethod
    def _strip_html(text: str) -> str:
        """
        Remove basic HTML tags from a text string.

        This is used when HTML-formatted UI content needs to be converted into
        plain text for cleaner display, logging, or export.

        Input:
            text:
                HTML or mixed HTML/plain-text string.

        Output:
            Plain text with tags removed and spacing normalized.
        """        
        if not text:
            return ""
        # very simple sanitizer: remove tags, collapse spaces
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
        text = re.sub(r"</p\s*>", "\n\n", text, flags=re.I)
        text = re.sub(r"<[^>]+>", "", text)      # strip other tags
        text = re.sub(r"[ \t]+\n", "\n", text)   # trim trailing spaces
        text = re.sub(r"\n{3,}", "\n\n", text)   # collapse extra blank lines
        return text.strip()


    def _legend_html(self, class_ids):
        """Return an HTML legend for the selected class_ids."""
        items = []
        seen = set()
        for cid in class_ids:
            cid = int(cid)
            if cid in seen:
                continue
            seen.add(cid)
            name  = self.class_mapping.get(str(cid), f"Class {cid}")
            color = self.color_for_class_id(cid)
            items.append(
                "<div style='display:flex;align-items:center;gap:8px;"
                "padding:4px 8px;border-radius:6px;background:rgba(255,255,255,0.03);'>"
                f"<span style='width:12px;height:12px;border-radius:3px;background:{color};"
                "display:inline-block;border:1px solid rgba(255,255,255,0.2)'></span>"
                f"<span style='font-size:13px;color:#e6e6e6'>{name}</span>"
                "</div>"
            )
        return (
            "<div style='margin-top:8px;display:flex;flex-wrap:wrap;gap:6px;'>"
            + "".join(items) +
            "</div>"
        )

    # ---- Guideline-aware helpers (NEW) ----
    def _region_type_for_class(self, cid: int) -> str:
        """
        Rough mapping of class_id -> region type for guideline text.
        - 'thoracic': ascending/arch + descending segments
        - 'abdominal': infrarenal + iliacs (AAA context)
        """
        # Ascending/arch + descending thoracic
        thoracic_ids = {1, 3, 5, 7, 8, 9, 10, 12, 14}
        # AAA / infrarenal + iliacs
        abdominal_ids = {17, 18, 19, 22, 23}

        if cid in thoracic_ids:
            return "thoracic"
        if cid in abdominal_ids:
            return "abdominal"
        return "other"

    def _guideline_note_for_diameter(self, cid: int, d_mm: float) -> str:
        """
        Return a short text snippet describing how this diameter sits
        relative to common TEVAR/AAA guideline thresholds.
        We never recommend treatment; we only contextualize the number.
        """
        region_type = self._region_type_for_class(cid)

        # Generic thresholds (simplified):
        # Thoracic (descending DTA): ~55 mm
        # AAA (infrarenal/iliacs): ~55 mm for fusiform in low-risk patients
        threshold_mm = 55.0

        if d_mm >= threshold_mm:
            if region_type == "thoracic":
                return (
                    "⚠️ Diameter ≥ 5.5 cm, above typical elective repair threshold "
                    "for descending thoracic aneurysms in low-risk patients."
                )
            elif region_type == "abdominal":
                return (
                    "⚠️ Diameter ≥ 5.5 cm, above usual elective repair threshold "
                    "for fusiform infrarenal AAAs in low-risk patients."
                )
            else:
                return (
                    "⚠️ Diameter ≥ 5.5 cm, above common intervention thresholds "
                    "for major aortic segments."
                )
        elif d_mm >= (threshold_mm - 5.0):
            # 50–55 mm zone
            if region_type == "thoracic":
                return (
                    "⚠️ 5.0–5.5 cm range: near the typical threshold used for "
                    "elective repair of descending thoracic aneurysms."
                )
            elif region_type == "abdominal":
                return (
                    "⚠️ 5.0–5.5 cm range: near the usual threshold for elective "
                    "AAA repair in many guidelines."
                )
            else:
                return (
                    "⚠️ 5.0–5.5 cm range: near common intervention thresholds "
                    "for major aortic segments."
                )
        else:
            return (
                "✅ Below common diameter thresholds for elective repair "
                
            )
        
    def _build_diameter_context_markdown(self, sid: str | None = None) -> str:
        """
        Build a simple markdown summary of diameters grouped by AORTIC_GROUPS
        for a given study slot (A/B). Used as context for abnormality analysis.
        """
        sid = sid or self.active_sid
        st = self.S(sid)

        if not st.diameter_result_dict:
            return ""

        lines: list[str] = []
        for group_name, cids in self.AORTIC_GROUPS.items():
            lines.append(f"### {group_name}")
            for cid in cids:
                if cid not in st.diameter_result_dict:
                    continue
                label = self.class_mapping.get(str(cid), f"Class {cid}")
                d_mm = float(st.diameter_result_dict[cid])
                lines.append(f"- {label}: {d_mm:.2f} mm")
            lines.append("")  # blank line between groups

        return "\n".join(lines).strip()


    def _sync_original_mask(self, sid: str = "A"):
        st = self.S(sid)

        if st.segmented_mask is None:
            st.segmented_mask_orig = None
            return

        if st.sr_image is None or st.orig_image is None:
            st.segmented_mask_orig = st.segmented_mask
            return

        try:
            sr_mask_img = sitk.GetImageFromArray(st.segmented_mask.astype(np.uint8))
            sr_mask_img.CopyInformation(st.sr_image)

            resampled = sitk.Resample(
                sr_mask_img,
                st.orig_image,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                sr_mask_img.GetPixelID()
            )
            st.segmented_mask_orig = sitk.GetArrayFromImage(resampled)
        except Exception as e:
            print("[warn] Could not resample mask to original space:", e)
            st.segmented_mask_orig = None


    def _apply_guideline_filters(self, q: str) -> str:

        q = (q or "").strip()
        if not q:
            return q

        if not getattr(self, "guidelines_only", False):
            return q

        # High-signal guideline + literature sources
        allowed_sites = [
            # Societies / guideline homes
            "vascular.org",          # SVS
            "esvs.org",              # ESVS
            "ahajournals.org",       # AHA journals
            "acc.org",               # ACC
            "jacc.org",              # JACC
            "escardio.org",          # ESC (sometimes cross-ref guidelines)
            "nice.org.uk",           # NICE
            "cdc.gov",               # CDC (screening, epidemiology)
            "nih.gov",               # NIH / NCBI

            # PubMed / NCBI
            "pubmed.ncbi.nlm.nih.gov",
            "ncbi.nlm.nih.gov",

            "nejm.org",
            "jamanetwork.com",
            "thelancet.com",
            "bmj.com",
            "sciencedirect.com",
            "springer.com",
            "nature.com",
            "oup.com",
            "wiley.com",
        ]

        site_clause = " OR ".join([f"site:{d}" for d in allowed_sites])
        return f"{q} ({site_clause})"
        

    def web_search(self, query: str) -> str:
        """
        Runs Google search with disk TTL cache and optional 'guidelines only' restriction.
        Requires global google_search + web_cache.
        """
        q = self._apply_guideline_filters(query)

        # 1) cache hit
        cached = web_cache.get(q)
        if cached is not None:
            return cached

        # 2) retry
        last_err = None
        for attempt in range(2):
            try:
                result = google_search.run(q)
                web_cache.set(q, result)
                return result
            except Exception as e:
                last_err = e
                time.sleep(0.4 * (attempt + 1))

        # 3) stale fallback (helps "Response ended prematurely")
        stale = web_cache.get_stale(q)
        if stale is not None:
            return stale + "\n\n[Note: served cached result due to temporary web error]"

        raise RuntimeError(f"Error while using web search: {last_err}")


    def run_structured_abnormality_analysis(self, sid: str | None = None) -> str:
        sid = sid or self.active_sid
        st = self.S(sid)

        # FIX: slot-aware diameters
        if not st.diameter_result_dict:
            return (
                "No diameters are available yet. Please run the diameter "
                "calculation first, then request an abnormality analysis."
            )

        context_md = self._build_diameter_context_markdown(sid=sid)


        system_prompt = """
You are a vascular imaging assistant. You are given measured aortic diameters (in mm)
across the thoracic and abdominal aorta, organized by anatomical group.

Your task is to produce a concise abnormality report based ONLY on these diameters.

STRICT FORMAT (markdown):

## Summary
- 1–3 bullet points summarizing the overall picture (normal / mildly dilated / aneurysmal, and where).

## Ascending / arch
- 1–3 bullet points describing any dilation or aneurysm in the ascending and arch segments.
- Refer qualitatively to common thresholds (e.g., "above typical 5.5 cm threshold")
  but do NOT give explicit management recommendations.

## Descending
- 1–3 bullet points describing the descending thoracic aorta (zones 3–5 and nearby).
- Again, comment only on diameter / dilation patterns and how they relate to usual thresholds.

## Abdominal / iliacs
- 1–3 bullet points summarizing infrarenal abdominal aorta and iliac segments.
- Qualitative comparison to usual AAA thresholds is fine, but no direct treatment advice.

Constraints:
- Base your comments ONLY on the diameters and groups provided.
- Do NOT mention tools, models, or that you are an AI.
- Do NOT tell the user what to do clinically; no management or treatment recommendations.
- Keep the whole response under about 200–250 words.
- Do NOT restate the original diameter table.
"""

        user_prompt = (
            "Here are the measured aortic diameters (in mm), grouped by region:\n\n"
            f"{context_md}\n\n"
            "Based ONLY on these diameters, write the structured abnormality report "
            "following the exact markdown format described in the system prompt."
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=600,
            )

            text = response.choices[0].message.content
            # Store into the slot (not agent-level)
            st.last_impression = self._clean_impression_text(text)
            st.pipeline_state["abn_done"] = True

            return text

        except Exception as e:
            return (
                "Abnormality analysis could not be generated due to an internal error: "
                f"{str(e)}"
            )
    def web_search_answer(self, query: str) -> str:
        """
        Simple web search helper:
        1) Use GoogleSearchAPIWrapper to fetch top results.
        2) Ask the OpenAI model to summarize them concisely.

        Triggered by prefixes:
          - 'google:'
          - 'web:'
          - 'search:'
        """
        global google_search, client

        # 1) Run Google search
        try:
            # returns a list of dicts with 'title', 'link', 'snippet'
            results = google_search.results(query, num_results=5)
        except Exception as e:
            return f"Web search error: {e}"

        if not results:
            return "I couldn't find any web results for that query."

        # 2) Build a short context string
        context_lines = []
        for i, r in enumerate(results, start=1):
            title = r.get("title", "").strip()
            snippet = r.get("snippet", "").strip()
            link = r.get("link", "").strip()
            context_lines.append(
                f"{i}. {title}\n"
                f"   Snippet: {snippet}\n"
                f"   URL: {link}\n"
            )
        context_text = "\n".join(context_lines)

        # 3) Ask OpenAI to synthesize a concise answer
        system_prompt = (
            "You are a helpful assistant. You are given a web-search query and "
            "a few search results (titles, snippets, URLs). "
            "Based ONLY on that information, answer the user's question clearly "
            "and concisely in 2–4 sentences. If the results are unclear or conflicting, "
            "say so explicitly."
        )

        user_prompt = (
            f"User query:\n{query}\n\n"
            f"Top search results:\n{context_text}\n\n"
            "Now give a concise answer to the user's query."
        )

        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=400,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"Error while summarizing web results: {e}"

    # def web_search_answer(self, query: str) -> str:
    #     """
    #     Use the LangChain + Google search agent to answer general questions.
    #     Triggered when the user prefixes their message with:
    #       - 'google:'
    #       - 'web:'
    #       - 'search:'
    #     We post-process the agent's text so we only show the actual answer,
    #     not the intermediate "Action / Action Input" logs.
    #     """
    #     global google_agent_executor

    #     if google_agent_executor is None:
    #         return (
    #             "Web search is not configured.\n\n"
    #             "Please set GOOGLE_API_KEY, GOOGLE_CSE_ID, and "
    #             "HUGGINGFACEHUB_API_TOKEN in your environment to enable it."
    #         )

    #     try:
    #         result = google_agent_executor.invoke({"input": query})

    #         # LangChain classic AgentExecutor usually returns a dict.
    #         text = ""
    #         if isinstance(result, dict):
    #             # Prefer the standard 'output' key if present
    #             if "output" in result:
    #                 text = str(result["output"])
    #             else:
    #                 # Fallback: stringify the whole dict
    #                 text = str(result)
    #         else:
    #             text = str(result)

    #         text = text.strip()

    #         # Many structured-chat prompts end with "Final Answer: ...".
    #         # If that's present, strip everything before it.
    #         m = re.search(r"Final Answer:\s*(.*)", text, flags=re.S)
    #         if m:
    #             cleaned = m.group(1).strip()
    #             if cleaned:
    #                 return cleaned

    #         # Otherwise, just return the text as-is.
    #         return text
    #     except Exception as e:
    #         return f"Error while using web search: {e}"


    def process_command(self, command, history):


        system_message_template = """
    You are AortaGPT, an AI assistant specialized in analyzing aorta imaging data.
    You are an expert in medical image analysis, particularly focusing on aorta imaging.
    Your primary role is to assist with analyzing NIfTI files using specialized tools.

    ### Tools Summary
    - **seg** → segmentation; params = [class_ids]
    - **diameter** → diameter measurement; params = [class_ids]
    - **super_resolution** → resolution enhancement; params = []    

    ### Tool Usage Rules

    1.  **File Requirement:**
        - **Condition:** The user asks to perform an action that requires a tool (like segmentation, super resolution or diameter calculation), BUT the 'Image Path' provided in the context is `None`.
        - **Action:** You MUST NOT output the tool JSON. Instead, you must respond with a clear message politely asking the user to upload a NIfTI file first. For example: "To perform this action, please upload a NIfTI image first."

    2.  **Tool: segmentation**
        - **Condition:** An 'Image Path' IS provided, and the user asks to segment specific anatomical regions.
        - **Action:** You must understand the user's input to select the segmentation area. Output the corresponding class ids (as integers) from the JSON mapping file.
        - **Format:** {{"tools": "seg", "params": [class_id1, class_id2, ...]}}
        - **Example:** For "Zone 0" and "Zone 3" (class ids 1 and 7), output: {{"tools": "seg", "params": [1, 7]}}
        - **Default:** If the user asks for segmentation without specifying a region, return the segmentation maps for all regions.

    3.  **Tool: diameter**
        - **Condition:** An 'Image Path' IS provided, and the user asks to calculate the diameter.
        - **Action:** You must understand the user's input to select the calculation area. There are three supported regions. If the user's request falls outside these categories, return an empty list for params.
        - **Supported Regions and their Class IDs:**
        - **Main Aorta:** `[1, 3, 5, 7, 8, 9, 10, 12, 14, 17]`
        - **Left Common Iliac Artery:** `[19]`
        - **Right Common Iliac Artery:** `[18]`
        - **Left External Iliac Artery:** `[23]`
        - **Right External Iliac Artery:** `[22]`
        - **Default:** If the user asks for diameter without specifying a region, return all the diameters for all regions.
        - **Format:** {{"tools": "diameter", "params": [class_id1, class_id2, ...]}}
        - **Example (user asks for "main aorta" diameter):** {{"tools": "diameter", "params": [1, 3, 5, 7, 8, 9, 10, 12, 14, 17]}}
        - **Example (user asks for "left iliac artery" diameter):** {{"tools": "diameter", "params": [19]}}
        - **If the user says "iliac artery" (no side and no branch):Ask: "The iliac arteries have left and right sides, and common and external branches. Which do you want (for example: left common, right external, right internal etc.)?" Do not emit JSON in this turn.
        - **If the user makes a vague request like 'compute aortic diameters' or 'compute all aortic diameters', 
          you MUST return all the diameters for all regions.'
        - **Different aortic classes for different Class IDs:**
        - **Ascending/arch:** `[1, 3, 5]`
        - **Descending:** `[7, 8, 9]`
        - **Visceral:** `[10, 12, 14]`
        - **Infrarenal:** `[17]`
        - **Iliacs:** `[18, 19, 22, 23]`
        - **If the user asks for a specific category (e.g. "ascending aorta"), you MUST map it to the corresponding class ids and output the JSON.

    4.  **Tool: Super Resolution**
        - **Condition:** An 'Image Path' IS provided, and the user asks to enhance the image resolution.
        - **Action:** You must output the tool JSON with an empty params list.
        - **Format:** {{"tools": "super_resolution", "params": []}}

    5. **Error Handling**
        - If request cannot be mapped, return empty params and ask user to clarify.

        
    **Constraint for Tool Calls:** When calling a tool, ALWAYS use the class ids from the JSON mapping file for region selection. Do not output anything except the required JSON format for tool invocation.

    ### Alias & Region Mapping Rules

    When the user mentions an anatomical region, you MUST automatically convert that region into the correct list of class_ids using:
    1. The provided class mapping; and
    2. Standard anatomical terminology (never invent class_ids).

    ---

    ## 1. Automatically map all common aortic region names

    You must directly map the following phrases (and their variants) without asking for confirmation:

    - “ascending aorta”
    - “aortic arch” / “arch”
    - “descending thoracic aorta”
    - “thoracic aorta” (ascending + arch + descending)
    - “abdominal aorta” (visceral + infrarenal)
    - “visceral aorta”
    - “suprarenal aorta”
    - “infrarenal aorta”
    - “thoracoabdominal aorta”
    - “entire aorta” / “whole aorta”
    - “aorta” (unspecified)
    - Any combination (“descending and infrarenal”)

    These map directly to class_ids based on the provided mapping.

    ---

    ## 2. Recognize synonyms, plurals, and common phrasing variations

    Automatically normalize:

    - “upper aorta” → ascending + arch
    - “mid aorta” → descending thoracic
    - “lower aorta” → abdominal (visceral + infrarenal)
    - “belly aorta” / “stomach aorta” → abdominal aorta
    - “pelvic aorta” → iliac region (clarification required)
    - “renal level” → suprarenal vs infrarenal (ask if ambiguous)
    - “iliacs” → common iliacs (clarification on laterality needed)

    Plural, singular, and hyphen variants must all be interpreted correctly.
    ---
    ## 3. Combine regions when multiple are mentioned

    If the user names more than one region, you MUST merge class_ids into a single list.

    Example:
    “Measure the descending AND infrarenal aorta”
    → class_ids for descending + infrarenal (merged)

    ---
    ## 4. Clarify only when ambiguity is true

    Clarification is *only* required when the anatomy is incomplete or laterality is missing.

    You MUST ask a targeted clarification when the user says:

    - “iliac arteries”
    - “iliac artery”
    - “common iliac” (no side given)
    - “external iliac” (no side given)
    - “pelvic arteries”
    - “iliac region”
    - “iliac segment”

    Ask:
    “Do you want right, left, or both? Common or external iliac arteries?”

    NEVER ask about “zone IDs.” Ask only about anatomical details.    

    ---

    ## 5. NEVER ask for confirmation for standard aortic segments

    If the user specifies any known aortic region:
    - ascending
    - arch
    - descending
    - thoracic
    - abdominal
    - visceral
    - suprarenal
    - infrarenal
    - thoracoabdominal
    - entire aorta

    → You MUST map directly to class_ids without asking the user anything.

    ---
    ## 6. Map side-specific iliac branches without clarification

    If laterality is clear, map directly:

    - “right common iliac” → right common iliac class_id
    - “left external iliac” → left external iliac class_id
    - “right and left iliac arteries” → bilateral common iliacs
    - “bilateral common iliacs” → both common iliac class_ids
    - “bilateral external iliacs” → both external iliac class_ids

    No clarification needed.

    ---

    ## 7. Normalize vague or colloquial anatomical language

    Interpret these meaningfully:

    - “top of the aorta” → ascending
    - “bottom of the aorta” → infrarenal
    - “middle of the aorta” → descending thoracic
    - “above the kidneys” → suprarenal visceral
    - “below the kidneys” → infrarenal

    If the phrase could refer to both visceral or infrarenal, ask for clarification.

    ---

    ## 8. Never invent class_ids or modify the mapping

    - Only use class_ids explicitly defined in the given mapping.
    - Never create new numbers.
    - Never approximate or guess.
    - Never combine regions using unlisted IDs.
    ---
    **CONTEXT FOR THIS CONVERSATION:**

    - **Image Path:** {path}
    - **Class ID to Region Name Mapping:**
    {class_map}
    ---
    ### 🔍 ADDITIONAL REFERENCE FROM TEVAR GUIDELINES (Society for Vascular Surgery)

    **Scope of Guidelines:**
    - Applies to descending thoracic aortic aneurysms (DTA) only.
    - Excludes arch disease (zones 0–1), aortic dissection, and trauma.
    - Recommends TEVAR over open repair in most elective and emergent settings due to lower morbidity and mortality.

    **Imaging Recommendations Before/During/After TEVAR:**
    - Use fine-cut (≤0.25 mm) CTA of the entire aorta, iliac/femoral arteries, and head/neck to assess vertebral arteries.
    - Strongly recommend 3D centerline reconstruction for diameter and length accuracy.
    - Post-TEVAR imaging: CTA at 1 month, 12 months, then annually for life (or more often if abnormalities like endoleaks are detected).

    **Diameter Thresholds for TEVAR:**
    - Elective TEVAR recommended for DTA aneurysms >5.5 cm in low-risk patients.
    - Higher thresholds suggested for high-risk patients with comorbidities (renal failure, paraplegia risk).
    - Consider earlier intervention for saccular aneurysms or rapid expansion cases.

    **Access & Procedural Notes:**
    - Use percutaneous femoral access with ultrasound guidance when possible.
    - Use iliac conduits or endoconduits for small or tortuous iliac vessels.
    - Minimize catheter/wire manipulation in the arch or near visceral arteries to reduce embolic risk.

    **Spinal Cord & Branch Vessel Considerations:**
    - Recommend cerebrospinal fluid drainage and controlled hypertension for spinal cord protection in high-risk cases (extensive DTA coverage, vertebral/iliac disease).
    - Recommend LSA revascularization if:
    - LIMA-LAD bypass graft exists
    - Left vertebral artery is dominant
    - Left arm dialysis access present
    - Extensive DTA coverage planned (>15 cm)

    **Additional Best Practices:**
    - Prefer CT overlay and roadmapping during procedure to reduce contrast usage.
    - Recommend TEVAR over open repair for ruptured DTAs when anatomy allows.
    - Contrast-enhanced MRA is an option for patients with iodine allergy.
    - TEVAR is an option for PAU, IMH, mycotic aneurysms, Kommerell diverticula, and tumors, though data is limited.

    ### 🩺 ADDITIONAL REFERENCE FROM AAA GUIDELINES (Society for Vascular Surgery)

    **Scope of Guidelines:**
    - Applies to patients with **abdominal aortic aneurysm (AAA)**.
    - Covers full spectrum: diagnosis, surveillance, operative strategy, perioperative care, long-term follow-up, and complications.

    **Key Recommendations for Screening and Surveillance:**
    - **Ultrasound** is preferred for screening and surveillance.
    - One-time screening recommended for:
    - Men/women aged 65–75 with history of smoking.
    - First-degree relatives of AAA patients aged 65–75 or older in good health.
    - Surveillance intervals:
    - **3-year** for AAA: 3.0–3.9 cm
    - **12-month** for AAA: 4.0–4.9 cm
    - **6-month** for AAA: 5.0–5.4 cm
    - CTA or duplex ultrasound recommended 1 month after EVAR, then 12 months, then yearly if stable.

    **Diameter Thresholds and Timing for Repair:**
    - **Elective repair** recommended for:
    - Fusiform AAA ≥5.5 cm in low-risk patients.
    - Saccular aneurysms or symptomatic aneurysms regardless of size.
    - Women with AAA 5.0–5.4 cm may also be considered.
    - **Ruptured AAA**: Immediate intervention required.

    **Imaging and Diagnosis:**
    - Use **outer wall-to-outer wall measurements** perpendicular to aortic path.
    - Use **CT** for symptomatic patients or if aneurysm rupture is suspected.

    **EVAR-Specific Guidance:**
    - EVAR is preferred for ruptured AAAs when anatomy allows.
    - Maintain flow to **at least one internal iliac artery**.
    - Use FDA-approved branch endografts if applicable.
    - Minimum volume standards: ≥10 EVARs/year at center with <2% mortality/conversion rate.

    **Open Surgical Repair (OSR):**
    - Indicated when EVAR is not feasible.
    - Preferred approach: **retroperitoneal**, especially for inflammatory AAA, hostile abdomen, or horseshoe kidney.
    - Centers should perform ≥10 open aortic ops/year with mortality <5%.

    **Risk and Comorbidity Management:**
    - Assess pre-op cardiac risk via METs and ECG.
    - Delay OSR if drug-eluting coronary stent placed recently.
    - Encourage smoking cessation ≥2 weeks before repair.

    **Complication Management:**
    - **Endoleaks**:
    - Type I and III: Treat.
    - Type II: Treat if sac is expanding; otherwise monitor.
    - **Graft infection**: Dental prophylaxis required.
    - Prompt investigation for symptoms of sepsis, limb ischemia, or GI bleeding post-repair.

    **Postoperative Surveillance:**
    - CTA and duplex ultrasound within 1 month after EVAR.
    - Annual imaging if no complications.
    - Color duplex preferred if available; CT if not.

    **Economic and System-Level Considerations:**
    - Recommended:
    - Door-to-intervention time <90 minutes for ruptured AAA.
    - AAA repairs at high-volume centers to reduce mortality.
    - Use of VQI perioperative risk scores in shared decision-making.

    **Guideline Usage:** Guideline context is reference-only: never output these guidelines directly in tool JSON. Use them only for conversational answers when user asks about treatment, thresholds, or recommendations.

    """
        st = self.S(self.active_sid)
        system_message = system_message_template.format(
            path=st.image_path, class_map=json.dumps(self.class_mapping, indent=2)
        )

        try:
            messages = [{"role": "system", "content": system_message}]
            if history:
                # keep only the most recent 10 exchanges to avoid token overflow
                trimmed_history = history[-10:]
                for user_msg, assistant_msg in trimmed_history:
                    if user_msg:
                        messages.append({"role": "user", "content": user_msg})
                    if assistant_msg:
                        messages.append({"role": "assistant", "content": assistant_msg})
            messages.append({"role": "user", "content": command})


            response = client.chat.completions.create(
                # model="gpt-5-mini",
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error processing command: {str(e)}"

    def safe_json_loads(self, text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def super_resolve(self, sid: str = "A"):
        st = self.S(sid)

        if st.orig_image is None and st.sitk_image is not None:
            st.orig_image = st.sitk_image
            st.orig_spacing = st.sitk_image.GetSpacing()

        output = self.super_res_tool.enhance_resolution(
            st.image_path,
            scale=st.spacing[2]
        )

        img = output if isinstance(output, sitk.Image) else sitk.GetImageFromArray(output)
        new_shape = img.GetSize()

        out_dir = Path(getattr(self, "output_dir", "outputs"))
        out_dir.mkdir(parents=True, exist_ok=True)

        fname = f"super_resolved_{sid}_{time.strftime('%Y%m%d-%H%M%S')}.nii.gz"
        out_path = out_dir / fname
        sitk.WriteImage(img, str(out_path))

        st.sr_image = img
        st.sr_spacing = img.GetSpacing()
        st.pipeline_state["sr_done"] = True

        # downstream uses SR
        st.image_path = str(out_path)
        st.spacing = st.sr_spacing

        message = (
            "Super resolution completed.\n"
            f"Saved as `{fname}`.\n"
            f"New shape: {new_shape} (x, y, z order)"
        )
        return output, out_path, new_shape, message



    def _format_duration(self, seconds: float) -> str:
        if seconds < 1e-3:
            return f"{seconds*1e6:.0f} µs"
        if seconds < 1:
            return f"{seconds*1e3:.0f} ms"
        if seconds < 60:
            return f"{seconds:.2f} s"
        m, s = divmod(seconds, 60)
        return f"{int(m)} min {s:.1f} s"


    def check_tools(self, response, sid: str | None = None):
        sid = sid or self.active_sid
        st = self.S(sid)

        fig = None
        tool_call = self.safe_json_loads(response)
        if not isinstance(tool_call, dict) or "tools" not in tool_call:
            return response, fig

        tool_type = tool_call.get("tools")
        class_ids = tool_call.get("params", [])

        if tool_type == "super_resolution":
            if st.image_path is None:
                return f"[Image {sid}] Please upload a NIfTI image first.", None
            _, _, _, msg = self.super_resolve(sid)
            return f"[Image {sid}] {msg}", None

        if tool_type == "seg":
            if st.image_path is None:
                return f"[Image {sid}] Please upload a NIfTI image first.", None
            if not class_ids:
                return "No class IDs provided for segmentation.", None
            msg = self.ensure_segmentation(sid)
            fig = self.create_3d_mesh(class_ids, sid=sid)
            return f"{msg}\nSegmented regions: {class_ids}", fig

        if tool_type == "diameter":
            if st.image_path is None:
                return f"[Image {sid}] Please upload a NIfTI image first.", None
            if not class_ids:
                return "Sorry, I cannot calculate the diameter for this region.", None
            msg = self.ensure_diameters(sid)
            fig = self.create_3d_mesh(class_ids, sid=sid)
            table = self.render_diameter_table_html(class_ids, sid=sid)
            return f"{msg}\n{table}", fig

        return response, None


def create_interface():
    agent = AortaAgent()
    graph = build_aorta_graph_step3(agent)

    initial_prompt = "Hello. I am AortaGPT. You can chat with me directly, or upload a NIfTI file to begin image analysis."
    # css = """
    # #chat-column {
    #     display: flex;
    #     flex-direction: column;
    #     height: 100%;
    # }
    # #chat-box {
    #     flex: 1;               /* expands to fill */
    #     overflow-y: auto;      /* scroll if content too tall */
    # }
    # """

    title = 'AortaGPT'
    # title = 'AortaGPT👑📜'
    # icons = ["marcus_icon_black_256.png", 
    #          "seneca_icon_black_256.png", 
    #          "epictetus_icon_black_256.png"]
    # icon = random.choice(icons)
    # m = data_url("assets/" + icon)
    # colors = ["#FFD700", "#CD7F32", "#4169E1", '#800080']  # gold, bronze, royal blue
    # chosen_color = random.choice(colors)

    with gr.Blocks(theme=gr.themes.Soft(), 
                #    css=css, 
                   title=title) as interface:
    # with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="AortaGPT") as interface:
        gr.HTML(
            """
        <style>
        body, .gradio-container {
            font-family: Arial, sans-serif !important;
        }
        </style>
            """
        )
        # gr.Markdown(
        #     f"<h1 style='display:inline-flex; align-items:center; gap:0.4em;'>"
        #     f"{title} "
        #     f"<img src='{m}' style='width:1.2em;height:1.2em;'/>"
        #     f"</h1>"
        # )
        gr.Markdown(
            f"<h1 style='display:inline-flex; align-items:center; gap:0.1em;'>"
            f"{title} "
            # f"<span style='display:inline-block; width:1.2em; height:1.2em; "
            # f"background-color:{chosen_color}; "
            # # f"background-color:#FFD700; "  # Gold
            # f"-webkit-mask:url({m}) center/contain no-repeat; "
            # f"mask:url({m}) center/contain no-repeat;'/>"
            f"</h1>"
        )

        gr.Markdown(
            "Upload a 3D NIfTI image, or have a multi-turn conversation directly with the AI assistant."
        )

        # ---------- Persistent states ----------
        seg_state = gr.State(None)         # 3D mesh fig (shared)
        pending_clar_state = gr.State(None)

        # Viewer A state
        ct_vol_state_A    = gr.State(None)
        # plane_state_A     = gr.State("sagittal")
        # slice_idx_state_A = gr.State(None)
        spacing_state_A   = gr.State((1.0, 1.0, 1.0))

        # Viewer B state
        ct_vol_state_B    = gr.State(None)
        # plane_state_B     = gr.State("sagittal")
        # slice_idx_state_B = gr.State(None)
        spacing_state_B   = gr.State((1.0, 1.0, 1.0))

        plane_state_shared = gr.State("sagittal")
        slice_idx_state_shared = gr.State(None)

        with gr.Row():
            # ---------------- Viewer A ----------------
            with gr.Column(scale=2):
                gr.Markdown("### Image A")

                status_html_A = gr.HTML()

                file_input_A = gr.File(label="Upload NIfTI Image (A)")

                with gr.Row():
                    sr_btn_A  = gr.Button("🌌 Super Resolution", variant="secondary")
                    seg_btn_A = gr.Button("🧩 Segmentation", variant="secondary")
                    dia_btn_A = gr.Button("📏 Diameters", variant="secondary")
                    abn_btn_A = gr.Button("Abnormality")
                    report_btn_A = gr.Button("PDF Report")
                    report_file_A = gr.File(label="Report (CT-A)", interactive=False, visible=False)

                    export_mask_btn_A = gr.Button("Export Seg Mask")
                    mask_file_A = gr.File(label="Mask", interactive=False, visible=False)


                image_source_A = gr.Radio(
                    ["Original", "Super-resolved"],
                    value="Original",
                    label="CT image source (A)"
                )

                ct_view_A = gr.Image(label="CT View (A)", interactive=False)
                # slice_slider_A = gr.Slider(minimum=0, maximum=1, value=0, step=1, label="Slice (A)", interactive=True)
                overlay_legend_A = gr.HTML(value="")

                # overlay_toggle_A = gr.Checkbox(label="Show segmentation overlay (A)", value=True)
                # overlay_alpha_A  = gr.Slider(label="Overlay opacity (A)", minimum=0.1, maximum=0.8, value=0.4, step=0.05)

                slice_png_file_A = gr.File(label="Slice PNG (A)", visible=False)

                # with gr.Row():
                #     axial_btn_A    = gr.Button("⬇️ Axial (A)", variant="secondary")
                #     sagittal_btn_A = gr.Button("↔️ Sagittal (A)", variant="secondary")
                #     coronal_btn_A  = gr.Button("↕️ Coronal (A)", variant="secondary")
                download_slice_btn_A = gr.Button("⬇️ Download slice (A) (PNG)", variant="secondary")

                seg_plot_A = gr.Plot(label="3D Segmentation (A)")

            # ---------------- Viewer B ----------------
            with gr.Column(scale=2):
                gr.Markdown("### Image B")

                status_html_B = gr.HTML()

                file_input_B = gr.File(label="Upload NIfTI Image (B)")

                with gr.Row():
                    sr_btn_B  = gr.Button("🌌 Super Resolution", variant="secondary")
                    seg_btn_B = gr.Button("🧩 Segmentation", variant="secondary")
                    dia_btn_B = gr.Button("📏 Diameters", variant="secondary")      
                    abn_btn_B = gr.Button("Abnormality")
                    report_btn_B = gr.Button("PDF Report")
                    report_file_B = gr.File(label="Report", interactive=False, visible=False)

                    export_mask_btn_B = gr.Button("Export Seg Mask")
                    mask_file_B = gr.File(label="Mask", interactive=False, visible=False)


                image_source_B = gr.Radio(
                    ["Original", "Super-resolved"],
                    value="Original",
                    label="CT image source (B)"
                )

                ct_view_B = gr.Image(label="CT View (B)", interactive=False)
                # slice_slider_B = gr.Slider(minimum=0, maximum=1, value=0, step=1, label="Slice (B)", interactive=True)
                overlay_legend_B = gr.HTML(value="")

                # overlay_toggle_B = gr.Checkbox(label="Show segmentation overlay (B)", value=True)
                # overlay_alpha_B  = gr.Slider(label="Overlay opacity (B)", minimum=0.1, maximum=0.8, value=0.4, step=0.05)

                slice_png_file_B = gr.File(label="Slice PNG (B)", visible=False)

                # with gr.Row():
                #     axial_btn_B    = gr.Button("⬇️ Axial (B)", variant="secondary")
                #     sagittal_btn_B = gr.Button("↔️ Sagittal (B)", variant="secondary")
                #     coronal_btn_B  = gr.Button("↕️ Coronal (B)", variant="secondary")
                download_slice_btn_B = gr.Button("⬇️ Download slice (B) (PNG)", variant="secondary")
 
                seg_plot_B = gr.Plot(label="3D Segmentation (B)")


            # ---------------- Chat + Plot ----------------
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=600,
                    value=[(None, initial_prompt)],
                    bubble_full_width=False,
                )

            with gr.Column():
                gr.Markdown("### Shared comparison controls")

                slice_slider_shared = gr.Slider(
                    minimum=0, maximum=1, value=0, step=1,
                    label="Shared Slice", interactive=True
                )

                overlay_toggle_shared = gr.Checkbox(
                    label="Show segmentation overlay (A+B)", value=True
                )

                overlay_alpha_shared = gr.Slider(
                    label="Overlay opacity (A+B)",
                    minimum=0.1, maximum=0.8, value=0.4, step=0.05
                )

                with gr.Row():
                    axial_btn_shared = gr.Button("⬇️ Axial", variant="secondary")
                    sagittal_btn_shared = gr.Button("↔️ Sagittal", variant="secondary")
                    coronal_btn_shared = gr.Button("↕️ Coronal", variant="secondary")


                with gr.Row():
                    msg_input = gr.Textbox(show_label=False, placeholder="Enter your command here...", scale=8)
                    send_button = gr.Button("Send", scale=1)

                guidelines_only = gr.Checkbox(
                    label="Guidelines only (SVS/ESVS/AHA/ACC/PubMed/journals)",
                    value=True,
                )
                clear_button = gr.Button("Clear Conversation")


        # ---------- Callbacks (keep viz + history persistent) ----------

        def handle_upload_slot(sid, file, chat_history):
            if file is None:
                return (
                    chat_history,
                    gr.update(),  # status
                    gr.update(),  # ct_view
                    gr.update(),  # slice_slider
                    None,         # ct_vol_state
                    "sagittal",   # plane_state
                    None,         # slice_idx_state
                    (1.0, 1.0, 1.0),  # spacing_state
                    gr.update(value="Original"),     # image_source
                    gr.update(value=""),             # overlay_legend
                    gr.update(visible=True),         # suggest_box
                )

            load_status = agent.load_nifti(file.name, sid=sid)
            chat_history = (chat_history or []) + [(None, load_status)]

            st = agent.S(sid)
            vol_zyx = sitk.GetArrayFromImage(st.sitk_image)

            plane = "sagittal"
            n = _axis_len(vol_zyx, plane)
            idx = n // 2
            spacing = st.spacing

            view_img = _render_slice(vol_zyx, plane, idx, spacing=spacing)

            return (
                chat_history,
                gr.update(value=_render_status_html_slot(sid)),
                gr.update(value=view_img),
                gr.update(minimum=0, maximum=n-1, value=idx, step=1),
                vol_zyx,
                plane,
                idx,
                spacing,
                gr.update(value="Original"),
                gr.update(value=""),
                gr.update(visible=True),
            )


        def clear_chat():
            agent.studies["A"] = StudyState()
            agent.studies["B"] = StudyState()
            agent.active_sid = "A"

            return (
                "",
                [(None, initial_prompt)],
                # gr.update(value=None),  # plot_output
                None,                   # seg_state
                gr.update(value=_render_status_html_slot("A")),
                gr.update(value=_render_status_html_slot("B")),
            )



        def add_user_message(user_message, chat_history, pending_clar):
            chat_history = chat_history or []
            msg = user_message or ""

            # If we are waiting for clarification, merge it into a clarified command
            if pending_clar and isinstance(pending_clar, dict):
                orig = pending_clar.get("original_user_text", "")
                msg = f"{orig}\nClarification: {msg}"
                pending_clar = None  # consumed

            chat_history.append((msg, None))
            return "", chat_history, pending_clar


        # Step 2 – generate assistant response (preserve segmentation fig if not updated)
        def generate_response(chat_history, figA, figB, pending_clar):
            # Always return: chatbot, seg_plot_A, seg_plot_B, statusA, statusB, pending_clar

            chat_history = chat_history or []

            if not chat_history:
                return (
                    chat_history,
                    figA,
                    figB,
                    gr.update(value=_render_status_html_slot("A")),
                    gr.update(value=_render_status_html_slot("B")),
                    pending_clar,
                )

            user_msg, assistant_msg = chat_history[-1]

            # --- 1) NOOP sentinel ---
            if assistant_msg == "__NOOP__":
                chat_history = chat_history[:-1]
                return (
                    chat_history,
                    figA,
                    figB,
                    gr.update(value=_render_status_html_slot("A")),
                    gr.update(value=_render_status_html_slot("B")),
                    pending_clar,
                )

            # --- 2) Silent abnormality analysis for PDF export ---
            if user_msg == "__REPORT_ABN__":
                sid = agent.active_sid
                st_slot = agent.S(sid)
                structured_text = agent.run_structured_abnormality_analysis(sid=sid)
                st_slot.last_impression = agent._clean_impression_text(structured_text)
                st_slot.pipeline_state["abn_done"] = True

                chat_history = chat_history[:-1]  # remove sentinel
                return (
                    chat_history,
                    figA,
                    figB,
                    gr.update(value=_render_status_html_slot("A")),
                    gr.update(value=_render_status_html_slot("B")),
                    pending_clar,
                )

            message = user_msg or ""

            # Pick which CT slot the user is referring to
            sid = _infer_sid_from_text(message)
            agent.set_active(sid)

            # Run langgraph
            out_state = graph.invoke({"user_text": message})

            # Clarification gating
            if out_state.get("needs_clarification"):
                pending_clar = out_state.get("pending_clarification")
                chat_history[-1] = (message, out_state.get("assistant_text", ""))
                return (
                    chat_history,
                    figA,
                    figB,
                    gr.update(value=_render_status_html_slot("A")),
                    gr.update(value=_render_status_html_slot("B")),
                    pending_clar,
                )

            assistant_text = out_state.get("assistant_text", "__IMAGING__")
            fig_out = out_state.get("fig", None)

            # If graph answered (web or imaging node produced text)
            if assistant_text != "__IMAGING__":
                chat_history[-1] = (message, assistant_text)

                # If a mesh figure was produced, route it to A or B plot
                if fig_out is not None:
                    if sid == "A":
                        figA = fig_out
                    else:
                        figB = fig_out

                return (
                    chat_history,
                    figA,
                    figB,
                    gr.update(value=_render_status_html_slot("A")),
                    gr.update(value=_render_status_html_slot("B")),
                    pending_clar,
                )

            # --- Special abnormality analysis (typed in chat) ---
            if message and "abnormality" in message.lower():
                structured_text = agent.run_structured_abnormality_analysis(sid=agent.active_sid)
                st_slot = agent.S(agent.active_sid)
                st_slot.last_impression = agent._clean_impression_text(structured_text)
                st_slot.pipeline_state["abn_done"] = True

                chat_history[-1] = (message, structured_text)
                return (
                    chat_history,
                    figA,
                    figB,
                    gr.update(value=_render_status_html_slot("A")),
                    gr.update(value=_render_status_html_slot("B")),
                    pending_clar,
                )

            # --- Fallback to legacy LLM tool calling ---
            response = agent.process_command(message, chat_history[:-1])
            bot_message, new_fig = agent.check_tools(response)
            chat_history[-1] = (message, bot_message)

            if new_fig is not None:
                if sid == "A":
                    figA = new_fig
                else:
                    figB = new_fig

            return (
                chat_history,
                figA,
                figB,
                gr.update(value=_render_status_html_slot("A")),
                gr.update(value=_render_status_html_slot("B")),
                pending_clar,
            )


        def set_active_A(chatbot):
            agent.active_sid = "A"
            return "", chatbot

        def set_active_B(chatbot):
            agent.active_sid = "B"
            return "", chatbot


        def update_both_views(idx, plane, overlay_on, alpha, source_A, source_B):
            # Pull current volumes
            vol_A, spacing_A = _get_vol_and_spacing_from_source("A", source_A)
            vol_B, spacing_B = _get_vol_and_spacing_from_source("B", source_B)

            img_A, img_B = None, None
            legend_A, legend_B = "", ""

            # ----- A -----
            if vol_A is not None:
                nA = _axis_len(vol_A, plane)
                idxA = min(max(int(idx), 0), nA - 1)

                stA = agent.S("A")
                overlay_mask_A = None
                if overlay_on:
                    if source_A == "Super-resolved" and stA.segmented_mask is not None and stA.segmented_mask.shape == vol_A.shape:
                        overlay_mask_A = stA.segmented_mask
                    elif source_A == "Original" and stA.segmented_mask_orig is not None and stA.segmented_mask_orig.shape == vol_A.shape:
                        overlay_mask_A = stA.segmented_mask_orig

                img_A, zone_ids_A = _render_slice(
                    vol_A, plane, idxA,
                    spacing=spacing_A,
                    mask_zyx=overlay_mask_A,
                    zone_colors=agent.ZONE_COLORS,
                    alpha=alpha,
                    return_zone_ids=True,
                )
                legend_A = _build_slice_legend_html(zone_ids_A)

            # ----- B -----
            if vol_B is not None:
                nB = _axis_len(vol_B, plane)
                idxB = min(max(int(idx), 0), nB - 1)

                stB = agent.S("B")
                overlay_mask_B = None
                if overlay_on:
                    if source_B == "Super-resolved" and stB.segmented_mask is not None and stB.segmented_mask.shape == vol_B.shape:
                        overlay_mask_B = stB.segmented_mask
                    elif source_B == "Original" and stB.segmented_mask_orig is not None and stB.segmented_mask_orig.shape == vol_B.shape:
                        overlay_mask_B = stB.segmented_mask_orig

                img_B, zone_ids_B = _render_slice(
                    vol_B, plane, idxB,
                    spacing=spacing_B,
                    mask_zyx=overlay_mask_B,
                    zone_colors=agent.ZONE_COLORS,
                    alpha=alpha,
                    return_zone_ids=True,
                )
                legend_B = _build_slice_legend_html(zone_ids_B)

            # shared slider max should match the smaller valid range if both exist
            valid_lengths = []
            if vol_A is not None:
                valid_lengths.append(_axis_len(vol_A, plane))
            if vol_B is not None:
                valid_lengths.append(_axis_len(vol_B, plane))

            if valid_lengths:
                n_shared = min(valid_lengths)
                idx = min(max(int(idx), 0), n_shared - 1)
                slider_update = gr.update(minimum=0, maximum=n_shared - 1, value=idx, step=1)
            else:
                slider_update = gr.update()

            return (
                gr.update(value=img_A),
                gr.update(value=img_B),
                gr.update(value=legend_A),
                gr.update(value=legend_B),
                slider_update,
            )


        def set_plane_shared(new_plane, overlay_on, alpha, source_A, source_B):
            # choose default idx from min valid length
            vol_A, _ = _get_vol_and_spacing_from_source("A", source_A)
            vol_B, _ = _get_vol_and_spacing_from_source("B", source_B)

            valid_lengths = []
            if vol_A is not None:
                valid_lengths.append(_axis_len(vol_A, new_plane))
            if vol_B is not None:
                valid_lengths.append(_axis_len(vol_B, new_plane))

            if not valid_lengths:
                return (
                    gr.update(), gr.update(),
                    gr.update(value=""), gr.update(value=""),
                    gr.update(), new_plane, None
                )

            n_shared = min(valid_lengths)
            idx = n_shared // 2

            imgA_upd, imgB_upd, legA_upd, legB_upd, slider_upd = update_both_views(
                idx, new_plane, overlay_on, alpha, source_A, source_B
            )

            return (
                imgA_upd,
                imgB_upd,
                legA_upd,
                legB_upd,
                slider_upd,
                new_plane,
                idx,
            )

        def run_sr_slot(sid: str, source, plane, idx, overlay_on, alpha, chat_history):
            agent.set_active(sid)
            msg = agent.ensure_super_resolved(sid)

            # After SR, force the viewer source to Super-resolved and re-render
            st = agent.S(sid)
            vol_zyx, spacing = _get_vol_and_spacing_from_source(sid, "Super-resolved")

            if vol_zyx is None:
                chat_history = (chat_history or []) + [(None, msg)]
                return (
                    chat_history,
                    gr.update(value=_render_status_html_slot(sid)),
                    gr.update(value=img),
                    gr.update(minimum=0, maximum=n-1, value=int(idx), step=1),
                    gr.update(value="Super-resolved"),
                    gr.update(value=legend_html),
                )

            n = _axis_len(vol_zyx, plane)
            if idx is None or not (0 <= int(idx) < n):
                idx = n // 2

            overlay_mask = None
            if overlay_on and st.segmented_mask is not None and st.segmented_mask.shape == vol_zyx.shape:
                overlay_mask = st.segmented_mask

            img, zone_ids = _render_slice(
                vol_zyx, plane, int(idx),
                spacing=spacing,
                mask_zyx=overlay_mask,
                zone_colors=agent.ZONE_COLORS,
                alpha=alpha,
                return_zone_ids=True,
            )
            legend_html = _build_slice_legend_html(zone_ids)

            chat_history = (chat_history or []) + [(None, msg)]
            return (
                chat_history,
                gr.update(value=_render_status_html_slot(sid)),
                gr.update(value=img),
                gr.update(minimum=0, maximum=n-1, value=int(idx), step=1),
                gr.update(value="Super-resolved"),
                gr.update(value=legend_html),
            )


        def run_seg_slot(sid: str, chat_history):
            agent.set_active(sid)
            msg = agent.ensure_segmentation(sid)
            fig = agent.create_3d_mesh(agent.classes, sid=sid)
            chat_history = (chat_history or []) + [(None, msg)]
            return chat_history, fig, gr.update(value=_render_status_html_slot(sid))



        def run_dia_slot(sid: str, chat_history):
            agent.set_active(sid)
            msg = agent.ensure_diameters(sid)
            table = agent.render_diameter_table_html(agent.classes, sid=sid)
            chat_history = (chat_history or []) + [(None, f"{msg}\n{table}")]
            return chat_history, gr.update(value=_render_status_html_slot(sid))


        def _build_slice_legend_html(zone_ids):
            """
            Build a compact legend HTML for the current slice, using the
            segmentation classes that actually appear in this slice.
            """
            if not zone_ids:
                return ""
            items = []
            seen = set()
            for cid in zone_ids:
                cid = int(cid)
                if cid in seen:
                    continue
                seen.add(cid)
                name = agent.class_mapping.get(str(cid), f"Class {cid}")
                color = agent.ZONE_COLORS.get(cid, "#bbbbbb")
                items.append(
                    "<div style='display:inline-flex;align-items:center;"
                    "gap:6px;padding:2px 6px;margin:2px 4px;border-radius:6px;"
                    "background:rgba(255,255,255,0.03);border:1px solid #333;'>"
                    f"<span style='width:10px;height:10px;border-radius:3px;"
                    f"background:{color};display:inline-block;'></span>"
                    f"<span style='font-size:11px;color:#e6e6e6;'>{name}</span>"
                    "</div>"
                )
            return (
                "<div style='margin-top:4px;font-size:11px;'>"
                "<span style='opacity:0.7;margin-right:4px;'>Visible zones:</span>"
                + "".join(items)
                + "</div>"
            )


        def _enqueue_seg_if_missing(chatbot):
            sid = agent.active_sid  # "A" or "B"
            # Adjust these fields to match your seg_state / caching structure
            has_mask = False
            try:
                has_mask = bool(agent.studies[sid].seg_mask_path)  # example
            except Exception:
                pass

            if not has_mask:
                return trigger_segmentation(chatbot)  # should return ("", chatbot)
            return "", chatbot


        # ---------- Wiring ----------
        def _get_vol_and_spacing_from_source(sid: str, source: str):
            st = agent.S(sid)
            if source == "Super-resolved" and st.sr_image is not None:
                return sitk.GetArrayFromImage(st.sr_image), st.sr_spacing
            if st.orig_image is not None:
                return sitk.GetArrayFromImage(st.orig_image), st.orig_spacing
            return None, None


        def trigger_segmentation(chat_history):
            # acts like user typed this
            chat_history = (chat_history or []) + [("Give me the segmentation for all zones.", None)]
            return "", chat_history

        def trigger_diameters(chat_history):
            chat_history = (chat_history or []) + [("Give me the diameter calculation for all zones.", None)]
            return "", chat_history
        
        def trigger_super(chat_history):
            chat_history = (chat_history or []) + [("Give me the super resolution", None)]
            return "", chat_history
        
        def trigger_abnormality(chat_history):
            """
            Pushes a user message asking for abnormality analysis based on diameters.
            """
            chat_history = (chat_history or []) + [("Give me an abnormality analysis based on the diameters.", None)]
            return "", chat_history


        def _window_to_uint8(arr, wl=40.0, ww=400.0):
            """Apply CT windowing and convert to uint8 0-255."""
            lo = wl - ww / 2.0
            hi = wl + ww / 2.0
            arr = np.clip(arr, lo, hi)
            arr = (arr - lo) / (hi - lo + 1e-6)  # [0,1]
            return (arr * 255.0).astype(np.uint8)

        def _render_slice(
            vol_zyx,
            plane,
            idx,
            spacing=(1.0, 1.0, 1.0),
            wl=40.0,
            ww=400.0,
            mask_zyx=None,
            zone_colors=None,
            alpha=0.4,
            return_zone_ids=False,
        ):
            """
            Render a CT slice (with optional segmentation overlay).

            vol_zyx: CT volume in (Z, Y, X)
            mask_zyx: segmentation in (Z, Y, X) with integer class IDs (same shape as vol_zyx)
            plane: 'axial', 'sagittal', or 'coronal'

            If return_zone_ids=True, returns (image, zone_ids) where zone_ids are the
            non-zero class IDs present in this slice. Otherwise returns just image.
            """
            if vol_zyx is None:
                return (None, []) if return_zone_ids else None

            sx, sy, sz = spacing  # SimpleITK order: (x, y, z)

            # --- 1) Pick CT slice + matching mask slice (if provided) ---
            has_mask = mask_zyx is not None and mask_zyx.shape == vol_zyx.shape

            if plane == "axial":
                img2d = vol_zyx[int(idx), :, :]    # (rows=Y, cols=X)
                row_sp, col_sp = sy, sx
                k = 0
                mask2d = mask_zyx[int(idx), :, :] if has_mask else None
            elif plane == "sagittal":
                img2d = vol_zyx[:, :, int(idx)]    # (rows=Z, cols=Y)
                row_sp, col_sp = sz, sy
                k = 3
                mask2d = mask_zyx[:, :, int(idx)] if has_mask else None
            elif plane == "coronal":
                img2d = vol_zyx[:, int(idx), :]    # (rows=Z, cols=X)
                row_sp, col_sp = sz, sx
                k = 3
                mask2d = mask_zyx[:, int(idx), :] if has_mask else None
            else:
                raise ValueError("Unknown plane")

            # --- 2) Rotate to viewing orientation (CT + mask) ---
            if k % 4:
                img2d = np.rot90(img2d, k=k)
                if mask2d is not None:
                    mask2d = np.rot90(mask2d, k=k)
                if k % 2 == 1:  # 90° or 270° => axes swapped
                    row_sp, col_sp = col_sp, row_sp

            # --- 3) Upsample to make pixels ~square in mm (CT + mask) ---
            min_sp = min(row_sp, col_sp)
            scale_r = max(1, int(round(row_sp / min_sp)))
            scale_c = max(1, int(round(col_sp / min_sp)))

            img2d = _nearest_resize(img2d, scale_r, scale_c)
            if mask2d is not None:
                mask2d = _nearest_resize(mask2d, scale_r, scale_c)

            # --- 4) Final orientation tweaks (same as before, but applied to mask too) ---
            if plane != "axial":
                img2d = np.rot90(img2d, k=-1)
                if mask2d is not None:
                    mask2d = np.rot90(mask2d, k=-1)
            if plane == "sagittal":
                img2d = np.fliplr(img2d)
                if mask2d is not None:
                    mask2d = np.fliplr(mask2d)

            # --- 5) Window CT to uint8 ---
            base = _window_to_uint8(img2d, wl, ww)

            # If no valid mask, just return the grayscale CT
            if mask2d is None:
                if return_zone_ids:
                    return base, []
                return base

            # --- 6) Build RGB overlay & collect zone ids ---
            rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
            if zone_colors is None:
                zone_colors = {}

            mask2d = mask2d.astype(np.int32)
            unique_ids = np.unique(mask2d)
            zone_ids = [int(i) for i in unique_ids if i != 0]

            for cid in unique_ids:
                if cid == 0:
                    continue
                hex_color = zone_colors.get(int(cid))
                if not hex_color or not isinstance(hex_color, str) or not hex_color.startswith("#"):
                    continue

                try:
                    r = int(hex_color[1:3], 16)
                    g = int(hex_color[3:5], 16)
                    b = int(hex_color[5:7], 16)
                except ValueError:
                    continue

                mask_bool = (mask2d == cid)
                if not np.any(mask_bool):
                    continue

                overlay = np.array([r, g, b], dtype=np.float32)
                rgb[mask_bool] = (1.0 - alpha) * rgb[mask_bool] + alpha * overlay

            img_out = rgb.astype(np.uint8)
            if return_zone_ids:
                return img_out, zone_ids
            return img_out




        def _axis_len(vol_zyx, plane):
            z, y, x = vol_zyx.shape
            return {"axial": z, "coronal": y, "sagittal": x}[plane]
        
        def _nearest_resize(img2d, scale_h=1, scale_w=1):
            if scale_h > 1:
                img2d = np.repeat(img2d, int(scale_h), axis=0)
            if scale_w > 1:
                img2d = np.repeat(img2d, int(scale_w), axis=1)
            return img2d

        def on_source_change_slot(sid, source, plane, idx, chat_history, overlay_on, alpha):
            st = agent.S(sid)

            # If SR requested but missing, run SR for this slot
            if source == "Super-resolved" and st.sr_image is None:
                if st.image_path is None:
                    chat_history = (chat_history or []) + [(None, f"[Image {sid}] Upload a NIfTI first.")]
                    source = "Original"
                else:
                    _, _, _, sr_message = agent.super_resolve(sid)
                    chat_history = (chat_history or []) + [(None, f"[Image {sid}] SR ran automatically.\n{sr_message}")]

            vol_zyx, spacing = _get_vol_and_spacing_from_source(sid, source)
            if vol_zyx is None:
                return gr.update(), gr.update(), None, (1.0,1.0,1.0), chat_history, gr.update(value=""), gr.update(value=_render_status_html_slot(sid))

            n = _axis_len(vol_zyx, plane)
            if idx is None or not (0 <= int(idx) < n):
                idx = n // 2

            # pick correct overlay mask for this slot+source
            overlay_mask = None
            if overlay_on:
                if source == "Super-resolved" and st.segmented_mask is not None and st.segmented_mask.shape == vol_zyx.shape:
                    overlay_mask = st.segmented_mask
                elif source == "Original" and st.segmented_mask_orig is not None and st.segmented_mask_orig.shape == vol_zyx.shape:
                    overlay_mask = st.segmented_mask_orig

            img, zone_ids = _render_slice(
                vol_zyx, plane, int(idx),
                spacing=spacing,
                mask_zyx=overlay_mask,
                zone_colors=agent.ZONE_COLORS,
                alpha=alpha,
                return_zone_ids=True,
            )
            legend_html = _build_slice_legend_html(zone_ids)

            return (
                gr.update(value=img),
                gr.update(minimum=0, maximum=n-1, value=int(idx), step=1),
                vol_zyx,
                spacing,
                chat_history,
                gr.update(value=legend_html),
                gr.update(value=_render_status_html_slot(sid)),
            )



        def on_source_change(source, plane, idx, chat_history, overlay_on, alpha):
            """
            If the user selects 'Super-resolved' and we haven't computed it yet,
            run SR once, then show that image. Otherwise just switch images.
            Also updates the per-slice overlay legend.
            """
            # If SR requested but not computed yet, run it now
            if source == "Super-resolved" and agent.sr_image is None:
                if agent.image_path is None:
                    # No image yet -> stay on Original and inform the user
                    note = "Please upload a NIfTI first before choosing Super-resolved."
                    chat_history = (chat_history or []) + [(None, note)]
                    source = "Original"
                else:
                    # Run SR once, store in agent, and notify in chat
                    output, out_path, new_shape, sr_message = agent.super_resolve()
                    chat_history = (chat_history or []) + [
                        (None, f"Super-resolution ran automatically.\n{sr_message}")
                    ]

            # Pull the correct volume + spacing (SR if available and requested)
            vol_zyx, spacing = _get_vol_and_spacing_from_source(source)
            if vol_zyx is None:
                return (
                    gr.update(),           # ct_view
                    gr.update(),           # slice_slider
                    gr.update(),           # ct_vol_state
                    gr.update(),           # spacing_state
                    chat_history,          # chatbot
                    gr.update(value=""),   # overlay_legend
                )

            # Ensure slider index is valid for the chosen source
            n = _axis_len(vol_zyx, plane)
            if idx is None or not (0 <= int(idx) < n):
                idx = n // 2

            # Use overlay only if requested AND segmentation mask exists
            # Use overlay only if requested AND matching mask exists
            overlay_mask = None
            if overlay_on:
                if source == "Super-resolved" and agent.segmented_mask is not None and agent.segmented_mask.shape == vol_zyx.shape:
                    overlay_mask = agent.segmented_mask
                elif source == "Original" and agent.segmented_mask_orig is not None and agent.segmented_mask_orig.shape == vol_zyx.shape:
                    overlay_mask = agent.segmented_mask_orig


            img, zone_ids = _render_slice(
                vol_zyx,
                plane,
                int(idx),
                spacing=spacing,
                mask_zyx=overlay_mask,
                zone_colors=agent.ZONE_COLORS,
                alpha=alpha,
                return_zone_ids=True,
            )

            legend_html = _build_slice_legend_html(zone_ids)

            return (
                gr.update(value=img),                                   # ct_view
                gr.update(minimum=0, maximum=n-1, value=int(idx), step=1),  # slice_slider
                vol_zyx,                                                # ct_vol_state
                spacing,                                                # spacing_state
                chat_history,                                           # chatbot
                gr.update(value=legend_html),                           # overlay_legend
            )

        def set_plane_slot(sid, new_plane, source, overlay_on, alpha):
            vol_zyx, spacing = _get_vol_and_spacing_from_source(sid, source)
            if vol_zyx is None:
                return gr.update(), gr.update(), new_plane, None, gr.update(value="")

            n = _axis_len(vol_zyx, new_plane)
            idx = n // 2

            st = agent.S(sid)
            overlay_mask = None
            if overlay_on:
                if source == "Super-resolved" and st.segmented_mask is not None and st.segmented_mask.shape == vol_zyx.shape:
                    overlay_mask = st.segmented_mask
                elif source == "Original" and st.segmented_mask_orig is not None and st.segmented_mask_orig.shape == vol_zyx.shape:
                    overlay_mask = st.segmented_mask_orig

            img, zone_ids = _render_slice(
                vol_zyx, new_plane, idx,
                spacing=spacing,
                mask_zyx=overlay_mask,
                zone_colors=agent.ZONE_COLORS,
                alpha=alpha,
                return_zone_ids=True,
            )
            legend_html = _build_slice_legend_html(zone_ids)

            return (
                gr.update(value=img),
                gr.update(minimum=0, maximum=n-1, value=idx, step=1),
                new_plane,
                idx,
                gr.update(value=legend_html),
            )


        def _infer_sid_from_text(text: str) -> str:
            t = (text or "").lower()

            # Explicit B indicators
            if any(k in t for k in ["image b", "ct b", "slot b", "case b", "study b"]):
                return "B"

            # Explicit A indicators
            if any(k in t for k in ["image a", "ct a", "slot a", "case a", "study a"]):
                return "A"

            # Natural language: second vs first
            if re.search(r"\b(second|2nd|image\s*2|ct\s*2|volume\s*2)\b", t):
                return "B"
            if re.search(r"\b(first|1st|image\s*1|ct\s*1|volume\s*1)\b", t):
                return "A"

            # Default: keep whatever is currently active (or A)
            return getattr(agent, "active_sid", "A") or "A"


        def update_slice_slot(sid, idx, vol_zyx, plane, spacing, source, overlay_on, alpha):
            st = agent.S(sid)

            if vol_zyx is None:
                vol_zyx, spacing = _get_vol_and_spacing_from_source(sid, source)
                if vol_zyx is None:
                    return gr.update(), gr.update(value="")

            overlay_mask = None
            if overlay_on:
                if source == "Super-resolved" and st.segmented_mask is not None and st.segmented_mask.shape == vol_zyx.shape:
                    overlay_mask = st.segmented_mask
                elif source == "Original" and st.segmented_mask_orig is not None and st.segmented_mask_orig.shape == vol_zyx.shape:
                    overlay_mask = st.segmented_mask_orig

            img, zone_ids = _render_slice(
                vol_zyx, plane, int(idx),
                spacing=spacing,
                mask_zyx=overlay_mask,
                zone_colors=agent.ZONE_COLORS,
                alpha=alpha,
                return_zone_ids=True,
            )
            legend_html = _build_slice_legend_html(zone_ids)
            return img, legend_html


        def update_slice(idx, vol_zyx, plane, spacing, source, overlay_on, alpha):
            # if vol_zyx is stale (None), pull fresh from agent by source
            if vol_zyx is None:
                vol_zyx, spacing = _get_vol_and_spacing_from_source(source)
                if vol_zyx is None:
                    return gr.update(), gr.update(value="")

            # Use overlay only if requested AND segmentation mask exists and matches this volume
            overlay_mask = None
            if overlay_on:
                if source == "Super-resolved" and agent.segmented_mask is not None and agent.segmented_mask.shape == vol_zyx.shape:
                    overlay_mask = agent.segmented_mask
                elif source == "Original" and agent.segmented_mask_orig is not None and agent.segmented_mask_orig.shape == vol_zyx.shape:
                    overlay_mask = agent.segmented_mask_orig

            img, zone_ids = _render_slice(
                vol_zyx,
                plane,
                int(idx),
                spacing=spacing,
                mask_zyx=overlay_mask,
                zone_colors=agent.ZONE_COLORS,
                alpha=alpha,
                return_zone_ids=True,
            )
            legend_html = _build_slice_legend_html(zone_ids)
            return img, legend_html

        def _set_plane_from_source(new_plane, source, overlay_on, alpha):
            vol_zyx, spacing = _get_vol_and_spacing_from_source(source)
            if vol_zyx is None:
                return (
                    gr.update(),           # ct_view
                    gr.update(),           # slice_slider
                    gr.update(),           # plane_state
                    gr.update(),           # slice_idx_state
                    gr.update(value=""),   # overlay_legend
                )

            n = _axis_len(vol_zyx, new_plane)
            idx = n // 2
            overlay_mask = None
            if overlay_on:
                if source == "Super-resolved" and agent.segmented_mask is not None and agent.segmented_mask.shape == vol_zyx.shape:
                    overlay_mask = agent.segmented_mask
                elif source == "Original" and agent.segmented_mask_orig is not None and agent.segmented_mask_orig.shape == vol_zyx.shape:
                    overlay_mask = agent.segmented_mask_orig

            img, zone_ids = _render_slice(
                vol_zyx,
                new_plane,
                idx,
                spacing=spacing,
                mask_zyx=overlay_mask,
                zone_colors=agent.ZONE_COLORS,
                alpha=alpha,
                return_zone_ids=True,
            )
            legend_html = _build_slice_legend_html(zone_ids)

            return (
                gr.update(value=img),
                gr.update(minimum=0, maximum=n-1, value=idx, step=1),
                new_plane,
                idx,
                gr.update(value=legend_html),
            )

        def set_axial_src(source, overlay_on, alpha):
            return _set_plane_from_source("axial", source, overlay_on, alpha)

        def set_sagittal_src(source, overlay_on, alpha):
            return _set_plane_from_source("sagittal", source, overlay_on, alpha)

        def set_coronal_src(source, overlay_on, alpha):
            return _set_plane_from_source("coronal", source, overlay_on, alpha)

        def _enqueue_diameters(chat_history):
            """
            Ensure diameters are computed; if already done, enqueue a NOOP.
            """
            chat_history = chat_history or []

            if getattr(agent, "diameter_result_dict", None) is None:
                chat_history = chat_history + [("Give me the diameter calculation", None)]
            else:
                chat_history = chat_history + [(None, "__NOOP__")]

            return "", chat_history, gr.update(value=_render_status_html_slot("A")), gr.update(value=_render_status_html_slot("B"))

        def _enqueue_abnormality(chat_history):

            chat_history = (chat_history or []) + [
                ("Give me an abnormality analysis based on the diameters.", None)
            ]
            return "", chat_history, gr.update(value=_render_status_html_slot("A")), gr.update(value=_render_status_html_slot("B"))

        
        def _enqueue_abnormality_for_report(chat_history):

            chat_history = chat_history or []
            chat_history = chat_history + [("__REPORT_ABN__", None)]
            return "", chat_history, gr.update(value=_render_status_html_slot("A")), gr.update(value=_render_status_html_slot("B"))


        def export_report():
            sid = agent.active_sid
            st = agent.S(sid)
            path = agent.generate_pdf_report(st.last_selected_class_ids, sid=sid)
            return (
                gr.update(value=path, visible=True),
                gr.update(value=_render_status_html_slot("A")),
                gr.update(value=_render_status_html_slot("B")),
            )

        def export_segmentation_mask():
            sid = agent.active_sid  # "A" or "B"

            # Example: you have a ready-to-save NIfTI mask or a path cached
            # Replace with your real source of the mask
            mask_path = agent.studies[sid].seg_mask_path  # e.g. "/tmp/mask_A.nii.gz"

            # If you need to *write* it here, do it and return the written path.
            # Must return something gr.File can consume (path string is fine).
            return mask_path

        def export_segmentation():

            if agent.segmented_mask is None:
                # Nothing to export -> hide the file component
                return (
                gr.update(value=None, visible=False),  # seg_file
                gr.update(value=_render_status_html_slot("A")), 
                gr.update(value=_render_status_html_slot("B"))
                                                )

            mask_arr = agent.segmented_mask.astype(np.uint8)
            mask_img = sitk.GetImageFromArray(mask_arr)
            mask_size = mask_img.GetSize()  # (X, Y, Z)

            ref_img = None
            for candidate in [
                getattr(agent, "sr_image", None),
                getattr(agent, "sitk_image", None),
                getattr(agent, "orig_image", None),
            ]:
                if candidate is None:
                    continue
                try:
                    if candidate.GetSize() == mask_size:
                        ref_img = candidate
                        break
                except Exception:
                    continue

            # Copy geometry *only* if sizes match
            if ref_img is not None:
                mask_img.CopyInformation(ref_img)
            else:
                # No matching reference geometry; export with default spacing/origin
                print(
                    "[warn] No reference image with matching size for segmentation; "
                    "exporting mask with default geometry."
                )

            # Prepare output directory
            out_dir = Path(getattr(agent, "output_dir", "outputs"))
            out_dir.mkdir(parents=True, exist_ok=True)

            # Save to disk
            fname = f"segmentation_mask_{time.strftime('%Y%m%d-%H%M%S')}.nii.gz"
            out_path = out_dir / fname
            sitk.WriteImage(mask_img, str(out_path))

            return (
                gr.update(value=str(out_path), visible=True),   # seg_file
                gr.update(value=_render_status_html_slot("A")),         # status_html_slot_A
                gr.update(value=_render_status_html_slot("B"))          # status_html_slot_B
            )

        def export_current_slice_png(
            vol_zyx,
            plane,
            idx,
            spacing,
            source,
            overlay_on,
            alpha,
        ):
  
            if vol_zyx is None:
                vol_zyx, spacing = _get_vol_and_spacing_from_source(source)
                if vol_zyx is None:
                    return gr.update(value=None, visible=False)

            # Ensure we have a valid slice index
            n = _axis_len(vol_zyx, plane)
            if idx is None or not (0 <= int(idx) < n):
                idx = n // 2
            idx = int(idx)

            # Optional overlay mask (only if enabled and shapes match)
            overlay_mask = None
            if (
                overlay_on
                and agent.segmented_mask is not None
                and agent.segmented_mask.shape == vol_zyx.shape
            ):
                overlay_mask = agent.segmented_mask

            img, _ = _render_slice(
                vol_zyx,
                plane,
                idx,
                spacing=spacing,
                mask_zyx=overlay_mask,
                zone_colors=agent.ZONE_COLORS,
                alpha=alpha,
                return_zone_ids=True,
            )
            if img is None:
                return gr.update(value=None, visible=False)

            # Make sure it's uint8 RGB
            img = np.asarray(img, dtype=np.uint8)

            # Save to disk
            from PIL import Image  # local import to avoid top-level dependency issues

            out_dir = Path(getattr(agent, "output_dir", "outputs"))
            out_dir.mkdir(parents=True, exist_ok=True)

            fname = f"ct_slice_{source.lower()}_{plane}_idx{idx}_{time.strftime('%Y%m%d-%H%M%S')}.png"
            out_path = out_dir / fname

            Image.fromarray(img).save(str(out_path))

            return gr.update(value=str(out_path), visible=True)


        def _render_status_html_slot(sid: str):
            st = agent.S(sid)
            s = st.pipeline_state

            def badge(done, label):
                color = "#28a745" if done else "#6c757d"
                return (
                    f"<span style='padding:2px 8px;border-radius:999px;"
                    f"background:{color};font-size:11px;margin-right:6px;'>"
                    f"{label}</span>"
                )

            return (
                "<div style='margin-bottom:6px;'>"
                + badge(s.get("sr_done", False),    "Super-res")
                + badge(s.get("seg_done", False),   "Segmentation")
                + badge(s.get("diam_done", False),  "Diameters")
                + badge(s.get("abn_done", False),   "Analysis")
                + badge(s.get("report_done", False),"Report")
                + "</div>"
            )


        def _render_status_html():
            s = getattr(agent, "pipeline_state", {
                "sr_done": False,
                "seg_done": False,
                "diam_done": False,
            })

            def badge(done, label):
                color = "#28a745" if done else "#6c757d"  # green / gray
                return (
                    f"<span style='padding:2px 8px;border-radius:999px;"
                    f"background:{color};font-size:11px;margin-right:6px;'>"
                    f"{label}</span>"
                )

            return (
                "<div style='margin-bottom:6px;'>"
                + badge(s.get("sr_done", False),    "Super-res")
                + badge(s.get("seg_done", False),   "Segmentation")
                + badge(s.get("diam_done", False),  "Diameters")
                + badge(s.get("abn_done", False),   "Analysis")      # NEW
                + badge(s.get("report_done", False),"Report")        # NEW
                + "</div>"
            )

        def set_guidelines_only(val: bool):
            agent.guidelines_only = bool(val)
            return gr.update()



        file_input_A.upload(
            fn=lambda f, ch: handle_upload_slot("A", f, ch),
            inputs=[file_input_A, chatbot],
            outputs=[
                chatbot,
                status_html_A,
                ct_view_A,
                slice_slider_A,
                ct_vol_state_A,
                plane_state_A,
                slice_idx_state_A,
                spacing_state_A,
                image_source_A,
                overlay_legend_A,
                # suggest_box,
            ],
        )

        file_input_B.upload(
            fn=lambda f, ch: handle_upload_slot("B", f, ch),
            inputs=[file_input_B, chatbot],
            outputs=[
                chatbot,
                status_html_B,
                ct_view_B,
                slice_slider_B,
                ct_vol_state_B,
                plane_state_B,
                slice_idx_state_B,
                spacing_state_B,
                image_source_B,
                overlay_legend_B,
                # suggest_box,
            ],
        )

        image_source_A.change(
            fn=lambda source, plane, idx, ch, ov, a: on_source_change_slot("A", source, plane, idx, ch, ov, a),
            inputs=[image_source_A, plane_state_A, slice_idx_state_A, chatbot, overlay_toggle_A, overlay_alpha_A],
            outputs=[ct_view_A, slice_slider_A, ct_vol_state_A, spacing_state_A, chatbot, overlay_legend_A, status_html_A],
        )

        image_source_B.change(
            fn=lambda source, plane, idx, ch, ov, a: on_source_change_slot("B", source, plane, idx, ch, ov, a),
            inputs=[image_source_B, plane_state_B, slice_idx_state_B, chatbot, overlay_toggle_B, overlay_alpha_B],
            outputs=[ct_view_B, slice_slider_B, ct_vol_state_B, spacing_state_B, chatbot, overlay_legend_B, status_html_B],
        )


        guidelines_only.change(
            fn=set_guidelines_only,
            inputs=[guidelines_only],
            outputs=[],
        )

        download_slice_btn_A.click(
            export_current_slice_png,
            inputs=[
                ct_vol_state_A,
                plane_state_A,
                slice_idx_state_A,
                spacing_state_A,
                image_source_A,
                overlay_toggle_A,
                overlay_alpha_A,
            ],
            outputs=[slice_png_file_A],
        )
        download_slice_btn_B.click(
            export_current_slice_png,
            inputs=[
                ct_vol_state_B,
                plane_state_B,
                slice_idx_state_B,
                spacing_state_B,
                image_source_B,
                overlay_toggle_B,
                overlay_alpha_B,
            ],
            outputs=[slice_png_file_B],
        )

        slice_slider_shared.change(
            fn=update_both_views,
            inputs=[
                slice_slider_shared,
                plane_state_shared,
                overlay_toggle_shared,
                overlay_alpha_shared,
                image_source_A,
                image_source_B,
            ],
            outputs=[
                ct_view_A,
                ct_view_B,
                overlay_legend_A,
                overlay_legend_B,
                slice_slider_shared,
            ],
        )


        axial_btn_shared.click(
            fn=lambda ov, a, srcA, srcB: set_plane_shared("axial", ov, a, srcA, srcB),
            inputs=[overlay_toggle_shared, overlay_alpha_shared, image_source_A, image_source_B],
            outputs=[
                ct_view_A, ct_view_B,
                overlay_legend_A, overlay_legend_B,
                slice_slider_shared,
                plane_state_shared,
                slice_idx_state_shared,
            ],
        )

        sagittal_btn_shared.click(
            fn=lambda ov, a, srcA, srcB: set_plane_shared("sagittal", ov, a, srcA, srcB),
            inputs=[overlay_toggle_shared, overlay_alpha_shared, image_source_A, image_source_B],
            outputs=[
                ct_view_A, ct_view_B,
                overlay_legend_A, overlay_legend_B,
                slice_slider_shared,
                plane_state_shared,
                slice_idx_state_shared,
            ],
        )

        coronal_btn_shared.click(
            fn=lambda ov, a, srcA, srcB: set_plane_shared("coronal", ov, a, srcA, srcB),
            inputs=[overlay_toggle_shared, overlay_alpha_shared, image_source_A, image_source_B],
            outputs=[
                ct_view_A, ct_view_B,
                overlay_legend_A, overlay_legend_B,
                slice_slider_shared,
                plane_state_shared,
                slice_idx_state_shared,
            ],
        )


        msg_input.submit(
            add_user_message,
            inputs=[msg_input, chatbot, pending_clar_state],
            outputs=[msg_input, chatbot, pending_clar_state],
        ).then(
    generate_response,
    inputs=[chatbot, seg_plot_A, seg_plot_B, pending_clar_state],
    outputs=[chatbot, seg_plot_A, seg_plot_B, status_html_A, status_html_B, pending_clar_state],
)

        send_button.click(
            add_user_message,
            inputs=[msg_input, chatbot, pending_clar_state],
            outputs=[msg_input, chatbot, pending_clar_state],
        ).then(
    generate_response,
    inputs=[chatbot, seg_plot_A, seg_plot_B, pending_clar_state],
    outputs=[chatbot, seg_plot_A, seg_plot_B, status_html_A, status_html_B, pending_clar_state],
)


        # Clear (wipe everything)
        clear_button.click(
            clear_chat,
            inputs=None,
            outputs=[msg_input, chatbot, seg_state, status_html_A, status_html_B],
            queue=False,
        )

        # super_suggest_btn.click(
        #     trigger_super,
        #     inputs=[chatbot],
        #     outputs=[msg_input, chatbot],
        # ).then(
        #     generate_response,
        #     inputs=[chatbot, seg_state, pending_clar_state],
        #     outputs=[chatbot, seg_state, status_html_A, status_html_B, pending_clar_state],
        # )
        # seg_suggest_btn.click(
        #     trigger_segmentation,
        #     inputs=[chatbot],
        #     outputs=[msg_input, chatbot],
        # ).then(
        #     generate_response,
        #     inputs=[chatbot, seg_state, pending_clar_state],
        #     outputs=[chatbot, seg_state, status_html_A, status_html_B, pending_clar_state],
        # )

        # dia_suggest_btn.click(
        #     trigger_diameters,
        #     inputs=[chatbot],
        #     outputs=[msg_input, chatbot],
        # ).then(
        #     generate_response,
        #     inputs=[chatbot, seg_state, pending_clar_state],
        #     outputs=[chatbot, seg_state, status_html_A, status_html_B, pending_clar_state],
        # )
        abn_btn_A.click(
            set_active_A,
            inputs=[chatbot],
            outputs=[msg_input, chatbot],
        ).then(
            _enqueue_diameters,
            inputs=[chatbot],
            outputs=[msg_input, chatbot],
        ).then(
            generate_response,
            inputs=[chatbot, seg_state, pending_clar_state],
            outputs=[chatbot, seg_state, status_html_A, status_html_B, pending_clar_state],
        ).then(
            _enqueue_abnormality,
            inputs=[chatbot],
            outputs=[msg_input, chatbot, status_html_A, status_html_B],
        ).then(
            generate_response,
            inputs=[chatbot, seg_state, pending_clar_state],
            outputs=[chatbot, seg_state, status_html_A, status_html_B, pending_clar_state],
        )

        abn_btn_B.click(
            set_active_B,
            inputs=[chatbot],
            outputs=[msg_input, chatbot],
        ).then(
            _enqueue_diameters,
            inputs=[chatbot],
            outputs=[msg_input, chatbot],
        ).then(
            generate_response,
            inputs=[chatbot, seg_state, pending_clar_state],
            outputs=[chatbot, seg_state, status_html_A, status_html_B, pending_clar_state],
        ).then(
            _enqueue_abnormality,
            inputs=[chatbot],
            outputs=[msg_input, chatbot, status_html_A, status_html_B],
        ).then(
            generate_response,
            inputs=[chatbot, seg_state, pending_clar_state],
            outputs=[chatbot, seg_state, status_html_A, status_html_B, pending_clar_state],
        )

        report_btn_A.click(
            set_active_A,
            inputs=[chatbot],
            outputs=[msg_input, chatbot],
        ).then(
            _enqueue_diameters,
            inputs=[chatbot],
            outputs=[msg_input, chatbot],
        ).then(
            generate_response,
            inputs=[chatbot, seg_state, pending_clar_state],
            outputs=[chatbot, seg_state, status_html_A, status_html_B, pending_clar_state],
        ).then(
            _enqueue_abnormality_for_report,
            inputs=[chatbot],
            outputs=[msg_input, chatbot, status_html_A, status_html_B],
        ).then(
            generate_response,
            inputs=[chatbot, seg_state, pending_clar_state],
            outputs=[chatbot, seg_state, status_html_A, status_html_B, pending_clar_state],
        ).then(
            export_report,
            inputs=None,
            outputs=[report_file_A, status_html_A, status_html_B],
        )

        report_btn_B.click(
            set_active_B,
            inputs=[chatbot],
            outputs=[msg_input, chatbot],
        ).then(
            _enqueue_diameters,
            inputs=[chatbot],
            outputs=[msg_input, chatbot],
        ).then(
            generate_response,
            inputs=[chatbot, seg_state, pending_clar_state],
            outputs=[chatbot, seg_state, status_html_A, status_html_B, pending_clar_state],
        ).then(
            _enqueue_abnormality_for_report,
            inputs=[chatbot],
            outputs=[msg_input, chatbot, status_html_A, status_html_B],
        ).then(
            generate_response,
            inputs=[chatbot, seg_state, pending_clar_state],
            outputs=[chatbot, seg_state, status_html_A, status_html_B, pending_clar_state],
        ).then(
            export_report,
            inputs=None,
            outputs=[report_file_B, status_html_A, status_html_B],
        )

        # seg_export_btn.click(
        #     export_segmentation,
        #     inputs=None,
        #     outputs=[seg_file, status_html_A, status_html_B],
        # )
        sr_btn_A.click(
            fn=lambda src, plane, idx, ov, a, ch: run_sr_slot("A", src, plane, idx, ov, a, ch),
            inputs=[image_source_A, plane_state_A, slice_idx_state_A, overlay_toggle_A, overlay_alpha_A, chatbot],
            outputs=[chatbot, status_html_A, ct_view_A, slice_slider_A, image_source_A, overlay_legend_A],
        )

        seg_btn_A.click(
            fn=lambda ch: run_seg_slot("A", ch),
            inputs=[chatbot],
            outputs=[chatbot, seg_plot_A, status_html_A],
        )


        dia_btn_A.click(
            fn=lambda ch: run_dia_slot("A", ch),
            inputs=[chatbot],
            outputs=[chatbot, status_html_A],
        )


        sr_btn_B.click(
            fn=lambda src, plane, idx, ov, a, ch: run_sr_slot("B", src, plane, idx, ov, a, ch),
            inputs=[image_source_B, plane_state_B, slice_idx_state_B, overlay_toggle_B, overlay_alpha_B, chatbot],
            outputs=[chatbot, status_html_B, ct_view_B, slice_slider_B, image_source_B, overlay_legend_B],
        )

        seg_btn_B.click(
            fn=lambda ch: run_seg_slot("B", ch),
            inputs=[chatbot],
            outputs=[chatbot, seg_plot_B, status_html_B],
        )

        dia_btn_B.click(
            fn=lambda ch: run_dia_slot("B", ch),
            inputs=[chatbot],
            outputs=[chatbot, status_html_B],
        )

        export_mask_btn_A.click(
            set_active_A,
            inputs=[chatbot],
            outputs=[msg_input, chatbot],
        ).then(
            _enqueue_seg_if_missing,
            inputs=[chatbot],
            outputs=[msg_input, chatbot],
        ).then(
            generate_response,
            inputs=[chatbot, seg_state, pending_clar_state],
            outputs=[chatbot, seg_state, status_html_A, status_html_B, pending_clar_state],
        ).then(
            export_segmentation_mask,
            inputs=None,
            outputs=[mask_file_A],
        )

        export_mask_btn_B.click(
            set_active_B,
            inputs=[chatbot],
            outputs=[msg_input, chatbot],
        ).then(
            _enqueue_seg_if_missing,
            inputs=[chatbot],
            outputs=[msg_input, chatbot],
        ).then(
            generate_response,
            inputs=[chatbot, seg_state, pending_clar_state],
            outputs=[chatbot, seg_state, status_html_A, status_html_B, pending_clar_state],
        ).then(
            export_segmentation_mask,
            inputs=None,
            outputs=[mask_file_B],
        )        

    return interface

if __name__ == "__main__":
    app_interface = create_interface()
    app_interface.queue().launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
