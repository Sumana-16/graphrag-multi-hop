"""
GraphRAG — Streamlit UI (redesigned).

Design goals
  * Stable: results live in st.session_state, so any rerun (slider, tab, button)
    keeps showing the last answer instead of wiping it.
  * Fast: heavy clients (Neo4j / Chroma / embedding model) are created once per
    server process with st.cache_resource, not once per browser session.
  * Clear: one question box, answer first, then Graph / Sources / Reasoning tabs.
"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import time
from typing import Dict, List, Optional

import requests
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from config.logger import log
from config.settings import settings
from generation.answer_generator import AnswerGenerator
from reasoning.multihop import MultiHopReasoner
from utils.translate import SUPPORTED_LANGUAGES, translate_to_english, translate_to_language
from utils.voice import text_to_speech

API_URL = f"http://127.0.0.1:{settings.api_port}"

st.set_page_config(page_title="GraphRAG", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem;}
      #MainMenu, footer {visibility: hidden;}
      .hero h1 {font-size: 2.1rem; margin: 0 0 .2rem 0; letter-spacing: -.02em;}
      .hero p  {color: #8b93a7; margin: 0 0 1.2rem 0; font-size: 1.02rem;}
      .stat {background: rgba(127,127,160,.10); border: 1px solid rgba(127,127,160,.22);
             border-radius: 12px; padding: .7rem .9rem; margin-bottom: .5rem;}
      .stat .n {font-size: 1.45rem; font-weight: 700; line-height: 1.1;}
      .stat .l {font-size: .75rem; color: #8b93a7; text-transform: uppercase; letter-spacing: .06em;}
      .chip {display:inline-block; padding: .12rem .6rem; margin: 0 .3rem .3rem 0; border-radius: 999px;
             background: rgba(99,102,241,.16); border: 1px solid rgba(99,102,241,.4); font-size: .8rem;}
      .src {border: 1px solid rgba(127,127,160,.25); border-radius: 12px; padding: .8rem 1rem; margin-bottom: .6rem;}
      .src b {font-size: .95rem;} .src small {color: #8b93a7;}
      div[data-testid="stForm"] {border: 1px solid rgba(127,127,160,.25); border-radius: 14px; padding: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = [
    "What companies are mentioned in the news and how are they related?",
    "Which companies are collaborating with Salesforce?",
    "Which companies are collaborating on robotics research?",
    "What did Anthropic invest in?",
]


# ----------------------------------------------------------------------------
# Shared resources / helpers
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Starting GraphRAG engine (first load only)…")
def get_engine():
    """Create the heavy clients once per server process."""
    return MultiHopReasoner(), AnswerGenerator()


@st.cache_data(ttl=20, show_spinner=False)
def get_stats() -> Optional[Dict]:
    try:
        r = requests.get(f"{API_URL}/stats", timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def graph_html(paths: List[List[str]]) -> str:
    net = Network(height="430px", width="100%", bgcolor="#0e1117", font_color="#e6e8ee",
                  directed=True, cdn_resources="remote")
    net.set_options(
        """{"physics":{"solver":"forceAtlas2Based",
                       "forceAtlas2Based":{"gravitationalConstant":-40,"centralGravity":0.02,
                                           "springLength":90,"springConstant":0.08,"avoidOverlap":0.6},
                       "stabilization":{"enabled":true,"iterations":200,"fit":true}},
            "nodes":{"shape":"dot","size":14,"font":{"size":13,"color":"#e6e8ee"}},
            "edges":{"width":1.2,"color":{"color":"#6b7280"},"arrows":{"to":{"enabled":true,"scaleFactor":0.6}},"smooth":{"type":"continuous"}},
            "interaction":{"hover":true,"zoomView":true,"dragView":true}}"""
    )
    seen, edges = set(), set()
    for path in paths:
        path = [str(n) for n in path if n and str(n).strip()]
        for i, node in enumerate(path):
            if node not in seen:
                net.add_node(node, label=node, title=node, size=20 if i == 0 else 14,
                             color="#22c55e" if i == 0 else "#6366f1")
                seen.add(node)
            if i < len(path) - 1:
                edges.add((node, path[i + 1]))
    for a, b in edges:
        if a in seen and b in seen and a != b:
            net.add_edge(a, b)
    html = net.generate_html()
    # Remove the white frame, and auto-fit the whole graph into view once it settles.
    css = ("<style>html,body{margin:0;padding:0;background:#0e1117;overflow:hidden}"
           "#mynetwork{border:none!important;width:100%!important}"
           "center,h1{display:none}.card{border:none!important}</style>")
    # The graph is drawn while its Streamlit tab may be hidden (zero width). Re-measure and
    # re-fit whenever the visible size changes, so it is centred as soon as the tab is opened.
    fit = ("<script>(function(){var H=430,lastW=-1;"
           "function f(){try{var w=document.documentElement.clientWidth;"
           "if(w<50)return;network.setSize('100%',H+'px');network.redraw();network.fit({animation:false});}catch(e){}}"
           "var t=setInterval(function(){if(typeof network!=='undefined'){clearInterval(t);"
           "network.once('stabilizationIterationsDone',f);"
           "setInterval(function(){var w=document.documentElement.clientWidth;if(w!==lastW){lastW=w;f();setTimeout(f,300);}},250);"
           "window.addEventListener('resize',f);}},100);})();</script>")
    return html.replace("</head>", css + "</head>", 1).replace("</body>", fit + "</body>", 1)


def run_query(question: str, depth: int, top_k: int, lang: str):
    """Run retrieval + generation and store the result in session state."""
    reasoner, generator = get_engine()
    t0 = time.time()
    q_en = translate_to_english(question, lang) if lang != "en" else question
    context = reasoner.retrieve_context(q_en, graph_depth=depth, top_k_vector=top_k)
    formatted = reasoner.format_context_for_llm(context)
    result = generator.generate_query_result(q_en, context, formatted)
    answer = translate_to_language(result.answer, lang) if lang != "en" else result.answer
    st.session_state.result = {
        "question": question,
        "answer": answer,
        "lang": lang,
        "entities": list(getattr(context, "entities_detected", []) or []),
        "graph_paths": result.graph_paths or [],
        "reasoning": result.reasoning_path or [],
        "sources": result.source_articles or [],
        "insufficient": bool(getattr(result, "insufficient_context", False)),
        "seconds": round(time.time() - t0, 1),
    }
    hist = st.session_state.history
    if question in hist:
        hist.remove(question)
    hist.insert(0, question)
    del hist[8:]


def queue_question(q: str):
    """Callback: put a question in the box. The user then presses Ask to run it."""
    st.session_state.q_input = q


# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
st.session_state.setdefault("result", None)
st.session_state.setdefault("history", [])
st.session_state.setdefault("pending", None)
st.session_state.setdefault("q_input", "")
st.session_state.setdefault("error", None)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 GraphRAG")
    _name_to_code = {name: code for code, name in SUPPORTED_LANGUAGES.items()}
    lang = _name_to_code[st.selectbox("Language", list(_name_to_code.keys()), key="lang_name")]
    depth = st.slider("Graph depth (hops)", 1, 5, 3, help="How far to walk through the knowledge graph")
    top_k = st.slider("Documents to retrieve", 1, 10, 5)

    st.markdown("#### Knowledge base")
    stats = get_stats()
    if stats:
        c1, c2 = st.columns(2)
        c1.markdown(f"<div class='stat'><div class='n'>{stats['total_articles']}</div><div class='l'>Articles</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='stat'><div class='n'>{stats['graph_nodes']}</div><div class='l'>Entities</div></div>", unsafe_allow_html=True)
        c1.markdown(f"<div class='stat'><div class='n'>{stats['graph_relationships']}</div><div class='l'>Relations</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='stat'><div class='n'>{stats['vector_chunks']}</div><div class='l'>Chunks</div></div>", unsafe_allow_html=True)
    else:
        st.caption("Stats unavailable (API offline).")

    if st.session_state.history:
        st.markdown("#### Recent questions")
        st.caption("Click to reuse, then press Ask.")
        for i, h in enumerate(st.session_state.history):
            st.button(h[:48] + ("…" if len(h) > 48 else ""), key=f"hist_{i}",
                      on_click=queue_question, args=(h,), use_container_width=True)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
st.markdown(
    "<div class='hero'><h1>🧠 GraphRAG</h1>"
    "<p>Ask multi-hop questions about Tech &amp; AI news — answered from a knowledge graph plus semantic search.</p></div>",
    unsafe_allow_html=True,
)

with st.form("ask", clear_on_submit=False):
    st.text_area("Your question", key="q_input", height=90,
                 placeholder="e.g. What is the relationship between OpenAI and Nvidia?",
                 label_visibility="collapsed")
    submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

st.caption("Examples — click one to fill the box, then press Ask:")
ex_cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    ex_cols[i].button(ex, key=f"ex_{i}", on_click=queue_question, args=(ex,), use_container_width=True)

# Decide whether to run a query on this rerun
to_run = None
if submitted:
    if st.session_state.q_input.strip():
        to_run = st.session_state.q_input.strip()
    else:
        st.warning("Type a question (or click an example) first, then press Ask.")

if to_run:
    st.session_state.error = None
    with st.status("Thinking — searching the graph and documents…", expanded=False) as status:
        try:
            run_query(to_run, depth, top_k, lang)
            status.update(label="Done", state="complete")
        except Exception as e:  # keep the UI alive and show the reason
            log.error(f"Streamlit query error: {e}")
            st.session_state.error = str(e)
            status.update(label="Failed", state="error")

if st.session_state.error:
    st.error(f"Something went wrong: {st.session_state.error}")

res = st.session_state.result
if res:
    st.markdown("---")
    st.markdown(f"**Q:** {res['question']}")
    meta = [f"⏱ {res['seconds']}s", f"{len(res['sources'])} sources", f"{len(res['graph_paths'])} graph paths"]
    st.caption("  ·  ".join(meta))
    if res["entities"]:
        st.markdown("".join(f"<span class='chip'>{e}</span>" for e in res["entities"]), unsafe_allow_html=True)
    if res["insufficient"]:
        st.warning("Not enough data in the knowledge base to answer this fully. Try rephrasing.")

    tab_ans, tab_graph, tab_src, tab_reason = st.tabs(["💡 Answer", "🕸️ Graph", "📚 Sources", "🔗 Reasoning"])

    with tab_ans:
        st.markdown(res["answer"])
        if st.button("🔊 Listen", key="tts"):
            with st.spinner("Generating audio…"):
                audio = text_to_speech(res["answer"][:1500], res["lang"])
            if audio:
                st.audio(audio, format="audio/mpeg")
            else:
                st.caption("Audio unavailable for this text/language.")

    with tab_graph:
        if res["graph_paths"]:
            st.caption("Green = starting entity · drag nodes, scroll to zoom")
            components.html(graph_html(res["graph_paths"]), height=440)
            with st.expander("View paths as text"):
                for i, p in enumerate(res["graph_paths"][:15], 1):
                    st.text(f"{i}. {' → '.join(map(str, p))}")
        else:
            st.info("No graph paths were found for this question.")

    with tab_src:
        if res["sources"]:
            for i, a in enumerate(res["sources"], 1):
                title = a.get("title") or "Untitled"
                st.markdown(
                    f"<div class='src'><b>{i}. {title}</b><br><small>{a.get('source', 'Unknown')} · "
                    f"ID {a.get('article_id', '')}</small></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No sources.")

    with tab_reason:
        if res["reasoning"]:
            for i, step in enumerate(res["reasoning"], 1):
                st.markdown(f"**{i}.** {step}")
        else:
            st.caption("No reasoning steps returned.")
else:
    st.info("Ask a question above, or click an example, to get started.")

st.markdown(
    "<p style='text-align:center;color:#8b93a7;font-size:.8rem;margin-top:3rem'>"
    f"Neo4j · ChromaDB · Gemini · <a href='{API_URL}/docs' target='_blank'>API docs</a></p>",
    unsafe_allow_html=True,
)
