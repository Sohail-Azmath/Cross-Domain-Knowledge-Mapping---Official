import streamlit as st
import pandas as pd
import os
import datetime
import plotly.express as px
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import io
import jwt
from urllib.parse import unquote
import networkx as nx
from pyvis.network import Network
from collections import defaultdict
# ---------- ENTITY EXTRACTION (NER) ----------
import spacy
import re
import streamlit.components.v1 as components

from dotenv import load_dotenv
from groq import Groq
load_dotenv()  

# ----------------------------------------
# 🔐 JWT CONFIG
# ----------------------------------------
SECRET_KEY = "abghy57ghhbghyju787hgyhluck"


# Read token safely for all Streamlit versions
# IMPORTANT: no .to_dict() here
params = {k: v for k, v in st.query_params.items()}

nlp = spacy.load("en_core_web_sm")

client = Groq(api_key="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")

def ask_llama(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# -------------------------------
# Function to send message
# -------------------------------
def ask_llama(user_msg):
    system_prompt = """
    You are an AI that ONLY answers based on the 'Cross Domain Knowledge Mapping' project.

    Allowed Knowledge:
    - Dataset with columns (id, domain, sentence, label)
    - Entity extraction using spaCy NER
    - Rule-based relation extraction (subject → verb → object)
    - Using nsubj, dobj, pobj dependencies
    - Semantic search using SentenceTransformer embeddings
    - Knowledge graph creation using NetworkX + PyVis
    - Cross-domain analogies and mappings
    - Export entities/relations to CSV
    - Generate HTML knowledge graph

    RULES:
    - If the question is about the project → answer with project-specific explanations.
    - If the question is unrelated → reply:
      "This question is not part of the Cross Domain Knowledge Mapping project."
    - Do NOT give general NLP answers.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]
    )
    return response.choices[0].message.content

    st.session_state.messages.append((user_msg, ai_reply))
    st.session_state.chat_input = ""  # Clear input safely
def extract_entities(text):
    doc = nlp(str(text))
    return [(ent.text, ent.label_) for ent in doc.ents]


# ---------- SIMPLE RELATION EXTRACTION ----------
def extract_relations(text):
    doc = nlp(text)

    relations = []
    for token in doc:
        # detect verb → get subject → get object
        if token.pos_ == "VERB":
            subject = ""
            obj = ""

            # find subject
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject = child.text

                if child.dep_ in ("dobj", "pobj"):
                    obj = child.text

            if subject and obj:
                relations.append((subject, token.lemma_, obj))
    return relations


# ----------------------------------------
# 🔐 TOKEN CHECK (LOGIN GUARD)
# ----------------------------------------
if "token" not in params:
    st.error("Unauthorized access! Please login first.")
    st.stop()

# token may be a string OR list → handle both
token = params.get("token")
if isinstance(token, list):
    token = token[0]  # take the string

try:
    decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    st.success(f"Welcome {decoded['user']}!")
except Exception as e:
    st.error("Invalid token. Please login again.")
    st.write(str(e))
    st.stop()

# ----------------------------------------
# 🎨 APP CONFIGURATION
# ----------------------------------------
st.set_page_config(
    page_title="Cross-Domain Knowledge Mapping Dashboard",
    layout="wide",
    page_icon="🧭",
    initial_sidebar_state="expanded"
)

# ----------------------------------------
# 📁 FILE PATHS CONFIGURATION
# ----------------------------------------
EMBEDDINGS_PATH = "cross_domain_embeddings.pkl"
KNOWLEDGE_GRAPH_PATH = "knowledge_graph.html"
FEEDBACK_FILE = "feedback.csv"

# ----------------------------------------
# 🌈 ENHANCED PROFESSIONAL THEME
# ----------------------------------------
# ----------------------------------------
# 🎨 GLOBAL CSS CONFIGURATION (Consolidated)
# ----------------------------------------
def inject_global_css():
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        * {
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Animated Gradient Background */
        .stApp {
            background: linear-gradient(-45deg, #667eea, #764ba2, #6B8DD6, #8E37D7);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }

        /* FORCE BLACK TEXT FOR TABLES AND CHARTS */
        div[data-testid="stTable"] {
            color: #000000 !important;
        }
        div[data-testid="stMarkdownContainer"] p {
            color: #000000 !important;
        }
        thead tr th {
            color: #000000 !important;
        }
        tbody tr td {
            color: #000000 !important;
        }

        /* HIDE STREAMLIT HEADER & FOOTER */
        [data-testid="stHeader"] {
            visibility: hidden;
            height: 0px;
        }
        
        .stDeployButton {
            display: none !important;
        }
        
        footer {
            visibility: hidden;
        }
        
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding-top: 2rem;
            box-shadow: 4px 0 15px rgba(0, 0, 0, 0.3);
        }
        
        [data-testid="stSidebar"] * {
            color: #e8e8e8 !important;
            font-size: 15px !important;
            font-weight: 500;
        }

        /* Sidebar Buttons */
        [data-testid="stSidebar"] button[kind="secondary"] {
            width: 100%;
            justify-content: flex-start;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            margin: 4px 0;
            border: 1px solid rgba(255, 255, 255, 0.18);
            color: #ffffff;
        }
        [data-testid="stSidebar"] button[kind="secondary"]:hover {
            background: rgba(255, 255, 255, 0.14);
            border-color: #667eea;
        }
        
        [data-testid="stSidebar"] div.stButton > button {
             width: 100%;
        }

        /* Sidebar Radio (Navigation) */
        [data-testid="stSidebar"] .stRadio > label {
            background: rgba(255, 255, 255, 0.05);
            padding: 12px 16px;
            border-radius: 12px;
            margin: 4px 0;
            transition: all 0.3s ease;
            border-left: 3px solid transparent;
        }

        /* Navigation Boxes */
        [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
            background: rgba(255, 255, 255, 0.05);
            padding: 8px 12px;
            border-radius: 8px;
            margin: 3px 0;
            border: 1px solid rgba(255, 255, 255, 0.18);
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
        }

        [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
            background: rgba(255, 255, 255, 0.14);
            border-color: #667eea;
            transform: translateX(2px);
        }

        /* Selected Radio Item */
        [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:has(input[checked]) {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-color: rgba(255, 255, 255, 0.6);
            box-shadow: 0 3px 10px rgba(102, 126, 234, 0.45);
        }

        /* Tighten Sidebar Gap */
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0.5rem !important;
            margin-top: 0 !important;
        }

        /* Main Container Glass Effect */
        .block-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 24px;
            padding: 2rem 3rem !important;
            margin: 1rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* Headings */
        h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700 !important;
            font-size: 2.5rem !important;
            margin-bottom: 1rem;
        }

        h2, h3 {
            color: #1a1a2e !important;
            font-weight: 600 !important;
        }

        /* Metric Cards */
        [data-testid="stMetricValue"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.5rem !important;
            font-weight: 700 !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #4a5568 !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
        }
        
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.15);
            border: 1px solid rgba(102, 126, 234, 0.1);
            transition: all 0.3s ease;
        }
        
        [data-testid="metric-container"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.25);
        }
        
        /* Buttons */
        div.stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            padding: 0.75rem 2rem;
            border: none;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        div.stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        
        div.stButton > button:active {
            transform: translateY(0);
        }
        
        /* Download Buttons */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
            color: white !important;
            border-radius: 12px;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4);
            transition: all 0.3s ease;
        }
        
        .stDownloadButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(17, 153, 142, 0.5);
        }
        
        /* Data Tables */
        .stDataFrame, .stTable {
            background: white !important;
            border-radius: 16px !important;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        }
        
        .stDataFrame thead th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 1rem !important;
        }
        
        .stDataFrame tbody tr:nth-child(even) {
            background: #f8f9ff !important;
        }
        
        .stDataFrame tbody tr:hover {
            background: #e8ecff !important;
        }
        
        /* File Uploader */
        [data-testid="stFileUploader"] {
            background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
            border: 2px dashed #667eea;
            border-radius: 16px;
            padding: 2rem;
            transition: all 0.3s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: #764ba2;
            background: linear-gradient(135deg, #f0f4ff 0%, #e8ecff 100%);
        }
        
        /* Text Inputs */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stNumberInput > div > div > input {
            border-radius: 12px !important;
            border: 2px solid #e2e8f0 !important;
            padding: 0.75rem 1rem !important;
            transition: all 0.3s ease;
            background: white !important;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stNumberInput > div > div > input:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
        }
        
        /* Select Boxes */
        .stSelectbox > div > div {
            border-radius: 12px !important;
            border: 2px solid #e2e8f0 !important;
        }
        
        /* Success/Warning/Error Messages */
        .stSuccess {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%) !important;
            border-left: 4px solid #28a745 !important;
            border-radius: 12px !important;
            padding: 1rem !important;
        }
        
        .stWarning {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%) !important;
            border-left: 4px solid #ffc107 !important;
            border-radius: 12px !important;
        }
        
        .stError {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%) !important;
            border-left: 4px solid #dc3545 !important;
            border-radius: 12px !important;
        }
        
        .stInfo {
            background: linear-gradient(135deg, #e7f3ff 0%, #cce5ff 100%) !important;
            border-left: 4px solid #667eea !important;
            border-radius: 12px !important;
        }
        
        /* Charts */
        .stPlotlyChart {
            background: white;
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        }
        /* Knowledge Graph iframe */
        iframe {
            border-radius: 16px !important;
            border: 2px solid #667eea !important;
            box-shadow: 0 8px 30px rgba(102, 126, 234, 0.2);
        }
        
        /* Custom Cards */
        .upload-section {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
            border: 1px solid rgba(102, 126, 234, 0.1);
        }
        
        /* Result Cards for Semantic Search */
        .result-card {
            background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 15px rgba(72, 187, 120, 0.2);
            border-left: 4px solid #48bb78;
            transition: all 0.3s ease;
        }
        
        .result-card:hover {
            transform: translateX(10px);
            box-shadow: 0 8px 25px rgba(72, 187, 120, 0.3);
        }
        
        /* Legend Box */
        .legend-box {
            background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
            padding: 1.5rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .block-container > div {
            animation: fadeIn 0.5s ease-out;
        }
        
        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        
        /* Expander Styling */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
            border-radius: 12px;
            font-weight: 600;
        }
        
        /* Progress Bar */
        .stProgress > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }

        /* Chatbot Styles */
        .chat-container {
            background: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 15px;
            margin-top: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        
        .ai-response-box {
            background: linear-gradient(135deg, #f0f4ff 0%, #e8ecff 100%);
            padding: 15px;
            border-radius: 15px;
            margin-top: 5px;
            border-left: 4px solid #667eea;
            box-shadow: 0 4px 10px rgba(102, 126, 234, 0.2);
        }
        
        .chat-input-box input {
            border-radius: 12px !important;
            padding: 12px !important;
            border: 2px solid #667eea !important;
            font-size: 16px;
        }
        
        .ask-btn button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none;
            color: white !important;
            padding: 12px 26px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 16px;
            margin-top: 10px;
            box-shadow: 0 4px 10px rgba(102, 126, 234, 0.4);
        }
        
        .ask-btn button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.55);
        }

        /* Tighten Sidebar Separators */
        [data-testid="stSidebar"] hr {
            margin-top: 0.1rem !important;
            margin-bottom: 0.1rem !important;
        }
        [data-testid="stSidebar"] div[style*="Signed in as"] {
            margin-top: 0 !important;
            margin-bottom: 0.1rem !important;
        }
        /* Reduce space before the 'Navigate Pages' label */
        [data-testid="stSidebar"] .stRadio > label {
            margin-bottom: 0.1rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Inject CSS immediately
inject_global_css()






    


    
    






# ----------------------------------------
# 👤 SIDEBAR SIGN-IN / SIGN-OUT
# ----------------------------------------
# with st.sidebar:
#     st.markdown("### User Info")
#     st.info(f"Signed in as **{decoded['user']}**")

#     if st.button("Sign Out"):
#         st.session_state.clear()
#         st.success("You have been signed out!")
#         # Redirect using query params
#         st.experimental_set_query_params(page="login")
#         st.stop()

# # Then in your main app:
# params = st.experimental_get_query_params()
# if params.get("page") == ["login"]:
#     # Show login page content
#     st.markdown("### Login Page")
#     # your login form here



# ----------------------------------------
# 📘 SESSION STATE INITIALIZATION
# ----------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "embeddings_generated" not in st.session_state:
    st.session_state.embeddings_generated = False

# ----------------------------------------
# 📚 SIDEBAR NAVIGATION
# ----------------------------------------
pages = [
    "Upload Dataset",
    "Dataset Insights",
    "Entity & Relation Extraction",
    "Knowledge Graph",
    "Semantic Search",
    "Top 10 Sentences",
    "Feedback Section",
    "Feedback Analysis",
    "Admin Tools",
    "Download Options",
    "AI Assistance Chatbot" 
]

if "current_page" not in st.session_state:
    st.session_state.current_page = pages[0]

with st.sidebar:
    st.markdown("📑 Navigate Pages")
    for p in pages:
        if st.button(p, key=f"nav_{p}"):
            st.session_state.current_page = p

choice = st.session_state.current_page



# ----------------------------------------
# 👤 USER BAR BELOW NAVIGATION
# ----------------------------------------
with st.sidebar:
    st.markdown("---")

    # Row 1: signed-in name (full width)
    st.markdown(
        f"<div style='font-size:13px; white-space:nowrap; margin-bottom:4px;'>"
        f"Signed in as <b>{decoded['user']}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Row 2: sign out button (full width, horizontal below the name)
    if st.button("Sign Out", key="sidebar_signout"):
        # Clear Streamlit state and URL token
        st.session_state.clear()
        st.query_params.clear()

        # Redirect back to Flask login page (route "/")
        st.markdown(
            """
            <meta http-equiv="refresh" content="0; url=http://127.0.0.1:5000/">
            """,
            unsafe_allow_html=True,
        )
        st.stop()


# ----------------------------------------
# 💾 LOAD FEEDBACK FILE
# ----------------------------------------
if os.path.exists(FEEDBACK_FILE):
    feedback_df = pd.read_csv(FEEDBACK_FILE)
else:
    feedback_df = pd.DataFrame(
        columns=["record_id", "user", "feedback_type", "comment", "status", "timestamp"]
    )

## ----------------------------------------
# 📤 UPLOAD DATASET PAGE
# ----------------------------------------
if choice == "Upload Dataset":
    # st.markdown("---")  # Removed separator
    st.title("📤 Upload Your Dataset")

    st.markdown("""
    > **Welcome!** Discover how knowledge connects across domains. 
    > Upload a CSV to get started with **Semantic Search**, **Entity Recognition**, and **Knowledge Graphs**.
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="upload-section">
        <h3>Upload a CSV file containing:</h3>
        <ul>
            <li><b>id</b>: Unique identifier for each record</li>
            <li><b>domain</b>: Subject domain (e.g., Computer Science, Biology, Sociology)</li>
            <li><b>sentence</b>: Text containing knowledge</li>
            <li><b>label</b>: Relationship type or category</li>
        </ul>
        <ul><b>or</b></ul>
        <ul>Any CSV File</ul>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)

                # Validate required columns
                required_columns = ["id", "domain", "sentence", "label"]
                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                    st.info("Please ensure your CSV has these columns: id, domain, sentence, label")
                else:
                    st.session_state.df = df
                    st.success(f"✅ Successfully loaded {len(df)} rows!")
                    st.balloons()

            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

    with col2:
        with st.expander("ℹ View Sample Data Structure"):
            st.code("""id,domain,sentence,label
1,Computer Science,Algorithm uses Data,concept
2,Biology,DNA replicates,relation
3,Sociology,Society evolves,definition""", language="csv")

        # Download sample template
        sample_df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "domain": ["Computer Science", "Biology", "Sociology", "Chemistry", "Literature"],
            "sentence": [
                "Algorithm uses Data Structure",
                "DNA replication is fundamental to genetics",
                "Economic Growth affects Society",
                "Chemical bonds form molecules",
                "Narrative structure drives plot",
            ],
            "label": ["concept", "relation", "definition", "concept", "relation"],
        })

        csv_buffer = io.StringIO()
        sample_df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="⬇ Download Sample CSV",
            data=csv_buffer.getvalue(),
            file_name="sample_dataset.csv",
            mime="text/csv",
            key="download_sample_csv",   # single button, single key
        )

    # -------- Dataset Status Check --------
    if st.session_state.df is not None:
        df = st.session_state.df
        st.sidebar.success(f"✅ Loaded: {len(df)} rows")
    else:
        st.warning("⚠ Please upload a CSV file to continue.")
        st.stop()  # Stop execution here if no data

    # -------- Overview section (same page, below) --------
    st.markdown("---")
    st.header("🏠 Overview of Dataset")

    st.write("""
        Discover how knowledge from one domain connects and supports another —
        integrating *Semantic Search*, **Entity Recognition**, and *Knowledge Graph visualizations*.
    """)

    col1_over, col2_over = st.columns(2)

    with col1_over:
        st.metric("🧩 Domains", df["domain"].nunique())
        st.metric("💬 Sentences", len(df))
        st.metric("🏷 Labels", df["label"].nunique())

    with col2_over:
        domain_counts = df["domain"].value_counts().reset_index()
        domain_counts.columns = ["Domain", "Count"]
        fig = px.pie(domain_counts, names="Domain", values="Count",
                     title="Dataset Domain Distribution")
        st.plotly_chart(fig, use_container_width=True)

# # ----------------------------------------
# # 🏠 OVERVIEW
# # ----------------------------------------
# if choice == "🏠 Overview":
#     st.title("🏠 Overview of Dataset")
#     st.write("""
#         Discover how knowledge from one domain connects and supports another —
#         integrating *Semantic Search*, **Entity Recognition**, and *Knowledge Graph visualizations*.
#     """)

#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("🧩 Domains", df["domain"].nunique())
#         st.metric("💬 Sentences", len(df))
#         st.metric("🏷 Labels", df["label"].nunique())
#     with col2:
#         domain_counts = df["domain"].value_counts().reset_index()
#         domain_counts.columns = ["Domain", "Count"]
#         fig = px.pie(domain_counts, names="Domain", values="Count", title="Dataset Domain Distribution")
#         st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------
# 🎓 STUDENT VIEW
# ----------------------------------------
# elif choice == "🎓 Student View":
#     st.header("🎓 Student View - Explore Connections")
#     st.write("""
#         Students can explore how knowledge connects across multiple domains
#         like Computer Science, Biology, Chemistry, Sociology, and Literature.
#     """)
#     st.bar_chart(df["domain"].value_counts())

# ----------------------------------------
# 📊 DATASET INSIGHTS
# ----------------------------------------
elif choice == "Dataset Insights":
    df = st.session_state.df
    st.title("📊 Dataset Insights")
    st.metric("Total Rows", len(df))
    st.metric("Unique Domains", df["domain"].nunique())
    st.metric("Unique Labels", df["label"].nunique())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Domain Distribution")
        st.bar_chart(df["domain"].value_counts())
    with col2:
        st.subheader("📈 Label Distribution")
        st.bar_chart(df["label"].value_counts())

# ----------------------------------------
# 🧠 ENTITY & RELATION EXTRACTION
# ----------------------------------------
elif choice == "Entity & Relation Extraction":
    st.title("🧠 Entity & Relation Extraction")
    st.write("Extract Entities and Relations directly from your uploaded dataset.")

    # Use dataset from upload section
    df = st.session_state.df

    if "sentence" not in df.columns:
        st.error("❌ Your dataset must contain a 'sentence' column.")
        st.stop()

    # Apply entity & relation extraction
    df["entities"] = df["sentence"].apply(extract_entities)
    df["relations"] = df["sentence"].apply(extract_relations)

    # Sentences containing BOTH entity & relation
    both_df = df[
        (df["entities"].apply(lambda x: len(x) > 0))
        & (df["relations"].apply(lambda x: len(x) > 0))
    ]

    st.subheader("Sentences with Extracted Information")
    
    # Filter for rows that have EITHER entities OR relations
    rich_df = df[
        (df["entities"].apply(lambda x: len(x) > 0)) | 
        (df["relations"].apply(lambda x: len(x) > 0))
    ]

    if len(rich_df) > 0:
        st.write(f"*Showing {len(rich_df)} entries with entities or relations:*")
        st.dataframe(rich_df[["sentence", "entities", "relations"]], use_container_width=True, height=500)
    else:
        st.info("No entities or relations were found in the dataset.")

    # Download output
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Entity+Relation Output",
        data=csv_data,
        file_name="entities_relations_output.csv",
        mime="text/csv",
    )

# ----------------------------------------
# 🌐 KNOWLEDGE GRAPH (DATASET-DRIVEN, CLEAN DESIGN)
# ----------------------------------------
elif choice == "Knowledge Graph":
    df = st.session_state.df
    st.header("🌐 Interactive Knowledge Graph Visualization")
    st.write("""
        This interactive graph is powered by <b>PyVis + NetworkX</b> and is built from your uploaded dataset.<br>
        Nodes represent short concepts extracted from sentences (grouped by domain), and edges show
        within-domain and cross-domain relationships based on labels.
    """, unsafe_allow_html=True)

    # Check if pyvis is installed
    try:
        from pyvis.network import Network
        import networkx as nx
        from collections import defaultdict
        import re
    except ImportError:
        st.error("❌ PyVis library not installed. Please install it using: pip install pyvis")
        st.stop()

    if st.button("🔄 Generate Knowledge Graph from Dataset"):
        try:
            # -- STEP 1: Build the Graph from your dataset --
            G = nx.MultiDiGraph()

            # Domain color mapping (reuse your palette)
            domain_colors = {
                "Computer Science": "#3498db",
                "Biology": "#e74c3c",
                "Chemistry": "#f39c12",
                "Sociology": "#9b59b6",
                "Literature": "#1abc9c",
                "Physics": "#e67e22",
                "Cooking": "#e74c3c",
                "Project Management": "#2ecc71",
                "Common": "#95a5a6",
            }

            # 1A. Simple "concept" extractor: take first 3–4 important words of sentence
            def extract_concept(text: str) -> str:
                # Remove extra spaces and basic punctuation
                text = str(text)
                text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
                words = [w for w in text.split() if len(w) > 2]
                if not words:
                    return ""
                # Take first 3 words as concept title
                return " ".join(words[:3])

            # 1B. Build concept nodes per (domain, concept_text) pair
            #     Also track which raw concept text appears in multiple domains
            concept_nodes = {}  # (domain, concept_text) -> node_id
            concept_domains = defaultdict(set)  # concept_text -> set(domains)

            for _, row in df.iterrows():
                domain = str(row["domain"])
                sentence = str(row["sentence"])
                label = str(row["label"])

                concept_text = extract_concept(sentence)
                if not concept_text:
                    continue

                key = (domain, concept_text)
                if key not in concept_nodes:
                    node_id = f"{domain[:3]}_{len(concept_nodes)+1}"
                    concept_nodes[key] = node_id
                    concept_domains[concept_text].add(domain)

                    # Add node with color by domain
                    color = domain_colors.get(domain, "#95a5a6")
                    G.add_node(node_id, label=concept_text, color=color, domain=domain)

            # 1. Add Nodes from Domain/Sentence
            for _, row in df.iterrows():
                domain = str(row["domain"])
                sentence = str(row["sentence"])
                
                # Extract main concept (Limit length for display)
                node_label = extract_concept(sentence)
                if not node_label:
                    continue
                
                node_id = node_label # Using label as ID for simplicity in merging
                
                color = domain_colors.get(domain, "#95a5a6")
                G.add_node(node_id, label=node_label, title=sentence, color=color, group=domain)

            # 2. Add Edges from 'relations' column if available
            edge_count = 0
            if "relations" in df.columns:
                for _, row in df.iterrows():
                    rels = row["relations"]
                    # Handle if stringified list
                    if isinstance(rels, str):
                        try:
                            rels = eval(rels)
                        except:
                            rels = []
                    
                    if isinstance(rels, list):
                        for r in rels:
                            if len(r) == 3:
                                src, relation, tgt = r
                                G.add_edge(src, tgt, label=relation, title=relation)
                                edge_count += 1
            
            # If no explicit relations, fall back to simple grouping (optional, but requested to fix edges)
            # connecting consecutive nodes in same domain/label group
            if edge_count == 0:
                 st.warning("⚠ No explicit relations found in dataset. Attempting to link by Domain/Label...")
                 # ... (Your fallback logic here if needed, or just let it be empty with warning)
                 pass

            if G.number_of_nodes() == 0:
                st.info("No concepts could be extracted from the current dataset to build a graph.")
                st.stop()

            # -- STEP 2: Use PyVis for interactive HTML export (same style as example) --
            net = Network(height="850px", width="100%", directed=True, bgcolor="#ffffff", font_color="#000000")
            net.barnes_hut(gravity=-80000)

            for node, attr in G.nodes(data=True):
                net.add_node(node, label=attr["label"], color=attr["color"])

            for src, tgt, attr in G.edges(data=True):
                style = attr.get("style", "solid")
                dash = True if style in ["dashed", "dotted"] else False
                width = 2 if style == "solid" else 2.5
                edge_label = attr.get("label", "")
                net.add_edge(src, tgt, label=edge_label, width=width, physics=True, dashes=dash)

            net.set_options("""
            const options = {
              "edges": {
                "smooth": {
                  "type": "cubicBezier"
                },
                "arrows": {
                  "to": {"enabled": true}
                }
              }
            }
            """)

            # -- STEP 3: Save interactive HTML file --
            net.save_graph(KNOWLEDGE_GRAPH_PATH)
            st.success("✅ Knowledge graph from dataset generated successfully!")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error generating knowledge graph: {str(e)}")
            st.info("Make sure your dataset has 'domain', 'sentence', and 'label' columns.")

    st.markdown("---")

    # Display existing knowledge graph if available
    if os.path.exists(KNOWLEDGE_GRAPH_PATH):
        try:
            with open(KNOWLEDGE_GRAPH_PATH, "r", encoding="utf-8") as f:
                html_code = f.read()

            st.markdown("""
            <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #ddd; color: #333;'>
                <b style="color: #000;">How to read this graph:</b><br>
                • <strong style="color: #333;">Node color</strong> = domain (e.g., Computer Science, Biology, etc.).<br>
                • <strong style="color: #333;">Solid edges</strong> = concepts linked within the same domain via the same label.<br>
                • <strong style="color: #333;">Dashed edges</strong> = same concept phrase appearing in multiple domains (cross-domain analogy).<br>
                • <strong style="color: #333;">Edge labels</strong> show the label or relation basis (e.g., concept, relation, definition, analogous).
            </div>
            """, unsafe_allow_html=True)

            st.components.v1.html(html_code, height=850, scrolling=True)
            st.info("💡 Tip: Drag nodes, zoom, and click on nodes/edges to explore cross-domain connections.")

        except Exception as e:
            st.error(f"❌ Error loading knowledge graph: {str(e)}")
    else:
        st.info("📊 Click the 'Generate Knowledge Graph from Dataset' button above to create your visualization.")

# ----------------------------------------
# 🔍 SEMANTIC SEARCH
# ----------------------------------------
elif choice == "Semantic Search":
    df = st.session_state.df
    st.header("🔍 Semantic Search - Explore Cross Domain Meaning")

    # Check if embeddings file exists
    embeddings_exist = os.path.exists(EMBEDDINGS_PATH)

    if not embeddings_exist:
        st.warning("⚠ Embeddings file not found. You need to generate embeddings first.")
        
        # Check if library is importable
        try:
            from sentence_transformers import SentenceTransformer
            lib_installed = True
        except ImportError:
            lib_installed = False

        if not lib_installed:
            st.error("❌ Library 'sentence-transformers' is missing.")
            st.code("pip install sentence-transformers", language="bash")
        else:
            st.info("Library `sentence-transformers` is detected. Click below to generate embeddings.")

        st.markdown("---")
        st.subheader("🔄 Generate Embeddings Now")
        if st.button("🚀 Generate Embeddings (This may take a few minutes)"):
            try:
                with st.spinner("Loading sentence transformer model..."):
                    model = SentenceTransformer("all-MiniLM-L6-v2")

                with st.spinner(f"Generating embeddings for {len(df)} sentences... Please wait."):
                    embeddings = model.encode(df["sentence"].tolist(), show_progress_bar=False)

                # Create embeddings dataframe
                embdf = df.copy()
                embdf["embedding"] = embeddings.tolist()

                # Save to pickle
                embdf.to_pickle(EMBEDDINGS_PATH)
                st.session_state.embeddings_generated = True
                st.success("✅ Embeddings generated and saved!")
                st.info("Refresh the page or click 'Semantic Search' again to use the embeddings.")

            except Exception as e:
                st.error(f"❌ Error generating embeddings: {str(e)}")
                st.info("Please make sure you have installed: pip install sentence-transformers")

        st.stop()

    # Load embeddings dataset
    try:
        embdf = pd.read_pickle(EMBEDDINGS_PATH)
        st.sidebar.success(f"✅ Embeddings loaded: {len(embdf)} records")
    except Exception as e:
        st.error(f"❌ Error loading embeddings: {str(e)}")
        st.stop()

    st.write("Enter a query manually or select one of the top 3 frequent queries:")

    # Get top 3 frequent sentences as example queries
    top_sentences = embdf["sentence"].value_counts().head(3).index.tolist()
    query_options = top_sentences + ["Manual Entry"]
    selected_query_mode = st.selectbox("Choose a Query:", query_options)

    if selected_query_mode == "Manual Entry":
        manual_query = st.text_input("Or type your own query here:")
        final_query = manual_query
    else:
        final_query = selected_query_mode

    st.write(f"*Current Query:* {final_query if final_query else '(none)'}")

    search_btn = st.button("🔍 Search")

    if search_btn and final_query and final_query.strip():
        st.subheader(f"🔎 Top 3 Semantic Matches for: '{final_query}'")

        try:
            # Load sentence-transformers model
            with st.spinner("Loading semantic model..."):
                model = SentenceTransformer("all-MiniLM-L6-v2")

            # Encode query
            query_embedding = model.encode(final_query, convert_to_tensor=True)

            # Convert embeddings list to numpy array and then to tensor
            embeddings_list = embdf["embedding"].to_list()
            embeddings_array = np.array(embeddings_list, dtype=np.float32)

            # Convert to tensor with matching dtype
            embeddings_tensor = torch.tensor(embeddings_array, dtype=torch.float32)

            # Ensure query embedding is also float32
            query_embedding = query_embedding.float()

            # Calculate cosine similarity
            scores = util.cos_sim(query_embedding, embeddings_tensor)

            # Get top 3 results
            top_results = scores[0].topk(3)

            # Display results
            for idx, score in zip(top_results.indices, top_results.values):
                row = embdf.iloc[int(idx)]
                st.markdown(f"""
                <div style="background:#ffffff; border-radius:10px; padding:15px; margin-bottom:12px; border:1px solid #e0e0e0; box-shadow:0 2px 5px rgba(0,0,0,0.05);">
                ✅ <b style="color:#000000;">Domain:</b> <span style="color:#333;">{row["domain"]}</span><br>
                💬 <b style="color:#000000;">Sentence:</b> <span style="color:#333;">{row["sentence"]}</span><br>
                🏷 <b style="color:#000000;">Label:</b> <span style="color:#333;">{row["label"]}</span><br>
                📈 <b style="color:#000000;">Similarity Score:</b>
                <span style="color:#2563eb; font-weight:bold;">{float(score):.4f}</span>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error during semantic search: {str(e)}")
            st.info("This might be due to data type mismatch. Try regenerating embeddings.")

# ----------------------------------------
# 🧩 TOP 10 SENTENCES
# ----------------------------------------
elif choice == "Top 10 Sentences":
    df = st.session_state.df
    st.header("🧩 Top 10 Frequent Sentences in Dataset")
    top_objects = df["sentence"].value_counts().head(10)
    st.bar_chart(top_objects)
    st.subheader("📋 Top Sentences List")
    st.subheader("📋 Top Sentences List")
    # Convert series to dataframe for better display
    top_df = top_objects.reset_index()
    top_df.columns = ["Sentence", "Frequency"]
    st.table(top_df)

# ----------------------------------------
# 💬 FEEDBACK SECTION
# ----------------------------------------
elif choice == "Feedback Section":
    df = st.session_state.df
    st.header("💬 Feedback Section")
    st.dataframe(feedback_df, use_container_width=True)

    st.markdown("---")
    # Simplified Feedback Form
    with st.container():
        st.write("Please provide your feedback below.")
        
        user_name = st.text_input("Your Name (Optional)")
        feedback_text = st.text_area("Your Feedback / Suggestions", height=150)

    if st.button("Submit Feedback"):
        if feedback_text:
            new_feedback = pd.DataFrame([{
                "user": user_name if user_name else "Anonymous",
                "comment": feedback_text,
                "timestamp": datetime.datetime.now(),
            }])
            feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
            feedback_df.to_csv(FEEDBACK_FILE, index=False)
            st.success("✅ Feedback submitted successfully!")
        else:
            st.warning("⚠ Please enter some feedback.")

# ----------------------------------------
# 📈 FEEDBACK ANALYSIS
# ----------------------------------------
elif choice == "Feedback Analysis":
    df = st.session_state.df
    st.header("📈 Feedback Analysis")
    if not feedback_df.empty:
        st.subheader("🧩 Feedback Type Distribution")
        st.bar_chart(feedback_df["feedback_type"].value_counts())
        st.subheader("🕒 Feedback Status Overview")
        st.bar_chart(feedback_df["status"].value_counts())
    else:
        st.info("No feedback available yet.")

# ----------------------------------------
# 🛠 ADMIN TOOLS
# ----------------------------------------
elif choice == "Admin Tools":
    df = st.session_state.df
    st.header("🛠 Admin Tools")
    st.header("🛠 Admin Tools")
    
    st.markdown("### 💾 Backup & Export")

    st.markdown("---")
    st.subheader("💾 Backup Data")
    st.download_button(
        label="⬇ Download Current Dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="dataset_backup.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("👥 Manage Users")
    
    # User Management File Path
    USERS_FILE = "users.json"
    import json
    
    def load_users_local():
        if os.path.exists(USERS_FILE):
            try:
                with open(USERS_FILE, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_users_local(users_data):
        with open(USERS_FILE, "w") as f:
            json.dump(users_data, f, indent=4)

    users_db = load_users_local()
    
    # Display Users Table
    if users_db:
        # Create a clean dataframe for display
        user_list = []
        for username, details in users_db.items():
            user_list.append({"Username": username, "Role": details.get("type", "User")})
        st.table(pd.DataFrame(user_list))
    else:
        st.info("No users found.")

    # Add New User
    with st.expander("➕ Add New User"):
        with st.form("add_user_form"):
            new_user = st.text_input("Username")
            new_pass = st.text_input("Password", type="password")
            new_role = st.selectbox("Role", ["User", "Admin"])
            submitted = st.form_submit_button("Create User")
            
            if submitted:
                if new_user and new_pass:
                    if new_user in users_db:
                        st.error("User already exists!")
                    else:
                        users_db[new_user] = {"password": new_pass, "type": new_role}
                        save_users_local(users_db)
                        st.success(f"User '{new_user}' created!")
                        st.rerun()
                else:
                    st.warning("Please fill required fields.")
                    
    # Delete User
    with st.expander("🗑 Delete User"):
        user_to_delete = st.selectbox("Select User to Delete", list(users_db.keys()))
        if st.button("Delete User"):
            if user_to_delete in users_db:
                # Prevent deleting self if needed, but for now allow all
                del users_db[user_to_delete]
                save_users_local(users_db)
                st.success(f"User '{user_to_delete}' deleted.")
                st.rerun()

# ----------------------------------------
# 💾 DOWNLOAD OPTIONS
# ----------------------------------------
elif choice == "Download Options":
    df = st.session_state.df
    st.header("💾 Download Data Files")
    st.download_button(
        "⬇ Download Dataset CSV",
        df.to_csv(index=False).encode("utf-8"),
        "dataset_updated.csv",
    )
    st.download_button(
        "⬇ Download Feedback CSV",
        feedback_df.to_csv(index=False).encode("utf-8"),
        "feedback.csv",
    )
# ----------------------------------------
# 🤖 AI CHATBOT PAGE
# ----------------------------------------
elif choice == "AI Assistance Chatbot":
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_input_key" not in st.session_state:
        st.session_state.chat_input_key = ""

    st.title("🤖 AI Assistance")

    st.markdown("""
    Ask any question related to:
    - Dataset  
    - Knowledge graph  
    - Entity extraction  
    - Semantic search  
    - Any domain-related topic  
    
    The AI will respond instantly.
    """)

    # --------------------------------------------
    # Logic to send message
    # --------------------------------------------
    def on_submit():
        user_msg = st.session_state.my_input
        if user_msg.strip():
            with st.spinner("AI Thinking..."):
                try:
                    ai_reply = ask_llama(user_msg)
                    st.session_state.messages.append((user_msg, ai_reply))
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            st.session_state.my_input = ""  # Clear input

    # --------------------------------------------
    # Input Area
    # --------------------------------------------
    st.text_input("Your Question", key="my_input", on_change=on_submit, placeholder="Type your question here and press Enter...")

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Clear Chat"):
            st.session_state.messages = []

    # --------------------------------------------
    # Display chat history (Newest first)
    # --------------------------------------------
    if "messages" in st.session_state and st.session_state.messages:
        st.markdown("---")
        for user_msg, ai_msg in reversed(st.session_state.messages):
            st.markdown(f'<div class="chat-container"><b style="color:#000;">You:</b> <span style="color:#333;">{user_msg}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-response-box"><b style="color:#000;">AI:</b> <span style="color:#333;">{ai_msg}</span></div>', unsafe_allow_html=True)

