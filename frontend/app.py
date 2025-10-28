# frontend/app.py

import streamlit as st
import sys
import os
import io
from contextlib import redirect_stdout

# Adjust Python path so backend modules are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from pipeline import main as run_pipeline

# ---------------------------------------------------------------------
# 🧠 Streamlit App Configuration
# ---------------------------------------------------------------------
st.set_page_config(page_title="AI Research Downloader", page_icon="🧠", layout="wide")

st.title("🧠 AI Research Downloader & Indexer")
st.markdown("""
Search and summarize the latest research papers directly from **ArXiv**, 
generate embeddings, and store them in **Pinecone**.
""")

# ---------------------------------------------------------------------
# 🔍 User Input Section
# ---------------------------------------------------------------------
query = st.text_input("Enter a research topic or paper title:", placeholder="e.g. Attention is all you need")

if st.button("Run Pipeline 🚀"):
    if not query.strip():
        st.warning("⚠️ Please enter a valid query before running.")
    else:
        st.info(f"Running RAG pipeline for: **{query}**")
        progress = st.progress(0)
        log_output = io.StringIO()

        with st.spinner("Fetching and processing papers..."):
            with redirect_stdout(log_output):
                try:
                    run_pipeline(query)  # ✅ Pass user query here
                    success = True
                except Exception as e:
                    success = False
                    st.error(f"❌ Pipeline failed: {e}")

        progress.progress(100)
        if success:
            st.success("✅ Pipeline complete! Check details below.")
        else:
            st.error("⚠️ Pipeline encountered errors. Check logs for details.")

        # ---------------------------------------------------------------------
        # 🧾 Logs Section
        # ---------------------------------------------------------------------
        logs = log_output.getvalue()
        st.subheader("🧾 Pipeline Logs")
        st.text_area("Detailed Execution Logs", logs, height=400)

        # ---------------------------------------------------------------------
        # 📄 Summaries Section
        # ---------------------------------------------------------------------
        summaries_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "backend")

        st.subheader("📄 Generated Summaries")
        found_any = False

        for root, _, files in os.walk(summaries_dir):
            for file in files:
                if file.endswith(".txt"):
                    found_any = True
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    with st.expander(f"🧩 {file.replace('.txt', '')}", expanded=False):
                        st.write(content)

        if not found_any:
            st.info("No summaries found yet — run the pipeline to generate them!")

# ---------------------------------------------------------------------
# 📌 Sidebar Info
# ---------------------------------------------------------------------
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
**AI Research Downloader** helps you:
- 🔍 Search ArXiv papers by topic  
- 🧠 Summarize abstracts using LLMs  
- 📊 Generate embeddings  
- 🪣 Store results in Pinecone  

Built with ❤️ using Python, Groq, and Streamlit.
""")
