import streamlit as st
import subprocess
import threading
import queue
import sys
import os

# ==========================================
# Streamlit App UI
# ==========================================

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("📚 AI Research Assistant – Research Pipeline UI")

st.markdown("This UI runs your backend pipeline (`backend/pipeline.py`) "
            "and streams the cinematic logs live into Streamlit.")

# ------------------------------------------
# User Inputs
# ------------------------------------------

query = st.text_input("🔍 Enter your research query:")
num_papers = st.number_input("📄 Number of papers to index", min_value=1, max_value=50, value=5)

run_btn = st.button("🚀 Run Pipeline")

# This is where logs will appear
log_box = st.empty()


# ==========================================
# Helper: Stream backend output LIVE
# ==========================================

def stream_subprocess(process, q):
    """Reads stdout line-by-line and pushes into a queue."""
    for line in iter(process.stdout.readline, ''):
        q.put(line)
    process.stdout.close()
    q.put(None)  # signal process complete


# ==========================================
# Run pipeline
# ==========================================

if run_btn:
    if not query.strip():
        st.error("Please enter a valid research query.")
        st.stop()

    BACKEND_FILE = os.path.join("backend", "pipeline.py")

    if not os.path.exists(BACKEND_FILE):
        st.error("❌ backend/pipeline.py not found! Check your folder structure.")
        st.stop()

    st.success("⚙️ Starting pipeline… logs will stream below.")

    # Launch backend subprocess
    process = subprocess.Popen(
        [sys.executable, BACKEND_FILE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Send inputs to backend’s input() calls
    process.stdin.write(query + "\n")
    process.stdin.write(str(num_papers) + "\n")
    process.stdin.flush()

    # Queue & thread for live output
    q = queue.Queue()
    t = threading.Thread(target=stream_subprocess, args=(process, q))
    t.start()

    logs = ""

    # Stream logs live into Streamlit
    while True:
        line = q.get()
        if line is None:
            break
        logs += line
        log_box.text(logs)

    process.wait()
    st.success("🎉 Pipeline finished!")
