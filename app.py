import streamlit as st
import subprocess
import os
import uuid
import difflib
import re
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ------------------------
# Streamlit Config
# ------------------------
st.set_page_config(page_title="OptiDiff", layout="wide")
st.title("🔬 OptiDiff - LLVM IR Diff & Pass Pipeline")

# ------------------------
# Sidebar: Optimization Levels
# ------------------------
st.sidebar.header("Optimization Levels")
left_opt = st.sidebar.radio("Left Pane (Before)", ["-O0", "-O1", "-O2", "-O3"], index=0)
right_opt = st.sidebar.radio("Right Pane (After)", ["-O0", "-O1", "-O2", "-O3"], index=2)

# ------------------------
# File Upload
# ------------------------
uploaded_file = st.file_uploader("Upload C/C++ Source File", type=["c", "cpp"])

# ------------------------
# Helpers
# ------------------------
def run_clang_and_opt(filepath, opt_level, file_id):
    """Generate LLVM IR + logs for a given optimization level"""
    ir_path = f"uploads/{file_id}{opt_level}.ll"
    log_path = f"uploads/{file_id}{opt_level}_log.txt"

    # Emit IR
    subprocess.run(["clang", "-S", "-emit-llvm", opt_level, filepath, "-o", ir_path])

    # Run opt with logs enabled
    subprocess.run([
        "opt", "-S", ir_path, "-o", os.devnull,
        f"-passes=default<{opt_level.replace('-', '')}>",
        "-print-before-all", "-print-after-all",
        "-time-passes", "-stats",
        "-debug-pass-manager"
    ], stdout=open(log_path, "w"), stderr=subprocess.STDOUT)

    return ir_path, log_path

def parse_pass_pipeline(log_text):
    """Parse -debug-pass-manager output into categories"""
    categories = {"Running": [], "Analysis": [], "Invalidating": [], "Skipping": []}
    for line in log_text.splitlines():
        if "Running pass" in line:
            categories["Running"].append(line.strip())
        elif "Running analysis" in line:
            categories["Analysis"].append(line.strip())
        elif "Invalidating" in line:
            categories["Invalidating"].append(line.strip())
        elif "Skipping" in line:
            categories["Skipping"].append(line.strip())
    return categories

def extract_ir_snapshots(log_text):
    """
    Extract IR before/after snapshots per pass using a robust splitting strategy.
    This version is designed to be more flexible with log file formatting.
    """
    snapshots = {}
    
    # This regex flexibly captures the key parts of a header, including optional semicolons.
    ir_dump_regex = re.compile(r";?\s?\*\*\* IR Dump (Before|After) ([\w:]+) on (.*?)\s*\*\*\*")
    
    # Split the log by the header. The capturing groups will be included in the list.
    blocks = ir_dump_regex.split(log_text)
    if len(blocks) <= 1:
        return {} # No matches found.

    # The result of split is: [text_before, state1, pass1, target1, content1, state2, pass2, target2, content2, ...]
    # We process the list in chunks of 4.
    for i in range(1, len(blocks), 4):
        state = blocks[i].lower()
        pass_name = blocks[i+1]
        target_raw = blocks[i+2].strip()
        content = blocks[i+3].strip()

        if not content:
            continue

        # Clean and standardize the key used for the dictionary.
        # This handles formats like "function 'main'", "[module]", "'main'", etc.
        target_clean = re.sub(r"^(function|module)\s*", "", target_raw).strip("'[]")
        key = f"{pass_name} on {target_clean}"

        # Logic to append a new 'before' snapshot or update the 'after' for the last run.
        if state == 'before':
            snapshots.setdefault(key, []).append({'before': content, 'after': None})
        elif state == 'after':
            if snapshots.get(key) and snapshots[key] and snapshots[key][-1]['after'] is None:
                snapshots[key][-1]['after'] = content
            else:
                # This 'after' has no preceding 'before', so create a new entry for it.
                snapshots.setdefault(key, []).append({'before': '# Before state not found in log', 'after': content})
    
    return snapshots


def parse_time_and_stats(log_text):
    """Parse -time-passes and -stats output with more robust regex."""
    timings = {}
    stats = {}
    in_timing_section = False
    in_stats_section = False

    for line in log_text.splitlines():
        # Detect start of sections
        if "Pass execution timing report" in line:
            in_timing_section = True
            in_stats_section = False
            continue
        if "Statistics Collected" in line:
            in_stats_section = True
            in_timing_section = False
            continue

        # Parse timing data from the report table
        if in_timing_section:
            # Regex for lines like: 0.0010 ( 0.0%) ... 0.0010 (14.1%)  Module Verifier
            time_match = re.match(r'\s*\d+\.\d+.*?\s+(\d+\.\d+)\s+\(.*?%\)\s+(.*)', line)
            if time_match:
                # Wall time is in seconds, convert to ms for better readability
                time_val_s = float(time_match.group(1))
                time_val_ms = time_val_s * 1000
                pass_name = time_match.group(2).strip()
                if time_val_ms > 0: # Only record passes that took a measurable amount of time
                    timings[pass_name] = f"{time_val_ms:.2f} ms"
        
        # Parse stats data from the report table
        if in_stats_section:
            # Regex for lines like: 16 dagcombine - Number of nodes combined
            stat_match = re.match(r'\s*(\d+)\s+([\w\s.-]+?)\s+-\s+(.*)', line)
            if stat_match:
                stat_val = stat_match.group(1).strip()
                key_name = stat_match.group(2).strip()
                desc = stat_match.group(3).strip()
                full_key = f"{key_name}: {desc}"
                stats[full_key] = stat_val

    return timings, stats

def render_diff(left_ir, right_ir, left_header="Before", right_header="After"):
    """Render side-by-side diff with visual enhancements."""
    left_lines = left_ir.splitlines()
    right_lines = right_ir.splitlines()
    diff = list(difflib.ndiff(left_lines, right_lines))

    html_lines = ["""
    <style>
    .diff-container {
        font-family: 'Courier New', Courier, monospace;
        font-size: 14px;
        border-radius: 8px;
        box-shadow: 0 4px 12px 0 rgba(0,0,0,0.1);
        overflow: hidden; /* Clips content to border-radius */
    }
    table {width:100%; border-collapse:collapse;}
    td, th {padding:4px 8px; vertical-align:top; border-bottom: 1px solid #e0e0e0;}
    .line-num {width:40px; text-align:right; color:#888; background-color: #f7f7f7; user-select: none;}
    .added {background-color:#e6ffed;}
    .removed {background-color:#ffeef0;}
    .content {white-space: pre-wrap; word-wrap: break-word;}
    th {background-color: #f2f2f2; text-align: left; font-size: 16px; position: sticky; top: 0;}
    </style>
    <div class="diff-container">
    <table>
    """]
    html_lines.append(f'<tr><th>-</th><th style="width:50%;">{left_header}</th><th>+</th><th style="width:50%;">{right_header}</th></tr>')
    
    l_idx, r_idx, i = 0, 0, 0
    while i < len(diff):
        line = diff[i]
        if line.startswith("- "):
            l_idx += 1
            if i + 1 < len(diff) and diff[i+1].startswith("+ "):
                r_idx += 1
                html_lines.append(f"<tr><td class='line-num'>{l_idx}</td><td class='removed content'>{line[2:]}</td><td class='line-num'>{r_idx}</td><td class='added content'>{diff[i+1][2:]}</td></tr>")
                i += 1
            else:
                html_lines.append(f"<tr><td class='line-num'>{l_idx}</td><td class='removed content'>{line[2:]}</td><td></td><td></td></tr>")
        elif line.startswith("+ "):
            r_idx += 1
            html_lines.append(f"<tr><td></td><td></td><td class='line-num'>{r_idx}</td><td class='added content'>{line[2:]}</td></tr>")
        elif line.startswith("  "):
            l_idx += 1
            r_idx += 1
            html_lines.append(f"<tr><td class='line-num'>{l_idx}</td><td class='content'>{line[2:]}</td><td class='line-num'>{r_idx}</td><td class='content'>{line[2:]}</td></tr>")
        i += 1

    html_lines.append("</table></div>")
    st.components.v1.html("\n".join(html_lines), height=600, scrolling=True)

# ------------------------
# Main Execution
# ------------------------
if uploaded_file:
    file_id = str(uuid.uuid4())
    filepath = f"uploads/{file_id}.c"
    os.makedirs("uploads", exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("🚀 Run Analysis"):
        with st.spinner("Compiling and analyzing optimization passes..."):
            left_ir_path, _ = run_clang_and_opt(filepath, left_opt, file_id+"_L")
            right_ir_path, right_log_path = run_clang_and_opt(filepath, right_opt, file_id+"_R")

            st.session_state.left_code = Path(left_ir_path).read_text(encoding="utf-8", errors="ignore")
            st.session_state.right_code = Path(right_ir_path).read_text(encoding="utf-8", errors="ignore")
            log_text = Path(right_log_path).read_text(encoding="utf-8", errors="ignore")

            st.session_state.pipeline = parse_pass_pipeline(log_text)
            st.session_state.snapshots = extract_ir_snapshots(log_text)
            st.session_state.timings, st.session_state.stats = parse_time_and_stats(log_text)
            st.session_state.ran_analysis = True

if st.session_state.get("ran_analysis"):
    st.header(f"⚙️ Final IR Comparison: `{left_opt}` vs `{right_opt}`", divider="rainbow")
    render_diff(st.session_state.left_code, st.session_state.right_code, left_header=left_opt, right_header=right_opt)
    st.info("This view shows the final generated IR after applying all optimizations for each level.")

    st.header(f"🚀 Pass Analysis for `{right_opt}`", divider="rainbow")
    
    tab1, tab2, tab3 = st.tabs(["📊 Pass Explorer", "📈 Pass Metrics", "📜 Full Pass Pipeline"])

    with tab1:
        st.subheader("Interactive Pass Inspector")
        st.info("Select any pass run to see the IR before and after its execution.")
        
        # Generate a list of all available pass snapshots for the dropdown
        pass_options = []
        for pass_key, runs in st.session_state.snapshots.items():
            for i, run in enumerate(runs):
                pass_options.append(f"{pass_key} (#{i+1})")

        if not pass_options:
            st.warning("No IR snapshots were captured. Check the LLVM log for details.")
        else:
            selected_pass_unique = st.selectbox("Select a pass run to inspect:", pass_options)
            if selected_pass_unique:
                # Parse the selection to get the key and run index
                match = re.match(r"(.*) \(#(\d+)\)", selected_pass_unique)
                if match:
                    snapshot_key = match.group(1)
                    run_index = int(match.group(2)) - 1
                    
                    if snapshot_key in st.session_state.snapshots and len(st.session_state.snapshots[snapshot_key]) > run_index:
                        snapshot = st.session_state.snapshots[snapshot_key][run_index]
                        
                        st.subheader("IR Transformation by Pass")
                        ir_before = snapshot.get('before', '# "Before" IR dump not found for this run.')
                        ir_after = snapshot.get('after', '# "After" IR dump not found for this run.')
                        render_diff(ir_before, ir_after)

    with tab2:
        st.subheader("Performance and Optimization Metrics")
        st.info("These charts and metrics provide an overview of all passes, not just the ones that changed the IR.")

        # --- Pre-calculate change-inducing passes for metrics ---
        change_inducing_passes = []
        if 'snapshots' in st.session_state:
            for key, runs in st.session_state.snapshots.items():
                for i, run in enumerate(runs):
                    if run.get('before') and run.get('after') and run['before'] != run['after']:
                        change_inducing_passes.append(f"{key} (#{i+1})")

        # --- Summary Metrics ---
        if st.session_state.timings:
            total_passes_run = len(st.session_state.pipeline.get("Running", []))
            total_changing_passes = len(change_inducing_passes)
            
            max_time = 0
            slowest_pass = "N/A"
            for pass_name, time_str in st.session_state.timings.items():
                try:
                    time_val = float(time_str.split()[0])
                    if time_val > max_time:
                        max_time = time_val
                        slowest_pass = pass_name
                except (ValueError, IndexError):
                    continue
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Passes Run", total_passes_run)
            col2.metric("Passes Changing IR", total_changing_passes)
            col3.metric("Most Time-Consuming", slowest_pass, f"{max_time:.2f} ms")
        
        # --- Timings Visualization ---
        st.markdown("##### ⏱️ Pass Execution Times")
        if st.session_state.timings:
            timing_data = []
            for pass_name, time_str in st.session_state.timings.items():
                try:
                    time_val = float(time_str.split()[0])
                    timing_data.append({"Pass": pass_name, "Time (ms)": time_val})
                except (ValueError, IndexError):
                    continue

            if timing_data:
                timing_df = pd.DataFrame(timing_data).sort_values("Time (ms)", ascending=False).set_index("Pass")
                st.bar_chart(timing_df)
            else:
                st.warning("No timing data available to display.")
        else:
            st.warning("No timing data was generated.")

        # --- Pass Distribution Visualization ---
        st.markdown("##### 📊 Pass Execution Distribution")
        if st.session_state.pipeline.get("Running"):
            # Extract just the pass name from the full "Running pass: ..." string
            pass_names = []
            for p in st.session_state.pipeline["Running"]:
                match = re.search(r"Running pass: ([\w:]+)", p)
                if match:
                    pass_names.append(match.group(1))
            
            if pass_names:
                pass_counts = pd.Series(pass_names).value_counts()
                
                # Group smaller slices into 'Others' for clarity
                if len(pass_counts) > 5:
                    top_5 = pass_counts.head(5)
                    others_count = pass_counts.iloc[5:].sum()
                    others_series = pd.Series([others_count], index=['Others'])
                    final_pass_counts = pd.concat([top_5, others_series])
                else:
                    final_pass_counts = pass_counts

                # Use Matplotlib to create a pie chart
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(final_pass_counts, labels=final_pass_counts.index, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)
            else:
                st.warning("No running pass data to display for pie chart.")
        else:
            st.warning("No running pass data available.")


    with tab3:
        st.subheader("Complete Pass Pipeline Log")
        st.info("This is the raw output from '-debug-pass-manager', showing the precise order of all operations.")
        for cat, passes in st.session_state.pipeline.items():
            if passes:
                with st.expander(f"{cat} ({len(passes)})", expanded=(cat=="Running")):
                    st.code("\n".join(passes), language="log")

