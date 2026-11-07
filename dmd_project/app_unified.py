"""
Unified Streamlit App: ACPS Real-Time Scheduling & Priority Platform + DMS Driver Perception & Unified HUD Studio
Merged multi-domain AI and cyber-physical systems platform combining:
1. DMS Driver Perception & Unified HUD Studio (Gaze 9-Zone, Drowsiness PERCLOS, Distraction 10-Class, CASDS Safety)
2. CASDS Dynamic Heterogeneous DAG Task Scheduling (IEEE TII 2026)
3. Priority Assignment Strategies (Uniform, Arithmetic, Geometric, Logarithmic)
4. Multiprocessor Systems: Task Allocation, Utilization & UMA/NUMA
5. Integrated Evaluation Studio: CASDS + Priority Shaping Curves
"""

import os
import sys
import runpy

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

app_path = os.path.join(ROOT_DIR, "app.py")
runpy.run_path(app_path, run_name="__main__")
