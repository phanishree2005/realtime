# CASDS: Context-Aware-Slack-Driven Dynamic Task Scheduling for Automotive Cyber-Physical Systems

> **Research Paper Implementation**: IEEE Transactions on Industrial Informatics  
> **Authors**: Chuang Zhang, Sifa Zheng, Chaoyi Chen, Haoran Li, Hao Cheng, Qing Xu, Keqiang Li (Tsinghua University)

---

## 📌 Project Overview
This project implements the complete research paper on **dynamic DAG task scheduling on heterogeneous Automotive Cyber-Physical Systems (ACPS)**.

Autonomous vehicles (AV) run concurrent, multi-criticality software submodules (Object Detection, Tracking, Localization, Fusion, Mission/Motion Planning) on resource-constrained heterogeneous hardware (CPU, GPU, ASIC accelerators). 

### Key Contributions Implemented:
1. **Multilevel Task-Application-System Model**:
   - Heterogeneous processing elements (PEs): 8 Arm CPUs, 2 Maxwell GPUs, and specialized hardware accelerators (DET, TRA, LOC).
   - Real-world ADSuite DAG applications (Type 1 – Type 4 DAGs from Fig. 4).
   - Mixed-Criticality (safety critical vs non-critical, and ISO 26262 ASIL A-D).
2. **Context-Aware Slack (CAS)**:
   - Depicts dynamic task urgency $U_i^a = CAS(v_i^a, P_t^s) = T_{rem}^{a,i} - PLCT(Q_u^a, P_t^s)$.
   - Calculates **Predictive Least Completion Time (PLCT)** incorporating both completed history and future execution queues.
   - **Negative Slack Deferral**: Tasks with $CAS < 0$ (impossible to meet deadline) are gracefully deferred, avoiding the classic **scheduling domino effect**.
3. **Adaptive Priority via EWM-TOPSIS**:
   - Balances Criticality ($\mathcal{C}_i^a$) and Urgency ($-CAS$) dynamically using the **Entropy-Weight Method (EWM)** and **TOPSIS**.
4. **Non-preemptive EFT Allocation**:
   - Allocates ready tasks to the PE with the minimum Earliest Finish Time (EFT).
5. **Comparative Benchmark Suite**:
   - **CASDS** (Proposed)
   - **HetSched** (Amarnath et al., IEEE CAL 2021)
   - **ADS** (Xie et al., IEEE TVT 2017)
   - **ASDYS** (Bai et al., IEEE TII 2021)
   - **TD_ADS** (Zhao et al., IEEE Access 2024)
   - **ULS** (Cai et al., IEEE TSC 2025)

---

## 🚀 How to Run the Project

### 1. Launch the Interactive Web Dashboard
The project includes a web dashboard with real-time Gantt charts, live algorithm comparisons, and metric graphs:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run Comparative Benchmark Experiments
To run the automated benchmark comparing all 6 schedulers across arrival rates $\lambda \in \{20, 30, 40\}$ and critical ratios $R_{crit} \in \{30\%, 60\%, 90\%\}$:
```bash
python -m src.casds.benchmarks
```
Results will be displayed in the terminal and saved to `results_experiments.json`.

### 3. Run Validation Quick Test
To quickly test all 6 schedulers on the 100-DAG test trace:
```bash
python test_sim.py
```

### 4. Run through the STOMP Simulator
You can also run directly through the STOMP simulation engine:
```bash
python simulator/stomp_main.py -c inputs/stomp_real.json
```

---

## 📊 Performance Metrics Computed

* **$DSR(S)$**: Overall Deadline Satisfaction Ratio ($\frac{|S^{se}|}{|S|}$).
* **$DSR(S_{crit})$**: Deadline Satisfaction Ratio of Safety-Critical DAGs.
* **Makespan ($MS$)**: Total system completion time.
* **PE Utilization ($U$)**: Average busy-time percentage across all heterogeneous computing units.
* **Effective Throughput ($ET$)**: Rate of successfully executed applications per second ($\frac{|S^{se}|}{Makespan}$).

---

## 📁 Directory Structure
```
realtime/
├── app.py                            # Streamlit Web Application (GUI & Gantt charts)
├── test_sim.py                       # Quick validation test script
├── PROJECT_GUIDE.md                  # Complete project documentation
├── inputs/
│   ├── ad/                           # Real-World Autonomous Driving Benchmark (ADSuite)
│   │   ├── dag_input/                # Type 1 - Type 4 GraphML & Table I execution profiles
│   │   └── trace_files/              # 10,100 preloaded DAG traces (ad_trace_0.1 to 1.0)
│   └── stomp_real.json               # Platform & hardware configuration
├── src/
│   └── casds/
│       ├── core.py                   # DAG, Task, PE, CAS, PLCT, and EWM-TOPSIS implementation
│       ├── schedulers.py             # CASDS + 5 comparative schedulers (HetSched, ADS, ASDYS, TD_ADS, ULS)
│       ├── simulator.py              # Discrete-event simulation engine
│       └── benchmarks.py             # Experiment runner for Figures 5, 6, 7 & Tables II, III
├── simulator/
│   ├── stomp_main.py                 # STOMP simulator CLI
│   └── task_policies/
│       ├── casds.py                  # STOMP plug-in policy for CASDS
│       └── ms_hyb_update.py          # STOMP plug-in policy for HetSched
└── utils/                            # Trace generators & STOMP visualization tools
```
