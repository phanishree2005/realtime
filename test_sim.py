"""
Quick validation test for CASDS and comparative algorithms.
"""

from src.casds.simulator import ACPSSimulator
from src.casds.schedulers import (
    CASDSScheduler,
    HetSchedScheduler,
    ADSScheduler,
    ASDYSScheduler,
    TD_ADSScheduler,
    ULSScheduler
)

def main():
    print("Initializing ACPSSimulator...")
    sim = ACPSSimulator()
    print("Loading test trace with 100 DAGs...")
    trace = sim.load_trace_file("inputs/ad/trace_files/ad_trace.trc", max_dags=100)
    print(f"Loaded {len(trace)} DAGs.")

    schedulers = [
        CASDSScheduler(),
        HetSchedScheduler(),
        ADSScheduler(),
        ASDYSScheduler(),
        TD_ADSScheduler(),
        ULSScheduler()
    ]

    print("\n" + "="*80)
    print(f"{'Algorithm':<12} | {'DSR(Total)':<11} | {'DSR(Crit)':<10} | {'DSR(NonCrit)':<12} | {'Makespan (s)':<12} | {'ET (DAG/s)':<10}")
    print("="*80)

    for sched in schedulers:
        res = sim.run_simulation(sched, trace)
        print(f"{res.scheduler_name:<12} | {res.dsr_total:>9.2f}% | {res.dsr_critical:>8.2f}% | {res.dsr_noncritical:>10.2f}% | {res.makespan/1000:>11.2f}s | {res.effective_throughput:>9.2f}")

    print("="*80)
    print("Validation Test Passed Successfully!")

if __name__ == '__main__':
    main()
