"""
Unit and integration tests for CASDS models, schedulers, and metrics.
"""

import unittest
import numpy as np

from casds.models.task import Task, TaskStatus
from casds.models.platform import HeterogeneousPlatform, ProcessingElement
from casds.models.dag import DAGApplication
from casds.benchmarks.adsuite_benchmark import create_adsuite_platform, create_adsuite_dag
from casds.benchmarks.selfpilot_benchmark import create_selfpilot_platform, create_selfpilot_dag
from casds.benchmarks.trace_generator import TraceGenerator
from casds.schedulers.casds_scheduler import CASDSScheduler
from casds.schedulers.baselines import (
    ADSScheduler, ASDYSScheduler, HetSchedScheduler, TD_ADSScheduler, ULSScheduler
)
from casds.schedulers.ablation_schedulers import AblationScheduler
from casds.simulation.simulator import ACPSSimulator


class TestCASDS(unittest.TestCase):

    def test_platform_creation(self):
        platform = create_adsuite_platform()
        self.assertEqual(platform.num_pes, 13)
        self.assertEqual(len(platform.get_pes_for_type("CPU")), 8)
        self.assertEqual(len(platform.get_pes_for_type("GPU")), 2)
        self.assertEqual(len(platform.get_pes_for_type("DET_ASIC")), 1)
        self.assertEqual(len(platform.get_pes_for_type("TRA_ASIC")), 1)
        self.assertEqual(len(platform.get_pes_for_type("LOC_ASIC")), 1)

    def test_adsuite_dags(self):
        platform = create_adsuite_platform()
        for dtype, expected_tasks in [("Type 1", 6), ("Type 2", 8), ("Type 3", 7), ("Type 4", 9)]:
            dag = create_adsuite_dag(app_id=1, dag_type=dtype, criticality=2)
            self.assertEqual(len(dag.tasks), expected_tasks)
            ordered = dag.compute_heft_ranks(platform)
            self.assertEqual(len(ordered), expected_tasks)
            entry = dag.get_entry_tasks()[0]
            exit_t = dag.get_exit_tasks()[0]
            self.assertGreater(dag.tasks[entry].upward_rank, dag.tasks[exit_t].upward_rank)

    def test_selfpilot_dag(self):
        platform = create_selfpilot_platform()
        self.assertEqual(platform.num_pes, 11)
        dag = create_selfpilot_dag(app_id=1, criticality=2)
        self.assertEqual(len(dag.tasks), 9)
        dag.compute_heft_ranks(platform)
        self.assertGreater(dag.tasks[0].upward_rank, 0)

    def test_casds_scheduler_simulation(self):
        platform = create_adsuite_platform()
        tg = TraceGenerator(seed=42)
        trace = tg.generate_adsuite_trace(num_dags=10, arrival_rate_lambda=20.0, critical_ratio=0.5)

        scheduler = CASDSScheduler()
        sim = ACPSSimulator(platform, scheduler)
        metrics = sim.run(trace)

        self.assertEqual(metrics.num_total, 10)
        self.assertGreaterEqual(metrics.dsr, 0.0)
        self.assertGreater(metrics.makespan_ms, 0.0)
        self.assertGreaterEqual(metrics.utilization, 0.0)
        self.assertGreaterEqual(metrics.effective_throughput, 0.0)

    def test_baseline_schedulers(self):
        platform = create_adsuite_platform()
        tg = TraceGenerator(seed=123)
        trace = tg.generate_adsuite_trace(num_dags=5, arrival_rate_lambda=20.0, critical_ratio=0.5)

        schedulers = [
            ADSScheduler(),
            ASDYSScheduler(),
            HetSchedScheduler(),
            TD_ADSScheduler(),
            ULSScheduler(),
            CASDSScheduler(),
        ]

        for s in schedulers:
            sim = ACPSSimulator(platform, s)
            m = sim.run(trace)
            self.assertEqual(m.num_total, 5)
            self.assertTrue(0.0 <= m.dsr <= 100.0)

    def test_ablation_variants(self):
        platform = create_adsuite_platform()
        tg = TraceGenerator(seed=42)
        trace = tg.generate_adsuite_trace(num_dags=5, arrival_rate_lambda=20.0, critical_ratio=0.5)

        for urg in ["Trem", "T-Ddl", "HetSched", "CAS"]:
            for prio in ["w/o_Crit", "Multikey", "Adaptive"]:
                s = AblationScheduler(urgency_mode=urg, priority_mode=prio)
                sim = ACPSSimulator(platform, s)
                m = sim.run(trace)
                self.assertEqual(m.num_total, 5)


if __name__ == "__main__":
    unittest.main()
