"""
Comprehensive Unit Tests for Priority Scheduling Project.
Tests dataset inspection, task modeling, priority assignment strategies,
analytical RTA engine, discrete-event simulation, and synthetic task generation.
"""

import unittest
import math
import numpy as np

from src.nuscenes_inspector import NuScenesInspector
from src.task_model import Task, Job, TaskSet
from src.priority_assigner import PriorityAssigner
from src.rta_analyzer import RTAAnalyzer
from src.discrete_event_simulator import DiscreteEventSimulator
from src.synthetic_task_generator import SyntheticTaskGenerator


class TestPriorityScheduling(unittest.TestCase):

    def test_nuscenes_inspector(self):
        """Verify nuScenes inspector reads metadata in read-only mode."""
        inspector = NuScenesInspector()
        summary = inspector.get_dataset_summary()
        self.assertIn("num_sensors", summary)
        self.assertGreaterEqual(summary.get("num_sensors", 0), 1)

        clutter = inspector.get_scene_clutter_distribution()
        self.assertIn("mean_objects_per_frame", clutter)
        self.assertGreater(clutter["mean_objects_per_frame"], 0)

    def test_task_model_and_cloning(self):
        """Test task attributes, utilization, and cloning."""
        t1 = Task(task_id=1, name="T1", period=50.0, deadline=50.0, wcet=10.0)
        t2 = Task(task_id=2, name="T2", period=100.0, deadline=100.0, wcet=20.0)

        ts = TaskSet([t1, t2])
        self.assertAlmostEqual(ts.total_utilization, 0.4, places=4)

        # Verify cloning preserves independence
        cloned = ts.clone()
        cloned[0].priority = 100
        self.assertEqual(ts[0].priority, 0)
        self.assertEqual(cloned[0].priority, 100)

    def test_priority_strategies_values_and_monotonicity(self):
        """Test all 4 priority strategies produce valid, monotonically increasing values."""
        assigner = PriorityAssigner(p_min=1, p_max=256)
        n = 10

        for strat in PriorityAssigner.ALL_STRATEGIES:
            p_vals = assigner.compute_priority_values(strat, n, enforce_unique=True)
            self.assertEqual(len(p_vals), n, f"Strategy {strat} failed length check")
            # Bounds check
            self.assertTrue(all(1 <= p <= 256 for p in p_vals), f"Strategy {strat} out of bounds: {p_vals}")
            # Monotonicity check
            for i in range(1, n):
                self.assertGreaterEqual(p_vals[i], p_vals[i - 1], f"Strategy {strat} not monotonic: {p_vals}")

    def test_configurable_logarithmic_priority(self):
        """Test configurable curvature beta and base in logarithmic strategy."""
        assigner_default = PriorityAssigner(p_min=1, p_max=100, logarithmic_beta=9.0, logarithmic_base=math.e)
        p_def = assigner_default.compute_priority_values(PriorityAssigner.STRATEGY_LOGARITHMIC, 5, enforce_unique=False)

        assigner_steep = PriorityAssigner(p_min=1, p_max=100, logarithmic_beta=99.0, logarithmic_base=math.e)
        p_steep = assigner_steep.compute_priority_values(PriorityAssigner.STRATEGY_LOGARITHMIC, 5, enforce_unique=False)

        # Higher beta produces faster initial growth
        self.assertGreaterEqual(p_steep[1], p_def[1])

        # Test custom pluggable function
        custom_func = lambda x, pmin, pmax: pmin + (pmax - pmin) * (math.log10(1 + 9 * x))
        assigner_custom = PriorityAssigner(p_min=1, p_max=100, custom_log_func=custom_func)
        p_cust = assigner_custom.compute_priority_values(PriorityAssigner.STRATEGY_LOGARITHMIC, 5)
        self.assertEqual(p_cust[0], 1)
        self.assertEqual(p_cust[-1], 100)

    def test_rta_analyzer_classical_benchmark(self):
        """
        Verify RTA on classical Liu & Layland benchmark:
        T1 = (T=5, C=2), T2 = (T=7, C=2), T3 = (T=15, C=4)
        Under Rate-Monotonic (P1 > P2 > P3):
        R1 = 2 <= 5 (schedulable)
        R2 = 2 + ceil(R2/5)*2 -> R2=4 <= 7 (schedulable)
        R3 = 4 + ceil(R3/5)*2 + ceil(R3/7)*2 -> R3=12 <= 15 (schedulable)
        """
        t1 = Task(task_id=1, name="T1", period=5.0, deadline=5.0, wcet=2.0, priority=3)
        t2 = Task(task_id=2, name="T2", period=7.0, deadline=7.0, wcet=2.0, priority=2)
        t3 = Task(task_id=3, name="T3", period=15.0, deadline=15.0, wcet=4.0, priority=1)

        ts = TaskSet([t1, t2, t3])
        analyzer = RTAAnalyzer()
        res = analyzer.analyze(ts)

        self.assertTrue(res["is_schedulable"])
        wcrt_map = {r["task_id"]: r["wcrt_ms"] for r in res["per_task_results"]}
        # R3 = 4 + ceil(14/5)*2 + ceil(14/7)*2 = 4 + 6 + 4 = 14 <= 15 (schedulable)
        self.assertEqual(wcrt_map[1], 2.0)
        self.assertEqual(wcrt_map[2], 4.0)
        self.assertEqual(wcrt_map[3], 14.0)

    def test_discrete_event_simulator_execution(self):
        """Verify discrete-event simulator runs without errors and calculates metrics."""
        t1 = Task(task_id=1, name="T1", period=20.0, deadline=20.0, wcet=4.0, priority=2)
        t2 = Task(task_id=2, name="T2", period=50.0, deadline=50.0, wcet=5.0, priority=1)

        ts = TaskSet([t1, t2])
        sim = DiscreteEventSimulator(measure_overhead=True)
        res = sim.run(ts, duration_ms=200.0)

        self.assertTrue(res["is_schedulable"])
        self.assertEqual(res["deadline_misses"], 0)
        self.assertGreater(res["completed_jobs"], 0)
        self.assertGreater(res["achieved_cpu_utilization"], 0.0)
        self.assertIn("mean_scheduling_overhead_us", res)

    def test_topsis_adaptive_priority(self):
        """Verify TOPSIS + Entropy Weight adaptive ranking combining criticality and urgency."""
        t1 = Task(1, "CriticalUrgent", period=50.0, deadline=50.0, wcet=45.0, criticality="ASIL-D")  # tight slack: 5ms
        t2 = Task(2, "NonCriticalRelaxed", period=100.0, deadline=100.0, wcet=10.0, criticality="QM") # slack: 90ms
        t3 = Task(3, "CriticalRelaxed", period=100.0, deadline=100.0, wcet=10.0, criticality="ASIL-D")# slack: 90ms

        assigner = PriorityAssigner(p_min=1, p_max=10, base_sorting="TOPSIS")
        scores = assigner.compute_topsis_scores([t1, t2, t3])
        self.assertEqual(len(scores), 3)
        # t1 (ASIL-D with tightest slack) must have highest TOPSIS closeness score
        self.assertGreater(scores[1], scores[2])
        # t3 (ASIL-D) must rank higher than t2 (QM) with same slack
        self.assertGreater(scores[3], scores[2])

        ts = TaskSet([t1, t2, t3])
        assigned = assigner.assign_priorities(ts, "Uniform")
        self.assertEqual(assigned[0].priority, 10)  # t1 gets top priority

    def test_synthetic_task_generator_uunifast(self):
        """Verify UUniFast generates task sets that accurately sum to target utilization."""
        gen = SyntheticTaskGenerator(seed=123)
        target_u = 0.75
        utils = gen.uunifast(10, target_u)
        self.assertEqual(len(utils), 10)
        self.assertAlmostEqual(sum(utils), target_u, places=5)

        # Verify perception task set generation
        ts = gen.generate_perception_task_set(target_utilization=0.65)
        self.assertEqual(len(ts), 10)
        self.assertAlmostEqual(ts.total_utilization, 0.65, delta=0.08)


if __name__ == "__main__":
    unittest.main()
