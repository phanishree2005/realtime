"""
Discrete-Event Real-Time Scheduling Simulator for Fixed-Priority Preemptive Scheduling (FPPS).
Simulates job releases, preemptions, CPU execution, context switching,
scheduling decision overhead, and deadline violations over time.
"""

import time
import heapq
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np

from .task_model import Task, TaskSet, Job


@dataclass(order=True)
class Event:
    """Simulation discrete event."""
    event_time: float
    event_type: str = field(compare=False)
    job: Optional[Job] = field(default=None, compare=False)


class DiscreteEventSimulator:
    """
    Simulates Fixed-Priority Preemptive Scheduling (FPPS) with microsecond
    scheduling overhead measurement, preemption tracking, and deadline miss detection.
    """

    def __init__(
        self,
        context_switch_penalty_ms: float = 0.005,  # 5 microseconds context switch penalty
        measure_overhead: bool = True
    ):
        self.context_switch_penalty_ms = context_switch_penalty_ms
        self.measure_overhead = measure_overhead

    def run(
        self,
        task_set: TaskSet,
        duration_ms: float = 1000.0,
        use_wcet: bool = True,
        jitter_factor: float = 0.0
    ) -> Dict[str, Any]:
        """
        Runs discrete-event simulation over duration_ms.

        Args:
            task_set: The TaskSet to simulate with assigned priorities.
            duration_ms: Total simulation horizon in milliseconds.
            use_wcet: If True, jobs execute for their exact WCET.
                      If False, jobs execute for sampled time in [BCET, WCET].
            jitter_factor: Extra execution jitter from scene clutter (0.0 to 0.3).
        """
        if len(task_set) == 0:
            return {"error": "Empty task set"}

        # 1. Generate all job releases within the horizon
        all_jobs: List[Job] = []
        event_queue: List[Event] = []
        job_id_counter = 0

        for task in task_set.tasks:
            t_rel = 0.0
            while t_rel < duration_ms:
                # Sample execution time
                if use_wcet:
                    exec_time = task.wcet * (1.0 + jitter_factor)
                else:
                    base = np.random.uniform(task.bcet, task.wcet)
                    exec_time = base * (1.0 + jitter_factor)

                job = Job(
                    job_id=job_id_counter,
                    task_id=task.task_id,
                    task_name=task.name,
                    priority=task.priority,
                    release_time=t_rel,
                    absolute_deadline=t_rel + task.deadline,
                    execution_time=exec_time,
                    remaining_time=exec_time
                )
                all_jobs.append(job)
                heapq.heappush(event_queue, Event(event_time=t_rel, event_type="RELEASE", job=job))
                job_id_counter += 1
                t_rel += task.period

        # Ready queue for released jobs:
        # Priority queue ordered by: (-priority, release_time, job_id)
        ready_queue: List[tuple] = []

        current_time = 0.0
        current_running_job: Optional[Job] = None
        current_job_start_or_resume_time: float = 0.0

        total_busy_time = 0.0
        preemption_count = 0
        context_switch_count = 0
        overhead_samples_ns: List[float] = []
        execution_trace: List[Dict[str, Any]] = []
        preemption_events: List[Dict[str, Any]] = []

        while event_queue:
            event = heapq.heappop(event_queue)
            now = event.event_time

            if now > duration_ms:
                break

            # Advance execution of the currently running job up to `now`
            if current_running_job is not None:
                elapsed = now - current_job_start_or_resume_time
                if elapsed > 0:
                    current_running_job.remaining_time -= elapsed
                    total_busy_time += elapsed
                    execution_trace.append({
                        "task_id": current_running_job.task_id,
                        "task_name": current_running_job.task_name,
                        "job_id": current_running_job.job_id,
                        "start": current_job_start_or_resume_time,
                        "finish": now,
                        "duration": elapsed,
                        "priority": current_running_job.priority
                    })

            current_time = now

            # Measure scheduling dispatch overhead
            t_decision_start = time.perf_counter_ns() if self.measure_overhead else 0

            if event.event_type == "RELEASE":
                released_job = event.job
                # Add to ready queue: (-priority, release_time, job_id, job)
                heapq.heappush(
                    ready_queue,
                    (-released_job.priority, released_job.release_time, released_job.job_id, released_job)
                )

                # Check preemption if currently running job has lower priority
                if current_running_job is not None and released_job.priority > current_running_job.priority:
                    # Preemption occurs!
                    preemption_count += 1
                    current_running_job.preemption_count += 1
                    preemption_events.append({
                        "time": now,
                        "preempted_task": current_running_job.task_name,
                        "preempting_task": released_job.task_name
                    })
                    # Put preempted job back into ready queue
                    heapq.heappush(
                        ready_queue,
                        (-current_running_job.priority, current_running_job.release_time, current_running_job.job_id, current_running_job)
                    )
                    current_running_job = None
                    context_switch_count += 1

            elif event.event_type == "COMPLETION":
                completed_job = event.job
                # Only process if it matches the current running job
                if current_running_job is not None and current_running_job.job_id == completed_job.job_id:
                    completed_job.finish_time = current_time
                    completed_job.is_completed = True
                    completed_job.remaining_time = 0.0
                    if completed_job.finish_time > completed_job.absolute_deadline + 1e-9:
                        completed_job.missed_deadline = True
                    current_running_job = None

            # If CPU is free, dispatch highest-priority job from ready queue
            if current_running_job is None and ready_queue:
                _, _, _, next_job = heapq.heappop(ready_queue)
                current_running_job = next_job

                if current_running_job.start_time is None:
                    current_running_job.start_time = current_time

                # Apply small context switch penalty if configured
                cs_cost = self.context_switch_penalty_ms
                current_job_start_or_resume_time = current_time + cs_cost
                total_busy_time += cs_cost
                context_switch_count += 1

                # Schedule completion event
                finish_time = current_job_start_or_resume_time + current_running_job.remaining_time
                heapq.heappush(
                    event_queue,
                    Event(event_time=finish_time, event_type="COMPLETION", job=current_running_job)
                )

            if self.measure_overhead:
                t_decision_end = time.perf_counter_ns()
                overhead_samples_ns.append(float(t_decision_end - t_decision_start))

        # Check all jobs for deadline misses
        deadline_misses = 0
        completed_count = 0
        response_times: List[float] = []

        for job in all_jobs:
            if job.release_time + job.execution_time > duration_ms:
                # Job arrived too close to end of simulation horizon, skip unfair edge penalty
                continue

            if job.is_completed:
                completed_count += 1
                if job.response_time is not None:
                    response_times.append(job.response_time)
                if job.missed_deadline:
                    deadline_misses += 1
            else:
                # Did not complete within simulation horizon -> missed deadline
                deadline_misses += 1

        eval_jobs_count = completed_count + (len(all_jobs) - completed_count)
        dmr = (deadline_misses / eval_jobs_count * 100.0) if eval_jobs_count > 0 else 0.0
        achieved_utilization = min(1.0, total_busy_time / duration_ms) if duration_ms > 0 else 0.0

        mean_overhead_us = (np.mean(overhead_samples_ns) / 1000.0) if overhead_samples_ns else 0.0
        max_overhead_us = (np.max(overhead_samples_ns) / 1000.0) if overhead_samples_ns else 0.0

        return {
            "total_released_jobs": len(all_jobs),
            "completed_jobs": completed_count,
            "deadline_misses": deadline_misses,
            "deadline_miss_ratio_pct": dmr,
            "is_schedulable": (deadline_misses == 0),
            "achieved_cpu_utilization": achieved_utilization,
            "nominal_utilization": task_set.total_utilization,
            "preemption_count": preemption_count,
            "context_switch_count": context_switch_count,
            "mean_response_time_ms": float(np.mean(response_times)) if response_times else 0.0,
            "max_response_time_ms": float(np.max(response_times)) if response_times else 0.0,
            "mean_scheduling_overhead_us": mean_overhead_us,
            "max_scheduling_overhead_us": max_overhead_us,
            "execution_trace": execution_trace,
            "preemption_events": preemption_events
        }
