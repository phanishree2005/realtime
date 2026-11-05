"""
Discrete-Event Simulation Engine for Dynamic Task Scheduling on ACPS.
Coordinates the simulation clock, application arrival events, task completions,
PE failure/recovery events, and scheduler invocations.
"""

import heapq
import time
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any

from .metrics import SchedulingMetrics
from ..models.dag import DAGApplication
from ..models.platform import HeterogeneousPlatform
from ..models.task import TaskStatus
from ..schedulers.base_scheduler import BaseScheduler


class EventType(Enum):
    PE_RECOVERY = 1
    TASK_FINISH = 2
    PE_FAILURE = 3
    APPLICATION_ARRIVAL = 4


class SimulationEvent:
    """Represents a discrete event in the simulation timeline."""
    def __init__(self, time_ms: float, event_type: EventType, data: Any):
        self.time_ms = time_ms
        self.event_type = event_type
        self.data = data

    def __lt__(self, other: "SimulationEvent") -> bool:
        if abs(self.time_ms - other.time_ms) > 1e-9:
            return self.time_ms < other.time_ms
        return self.event_type.value < other.event_type.value


class ACPSSimulator:
    """
    Simulates dynamic execution of DAG applications on a heterogeneous ACPS platform.
    """
    def __init__(
        self,
        platform: HeterogeneousPlatform,
        scheduler: BaseScheduler
    ):
        self.platform = platform
        self.scheduler = scheduler
        self.event_queue: List[SimulationEvent] = []
        self.current_time_ms: float = 0.0

    def run(
        self,
        applications: List[DAGApplication],
        pe_failures: Optional[List[Tuple[float, int, float]]] = None
    ) -> SchedulingMetrics:
        """
        Executes dynamic simulation for the given application trace.
        pe_failures: list of (failure_time, pe_id, recovery_duration)
        """
        start_wall_clock = time.time()
        self.event_queue = []
        self.current_time_ms = 0.0

        # Clone platform and applications to preserve pristine inputs
        sim_platform = self.platform.clone()
        sim_apps = [app.clone() for app in applications]
        app_map: Dict[int, DAGApplication] = {a.app_id: a for a in sim_apps}

        self.scheduler.reset()

        # Enqueue application arrival events
        for app in sim_apps:
            heapq.heappush(
                self.event_queue,
                SimulationEvent(app.arrival_time, EventType.APPLICATION_ARRIVAL, app.app_id)
            )

        # Enqueue PE failure events if present
        if pe_failures:
            for fail_time, pe_id, duration in pe_failures:
                heapq.heappush(
                    self.event_queue,
                    SimulationEvent(fail_time, EventType.PE_FAILURE, (pe_id, duration))
                )

        # Main discrete-event loop
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time_ms = event.time_ms

            if event.event_type == EventType.APPLICATION_ARRIVAL:
                app_id = event.data
                app = app_map[app_id]
                self.scheduler.add_application(app, sim_platform)

            elif event.event_type == EventType.TASK_FINISH:
                app_id, task_id, pe_id, finish_time = event.data
                app = app_map[app_id]
                app.mark_task_finished(task_id, finish_time)

            elif event.event_type == EventType.PE_FAILURE:
                pe_id, duration = event.data
                pe = sim_platform.get_pe(pe_id)
                pe.is_failed = True
                pe.failure_end_time = self.current_time_ms + duration
                # Enqueue recovery event
                heapq.heappush(
                    self.event_queue,
                    SimulationEvent(pe.failure_end_time, EventType.PE_RECOVERY, pe_id)
                )

            elif event.event_type == EventType.PE_RECOVERY:
                pe_id = event.data
                pe = sim_platform.get_pe(pe_id)
                pe.is_failed = False

            # Process all simultaneous events occurring at the exact same timestamp
            while self.event_queue and abs(self.event_queue[0].time_ms - self.current_time_ms) < 1e-9:
                next_ev = heapq.heappop(self.event_queue)
                if next_ev.event_type == EventType.APPLICATION_ARRIVAL:
                    app = app_map[next_ev.data]
                    self.scheduler.add_application(app, sim_platform)
                elif next_ev.event_type == EventType.TASK_FINISH:
                    a_id, t_id, _, f_time = next_ev.data
                    app_map[a_id].mark_task_finished(t_id, f_time)
                elif next_ev.event_type == EventType.PE_FAILURE:
                    p_id, dur = next_ev.data
                    p = sim_platform.get_pe(p_id)
                    p.is_failed = True
                    p.failure_end_time = self.current_time_ms + dur
                    heapq.heappush(self.event_queue, SimulationEvent(p.failure_end_time, EventType.PE_RECOVERY, p_id))
                elif next_ev.event_type == EventType.PE_RECOVERY:
                    sim_platform.get_pe(next_ev.data).is_failed = False

            # Invoke dynamic scheduler to allocate ready tasks to available PEs
            decisions = self.scheduler.schedule(self.current_time_ms, sim_platform)

            # Enqueue task finish events for newly launched tasks
            for a_id, t_id, p_id, st, duration in decisions:
                finish_t = st + duration
                heapq.heappush(
                    self.event_queue,
                    SimulationEvent(finish_t, EventType.TASK_FINISH, (a_id, t_id, p_id, finish_t))
                )

            # Cleanup completed applications
            self.scheduler.clean_completed_applications()

        scheduler_elapsed_s = time.time() - start_wall_clock

        # Calculate final metrics
        metrics = SchedulingMetrics(
            applications=sim_apps,
            platform=sim_platform,
            total_time_ms=self.current_time_ms,
            scheduler_runtime_s=scheduler_elapsed_s
        )

        return metrics
