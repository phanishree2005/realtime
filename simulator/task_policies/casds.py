#!/usr/bin/env python
"""
STOMP Scheduling Policy Plug-in: CASDS
Context-Aware-Slack-Driven Dynamic Scheduling for ACPS
Reference: IEEE Transactions on Industrial Informatics
"""

import math
import numpy as np
from datetime import datetime, timedelta
from stomp import BaseSchedulingPolicy


class SchedulingPolicy(BaseSchedulingPolicy):

    def init(self, servers, stomp_stats, stomp_params):
        self.servers = servers
        self.n_servers = len(servers)
        self.stomp_stats = stomp_stats
        self.stomp_params = stomp_params
        self.stats = {}
        self.stats['Task Issue Posn'] = np.zeros(12, dtype=int)
        self.ta_time = timedelta(microseconds=0)
        self.to_time = timedelta(microseconds=0)

    def calculate_cas_slack(self, task, sim_time, stomp_obj):
        """
        Computes Context-Aware Slack (CAS) for a task in STOMP:
        Trem = arrival_time + deadline - sim_time
        PLCT = estimated remaining execution time on available servers
        CAS = Trem - PLCT
        """
        # Historical & absolute deadline
        trem = (task.arrival_time + task.deadline) - sim_time

        # Predictive Least Completion Time (PLCT)
        min_exec = task.min_time if hasattr(task, 'min_time') and task.min_time is not None else 10.0
        if min_exec is None:
            min_exec = min(task.mean_service_time_dict.values()) if task.mean_service_time_dict else 10.0

        # Estimate server queue delay
        min_server_avail = float('inf')
        for s in self.servers:
            if s.type in task.mean_service_time_dict:
                avail = s.curr_job_end_time - sim_time if s.busy else 0
                if avail < min_server_avail:
                    min_server_avail = avail

        if min_server_avail == float('inf'):
            min_server_avail = 0.0

        plct = min_exec + min_server_avail
        cas = trem - plct
        return cas

    def rank_and_prioritize_tasks(self, tasks, sim_time, stomp_obj):
        """
        Calculates adaptive priority Pa_i based on TOPSIS and Entropy Weight Method
        balancing Criticality (C) and Urgency (-CAS).
        """
        if not tasks:
            return tasks

        cas_list = []
        for t in tasks:
            cas = self.calculate_cas_slack(t, sim_time, stomp_obj)
            cas_list.append(cas)

        N = len(tasks)
        if N <= 1:
            for t in tasks:
                t.dynamic_prio = 1.0
            return tasks

        # Attribute matrix: col 0 = Criticality, col 1 = -CAS (higher is more urgent)
        raw_mat = np.zeros((N, 2), dtype=float)
        for i, t in enumerate(tasks):
            raw_mat[i, 0] = float(t.priority)
            raw_mat[i, 1] = -float(cas_list[i])

        # Min-max normalization
        norm_mat = np.zeros_like(raw_mat)
        for m in range(2):
            c_min = np.min(raw_mat[:, m])
            c_max = np.max(raw_mat[:, m])
            diff = c_max - c_min
            if diff > 1e-9:
                norm_mat[:, m] = (raw_mat[:, m] - c_min) / diff
            else:
                norm_mat[:, m] = 1.0

        # Entropy Weight Method
        eps = 1e-12
        weights = np.array([0.5, 0.5])
        col_sums = np.sum(norm_mat, axis=0)
        entropies = np.zeros(2)
        for m in range(2):
            if col_sums[m] > 1e-9:
                p = norm_mat[:, m] / col_sums[m]
                p = np.where(p > eps, p, eps)
                entropies[m] = -np.sum(p * np.log(p)) / np.log(N)
            else:
                entropies[m] = 1.0

        d = 1.0 - entropies
        if np.sum(d) > 1e-9:
            weights = d / np.sum(d)

        wt_mat = norm_mat * weights
        z_plus = np.max(wt_mat, axis=0)
        z_minus = np.min(wt_mat, axis=0)

        for i, t in enumerate(tasks):
            d_plus = math.sqrt(np.sum((wt_mat[i] - z_plus) ** 2))
            d_minus = math.sqrt(np.sum((wt_mat[i] - z_minus) ** 2))
            tot = d_plus + d_minus
            sim = d_minus / tot if tot > 1e-9 else 0.5
            # Negative CAS deferral: if deadline cannot be met under any schedule, lower priority
            if cas_list[i] < 0:
                sim -= 10.0  # Deferred behind healthy tasks
            t.dynamic_prio = sim

        tasks.sort(key=lambda t: t.dynamic_prio, reverse=True)
        return tasks

    def assign_task_to_server(self, sim_time, tasks, dags_dropped, stomp_obj):
        if not tasks:
            return None

        # Sort tasks dynamically via CASDS TOPSIS
        tasks = self.rank_and_prioritize_tasks(tasks, sim_time, stomp_obj)

        for task in list(tasks):
            # Compute EFT across target servers
            target_servers = []
            for server in self.servers:
                if server.type in task.mean_service_time_dict:
                    mean_time = task.mean_service_time_dict[server.type]
                    rem_time = (server.curr_job_end_time - sim_time) if server.busy else 0
                    target_servers.append(mean_time + rem_time)
                else:
                    target_servers.append(float('inf'))

            if min(target_servers) == float('inf'):
                continue

            best_server_idx = target_servers.index(min(target_servers))
            server = self.servers[best_server_idx]

            # If target server is free right now, allocate!
            if not server.busy:
                tasks.remove(task)
                task.ptoks_used = task.power_dict.get(server.type, 0)
                server.assign_task(sim_time, task)
                return server
            else:
                task.possible_server_idx = best_server_idx

        return None

    def remove_task_from_server(self, sim_time, server):
        pass

    def output_final_stats(self, sim_time):
        pass
