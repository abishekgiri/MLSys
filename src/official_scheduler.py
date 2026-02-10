"""
Baseline scheduler for the official MLSys 2026 format.
Generates one-op subgraphs with a feasible granularity and computes latency.
"""

import math
from pathlib import Path
from typing import List, Set, Tuple, Optional
from src.official_types import (
    OfficialProblem,
    OfficialGranularity,
    OfficialSubgraph,
    OfficialSolution,
)
from src.official_evaluator import (
    topological_sort,
    working_set_bytes,
    compute_subgraph_latency,
    compute_output_dims,
    compute_k_dimension,
    build_tensor_producers,
)
from src.official_evaluator import _boundary_bytes_for_subgraph


class OfficialScheduler:
    def __init__(
        self,
        problem: OfficialProblem,
        retain_budget: float = 0.0,
        problem_path: Optional[str] = None,
        debug_fuse: bool = False,
        debug_fuse_file: str = 'schedules/debug_fuse_2026-17.txt',
        fuse_accept_slack: float = 0.0,
        connectivity: str = 'strict',
    ):
        self.problem = problem
        self.retain_budget = max(0.0, min(retain_budget, 0.9))
        self.retain_cap = int(self.problem.fast_memory_capacity * self.retain_budget)
        self.compute_cap = self.problem.fast_memory_capacity - self.retain_cap
        self.problem_path = problem_path or ''
        self.debug_fuse = debug_fuse
        self.debug_fuse_file = debug_fuse_file
        self.fuse_accept_slack = max(0.0, fuse_accept_slack)
        self.connectivity = connectivity

    def _granularity_candidates(self, op_idx: int) -> List[OfficialGranularity]:
        op = self.problem.ops[op_idx]
        out_w, out_h = compute_output_dims(self.problem, op_idx)
        native = self.problem.native_granularity
        w = min(native.width, out_w)
        h = min(native.height, out_h)

        width_candidates = []
        height_candidates = []
        while True:
            if w not in width_candidates:
                width_candidates.append(w)
            if w == 1:
                break
            w = max(1, math.ceil(w / 2))
        while True:
            if h not in height_candidates:
                height_candidates.append(h)
            if h == 1:
                break
            h = max(1, math.ceil(h / 2))

        k_candidates = [1]
        if op.op_type == "MatMul":
            K = compute_k_dimension(self.problem, op_idx)
            k_candidates = []
            k_val = K
            while True:
                if k_val not in k_candidates:
                    k_candidates.append(k_val)
                if k_val == 1:
                    break
                k_val = max(1, math.ceil(k_val / 2))

        candidates: List[OfficialGranularity] = []
        for w_val in width_candidates:
            for h_val in height_candidates:
                for k_val in k_candidates:
                    candidates.append(OfficialGranularity(w_val, h_val, k_val))
        return candidates

    def _choose_granularity(self, op_idx: int) -> OfficialGranularity:
        best_gran = None
        best_latency = None
        for gran in self._granularity_candidates(op_idx):
            if working_set_bytes(self.problem, op_idx, gran) > self.compute_cap:
                continue
            latency = compute_subgraph_latency(
                self.problem,
                [op_idx],
                gran,
                [],
                None,
            )
            if best_latency is None or latency < best_latency:
                best_latency = latency
                best_gran = gran

        if best_gran is None:
            raise ValueError(f"Op {op_idx} cannot fit in fast memory even at 1x1")

        return best_gran

    def schedule(self) -> OfficialSolution:
        order = topological_sort(self.problem)
        subgraphs: List[OfficialSubgraph] = []

        for op_idx in order:
            gran = self._choose_granularity(op_idx)
            tensors_to_retain: List[int] = []
            traversal_order = None
            latency = compute_subgraph_latency(
                self.problem,
                [op_idx],
                gran,
                tensors_to_retain,
                traversal_order,
            )
            subgraphs.append(
                OfficialSubgraph(
                    ops=[op_idx],
                    tensors_to_retain=tensors_to_retain,
                    granularity=gran,
                    traversal_order=traversal_order,
                    subgraph_latency=latency,
                )
            )

        return OfficialSolution(subgraphs=subgraphs)


class OfficialFusionScheduler(OfficialScheduler):
    """
    Greedy fusion scheduler that groups compatible Pointwise ops into subgraphs.
    Falls back to single-op subgraphs for MatMul or incompatible ops.
    """

    def __init__(
        self,
        problem: OfficialProblem,
        retain_budget: float = 0.0,
        problem_path: Optional[str] = None,
        debug_fuse: bool = False,
        debug_fuse_file: str = 'schedules/debug_fuse_2026-17.txt',
        fuse_accept_slack: float = 0.0,
        connectivity: str = 'strict',
    ):
        super().__init__(
            problem,
            retain_budget=retain_budget,
            problem_path=problem_path,
            debug_fuse=debug_fuse,
            debug_fuse_file=debug_fuse_file,
            fuse_accept_slack=fuse_accept_slack,
            connectivity=connectivity,
        )
        self._producers = build_tensor_producers(problem)

    def _connected_to_subgraph(self, op_idx: int, subgraph_ops: List[int]) -> bool:
        op = self.problem.ops[op_idx]
        op_set = set(subgraph_ops)
        for inp in op.inputs:
            producer = self._producers.get(inp)
            if producer in op_set:
                return True
        return False

    def _deps_satisfied(self, op_idx: int, scheduled: Set[int], subgraph_ops: List[int]) -> bool:
        op = self.problem.ops[op_idx]
        subgraph_set = set(subgraph_ops)
        for inp in op.inputs:
            producer = self._producers.get(inp)
            if producer is None:
                continue
            if producer not in scheduled and producer not in subgraph_set:
                return False
        return True

    def _is_connected(
        self,
        op_idx: int,
        produced_in_sg: Set[int],
        consumed_in_sg: Set[int],
        boundary_in: Set[int],
        boundary_out: Set[int],
        connectivity: str,
    ) -> bool:
        op = self.problem.ops[op_idx]
        if any(inp in produced_in_sg for inp in op.inputs):
            return True
        if any(out in consumed_in_sg for out in op.outputs):
            return True
        if connectivity == 'strict':
            return False
        boundary = boundary_in | boundary_out
        if any(inp in boundary for inp in op.inputs):
            return True
        if any(out in boundary for out in op.outputs):
            return True
        return False

    def _op_working_set(self, op_idx: int, gran: OfficialGranularity) -> int:
        op = self.problem.ops[op_idx]
        w = gran.width
        h = gran.height
        if op.op_type == "Pointwise":
            return (len(op.inputs) + len(op.outputs)) * w * h
        if op.op_type == "MatMul":
            K = compute_k_dimension(self.problem, op_idx)
            k_eff = min(gran.depth, K)
            return (h * k_eff) + (k_eff * w) + (w * h)
        raise ValueError(f"Unknown op type: {op.op_type}")

    def _choose_granularity_for_subgraph(self, subgraph_ops: List[int]) -> OfficialGranularity:
        out_ws = []
        out_hs = []
        for op_idx in subgraph_ops:
            w, h = compute_output_dims(self.problem, op_idx)
            out_ws.append(w)
            out_hs.append(h)
        min_w = min(out_ws)
        min_h = min(out_hs)

        native = self.problem.native_granularity
        w = min(native.width, min_w)
        h = min(native.height, min_h)

        widths = []
        heights = []
        while True:
            if w not in widths:
                widths.append(w)
            if w == 1:
                break
            w = max(1, math.ceil(w / 2))
        while True:
            if h not in heights:
                heights.append(h)
            if h == 1:
                break
            h = max(1, math.ceil(h / 2))

        matmuls = [op_idx for op_idx in subgraph_ops if self.problem.ops[op_idx].op_type == "MatMul"]
        if matmuls:
            max_k = max(compute_k_dimension(self.problem, op_idx) for op_idx in matmuls)
            k_candidates = []
            k_val = max_k
            while True:
                if k_val not in k_candidates:
                    k_candidates.append(k_val)
                if k_val == 1:
                    break
                k_val = max(1, math.ceil(k_val / 2))
        else:
            k_candidates = [1]

        best_gran = None
        best_latency = None
        for w_val in widths:
            for h_val in heights:
                for k_val in k_candidates:
                    gran = OfficialGranularity(width=w_val, height=h_val, depth=k_val)
                    fits = True
                    for op_idx in subgraph_ops:
                        if self._op_working_set(op_idx, gran) > self.compute_cap:
                            fits = False
                            break
                    if not fits:
                        continue
                    latency = compute_subgraph_latency(
                        self.problem,
                        subgraph_ops,
                        gran,
                        [],
                        None,
                    )
                    if best_latency is None or latency < best_latency:
                        best_latency = latency
                        best_gran = gran

        if best_gran is None:
            raise ValueError("Subgraph cannot fit in fast memory")
        return best_gran

    def _tensor_size(self, tensor_id: int) -> int:
        tensor = self.problem.tensors[tensor_id]
        return tensor.width * tensor.height

    def _subgraph_inputs_outputs(self, subgraph_ops: List[int]) -> Tuple[Set[int], Set[int]]:
        inputs = set()
        outputs = set()
        for op_idx in subgraph_ops:
            op = self.problem.ops[op_idx]
            inputs.update(op.inputs)
            outputs.update(op.outputs)
        return inputs, outputs

    def _subgraph_working_set(self, subgraph_ops: List[int], gran: OfficialGranularity) -> int:
        max_ws = 0
        for op_idx in subgraph_ops:
            max_ws = max(max_ws, self._op_working_set(op_idx, gran))
        return max_ws

    def _assign_tensors_to_retain(self, subgraphs: List[OfficialSubgraph]) -> None:
        # Build usage lists per tensor.
        uses = {}
        subgraph_inputs = []
        subgraph_outputs = []
        for idx, sg in enumerate(subgraphs):
            inputs, outputs = self._subgraph_inputs_outputs(sg.ops)
            subgraph_inputs.append(inputs)
            subgraph_outputs.append(outputs)
            for t in inputs:
                uses.setdefault(t, []).append(idx)

        use_pos = {t: 0 for t in uses}

        for idx, sg in enumerate(subgraphs):
            # Determine available capacity for retention based on next subgraph working set.
            available = self.retain_cap

            candidates = []
            inputs = subgraph_inputs[idx]
            outputs = subgraph_outputs[idx]
            # Consider retaining outputs and inputs that will be used again.
            for t in outputs.union(inputs):
                if t not in uses:
                    continue
                pos = use_pos.get(t, 0)
                next_use = None
                if pos < len(uses[t]) and uses[t][pos] == idx:
                    if pos + 1 < len(uses[t]):
                        next_use = uses[t][pos + 1]
                elif pos < len(uses[t]):
                    next_use = uses[t][pos]

                if next_use is None:
                    continue

                distance = max(1, next_use - idx)
                size = self._tensor_size(t)
                benefit = size / distance
                candidates.append((benefit, size, t))

            candidates.sort(reverse=True)
            retained = []
            used = 0
            for _, size, t in candidates:
                if used + size <= available:
                    retained.append(t)
                    used += size

            sg.tensors_to_retain = retained

            # Advance use_pos for inputs consumed at this subgraph.
            for t in inputs:
                if t in uses and use_pos[t] < len(uses[t]) and uses[t][use_pos[t]] == idx:
                    use_pos[t] += 1

        resident: Set[int] = set()
        for sg in subgraphs:
            sg.subgraph_latency = compute_subgraph_latency(
                self.problem,
                sg.ops,
                sg.granularity,
                sg.tensors_to_retain,
                sg.traversal_order,
                resident_tensors=resident,
            )
            resident = set(sg.tensors_to_retain)

    def _build_subgraphs(self, connectivity_mode: str, debug_suffix: Optional[str]) -> Tuple[List[OfficialSubgraph], int]:
        order = topological_sort(self.problem)
        order_index = {op_idx: i for i, op_idx in enumerate(order)}
        n_ops = len(order)
        subgraphs: List[OfficialSubgraph] = []
        scheduled: Set[int] = set()
        split_count = 0

        debug_enabled = self.debug_fuse and ('mlsys-2026-17' in (self.problem_path or ''))
        dbg = None
        if debug_enabled:
            dbg_path = Path(self.debug_fuse_file)
            if debug_suffix:
                dbg_path = dbg_path.with_name(f"{dbg_path.stem}_{debug_suffix}{dbg_path.suffix}")
            dbg_path.parent.mkdir(parents=True, exist_ok=True)
            dbg = dbg_path.open('w')

        def log(msg: str) -> None:
            if dbg:
                dbg.write(msg + '\n')

        preds: List[Set[int]] = [set() for _ in range(n_ops)]
        succs: List[Set[int]] = [set() for _ in range(n_ops)]
        for op_idx, op in enumerate(self.problem.ops):
            for inp in op.inputs:
                producer = self._producers.get(inp)
                if producer is None:
                    continue
                preds[op_idx].add(producer)
                succs[producer].add(op_idx)

        def is_connected_subgraph(ops: List[int]) -> bool:
            if len(ops) <= 1:
                return True
            op_set = set(ops)
            adj = {op: set() for op in ops}
            for u in ops:
                for v in succs[u]:
                    if v in op_set:
                        adj[u].add(v)
                        adj[v].add(u)
            start = ops[0]
            stack = [start]
            seen = {start}
            while stack:
                cur = stack.pop()
                for nxt in adj[cur]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            return len(seen) == len(op_set)

        def split_connected_components(ops: List[int]) -> List[List[int]]:
            if len(ops) <= 1:
                return [ops]
            op_set = set(ops)
            adj = {op: set() for op in ops}
            for u in ops:
                for v in succs[u]:
                    if v in op_set:
                        adj[u].add(v)
                        adj[v].add(u)
            components = []
            unseen = set(ops)
            while unseen:
                start = min(unseen, key=lambda o: order_index[o])
                stack = [start]
                seen = {start}
                unseen.remove(start)
                while stack:
                    cur = stack.pop()
                    for nxt in adj[cur]:
                        if nxt in unseen:
                            unseen.remove(nxt)
                            seen.add(nxt)
                            stack.append(nxt)
                comp = sorted(seen, key=lambda o: order_index[o])
                components.append(comp)
            return components

        while len(scheduled) < n_ops:
            local_unsat = {}
            ready: Set[int] = set()
            for op_idx in range(n_ops):
                if op_idx in scheduled:
                    continue
                cnt = sum(1 for p in preds[op_idx] if p not in scheduled)
                local_unsat[op_idx] = cnt
                if cnt == 0:
                    ready.add(op_idx)

            if not ready:
                raise ValueError("No ready ops to schedule; cycle?")

            seed = min(ready, key=lambda o: order_index[o])
            ready.remove(seed)
            current_ops = [seed]
            in_sg: Set[int] = {seed}
            try:
                gran = self._choose_granularity_for_subgraph(current_ops)
            except ValueError:
                gran = self._choose_granularity(seed)
            current_latency = compute_subgraph_latency(
                self.problem,
                current_ops,
                gran,
                [],
                None,
            )

            produced_in_sg: Set[int] = set(self.problem.ops[seed].outputs)
            consumed_in_sg: Set[int] = set(self.problem.ops[seed].inputs)
            boundary_in: Set[int] = set(self.problem.ops[seed].inputs)
            boundary_out: Set[int] = set(self.problem.ops[seed].outputs)

            for succ in succs[seed]:
                if succ in scheduled or succ in in_sg:
                    continue
                local_unsat[succ] -= 1
                if local_unsat[succ] == 0:
                    ready.add(succ)

            while True:
                candidates = [
                    u for u in ready
                    if self._is_connected(
                        u,
                        produced_in_sg,
                        consumed_in_sg,
                        boundary_in,
                        boundary_out,
                        connectivity_mode,
                    )
                ]
                if not candidates:
                    break
                candidates.sort(key=lambda o: order_index[o])

                accepted = False
                for cand in candidates:
                    trial_ops = current_ops + [cand]
                    try:
                        trial_gran = self._choose_granularity_for_subgraph(trial_ops)
                    except ValueError:
                        log(f"TRY merge sg@{seed} + op{cand}: RESULT=REJECT reason=WS_OVER_CAP")
                        continue

                    trial_latency = compute_subgraph_latency(
                        self.problem,
                        trial_ops,
                        trial_gran,
                        [],
                        None,
                    )

                    cand_gran = self._choose_granularity_for_subgraph([cand])
                    cand_latency = compute_subgraph_latency(
                        self.problem,
                        [cand],
                        cand_gran,
                        [],
                        None,
                    )

                    cur_in, cur_out = _boundary_bytes_for_subgraph(
                        self.problem,
                        current_ops,
                        gran,
                        resident=set(),
                        retained=set(),
                    )
                    cand_in, cand_out = _boundary_bytes_for_subgraph(
                        self.problem,
                        [cand],
                        cand_gran,
                        resident=set(),
                        retained=set(),
                    )
                    trial_in, trial_out = _boundary_bytes_for_subgraph(
                        self.problem,
                        trial_ops,
                        trial_gran,
                        resident=set(),
                        retained=set(),
                    )
                    saved_load = (cur_in + cand_in) - trial_in
                    saved_store = (cur_out + cand_out) - trial_out
                    ws_bytes = self._subgraph_working_set(trial_ops, trial_gran)

                    separate_latency = current_latency + cand_latency
                    accept_limit = separate_latency * (1.0 + self.fuse_accept_slack)
                    if trial_latency > accept_limit:
                        log(
                            f"TRY merge sg@{seed} + op{cand}: compute_cap={self.compute_cap}B "
                            f"ws={ws_bytes}B tile=({trial_gran.width},{trial_gran.height},{trial_gran.depth}) "
                            f"separate_lat={separate_latency:.2f} fused_lat={trial_latency:.2f} "
                            f"boundary_saved: load={saved_load}B store={saved_store}B RESULT=REJECT reason=FUSED_WORSE_LAT"
                        )
                        continue

                    accept_reason = "OK"
                    if self.fuse_accept_slack > 0.0 and trial_latency > separate_latency:
                        accept_reason = f"SLACK({self.fuse_accept_slack:.2f})"

                    log(
                        f"TRY merge sg@{seed} + op{cand}: compute_cap={self.compute_cap}B "
                        f"ws={ws_bytes}B tile=({trial_gran.width},{trial_gran.height},{trial_gran.depth}) "
                        f"separate_lat={separate_latency:.2f} fused_lat={trial_latency:.2f} "
                        f"boundary_saved: load={saved_load}B store={saved_store}B RESULT=ACCEPT reason={accept_reason}"
                    )

                    current_ops = trial_ops
                    in_sg.add(cand)
                    ready.remove(cand)
                    gran = trial_gran
                    current_latency = trial_latency
                    produced_in_sg.update(self.problem.ops[cand].outputs)
                    consumed_in_sg.update(self.problem.ops[cand].inputs)
                    boundary_in.update(self.problem.ops[cand].inputs)
                    boundary_out.update(self.problem.ops[cand].outputs)

                    for succ in succs[cand]:
                        if succ in scheduled or succ in in_sg:
                            continue
                        local_unsat[succ] -= 1
                        if local_unsat[succ] == 0:
                            ready.add(succ)

                    accepted = True
                    break

                if not accepted:
                    break

            tensors_to_retain = []
            traversal_order = None
            if is_connected_subgraph(current_ops):
                subgraphs.append(
                    OfficialSubgraph(
                        ops=current_ops,
                        tensors_to_retain=tensors_to_retain,
                        granularity=gran,
                        traversal_order=traversal_order,
                        subgraph_latency=current_latency,
                    )
                )
            else:
                split_count += 1
                for comp in split_connected_components(current_ops):
                    comp_gran = self._choose_granularity_for_subgraph(comp)
                    comp_latency = compute_subgraph_latency(
                        self.problem,
                        comp,
                        comp_gran,
                        [],
                        None,
                    )
                    subgraphs.append(
                        OfficialSubgraph(
                            ops=comp,
                            tensors_to_retain=[],
                            granularity=comp_gran,
                            traversal_order=None,
                            subgraph_latency=comp_latency,
                        )
                    )
            scheduled.update(current_ops)

        self._assign_tensors_to_retain(subgraphs)

        if dbg:
            dbg.close()

        return subgraphs, split_count

    def schedule(self) -> OfficialSolution:
        if self.connectivity == 'auto':
            strict_subgraphs, strict_splits = self._build_subgraphs('strict', debug_suffix='strict')
            loose_subgraphs, loose_splits = self._build_subgraphs('loose', debug_suffix='loose')
            strict_latency = sum(sg.subgraph_latency for sg in strict_subgraphs)
            loose_latency = sum(sg.subgraph_latency for sg in loose_subgraphs)
            if loose_latency <= strict_latency:
                return OfficialSolution(
                    subgraphs=loose_subgraphs,
                    connectivity_requested=self.connectivity,
                    connectivity_chosen='loose',
                    latency_strict=strict_latency,
                    latency_loose=loose_latency,
                    loose_components_split=loose_splits,
                )
            return OfficialSolution(
                subgraphs=strict_subgraphs,
                connectivity_requested=self.connectivity,
                connectivity_chosen='strict',
                latency_strict=strict_latency,
                latency_loose=loose_latency,
                loose_components_split=loose_splits,
            )

        subgraphs, _ = self._build_subgraphs(self.connectivity, debug_suffix=None)
        return OfficialSolution(
            subgraphs=subgraphs,
            connectivity_requested=self.connectivity,
            connectivity_chosen=self.connectivity,
            latency_strict=None,
            latency_loose=None,
            loose_components_split=0,
        )
