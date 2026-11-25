"""Bounded CPU/GPU curve pipelines; execution experiment, not model training.

Run from Bezier-Engine with the parent environment:
  ..\\.venv\\Scripts\\python.exe -m bezier_engine.curve_pipeline --output runs/curve_pipeline
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time

import torch
from torch import nn

from . import PROJECT
from .common import configure, environment, write_json
from bezier_sphere.geometry import bernstein, bezier
from bezier_sphere.model import ModelConfig, PortsAndCurve


@dataclass(frozen=True)
class Case:
    approach: str
    width: int
    router_width: int
    roots: int
    depth: int
    samples: int
    # Old saved cases omitted this field and used one hidden router layer.
    hidden_layers: int = 1

    @property
    def nodes(self):
        return self.roots * sum(self.samples ** level for level in range(self.depth))

    @property
    def leaves_per_root(self):
        return self.samples ** self.depth


class Producer(nn.Module):
    def __init__(self, case: Case):
        super().__init__()
        self.route = PortsAndCurve(ModelConfig(width=case.width, router_width=case.router_width,
                                              router_layers=case.hidden_layers))
        # Only used for the synthetic dependent-tree experiment.
        self.child = nn.Sequential(nn.Linear(3, case.width), nn.SiLU())


def evaluate(control, basis, out=None):
    """Same four-term Bernstein sum as geometry.bezier, with cached fixed times."""
    return torch.matmul(basis, control, out=out)


@torch.inference_mode()
def full_frontier(model, inputs, basis, case):
    state = inputs
    for level in range(case.depth):
        points = evaluate(model.route(state), basis)
        if level + 1 < case.depth:
            state = model.child(points.reshape(-1, 3))
    return points.reshape(case.roots, case.leaves_per_root, 3)


@dataclass(frozen=True)
class Schedule:
    mode: str
    chunk: int = 0
    slots: int = 1
    workers: int = 1
    native_threads: int = 1


class Slot:
    def __init__(self, case, chunk):
        max_nodes = min(chunk, case.roots) * case.samples ** (case.depth - 1)
        self.controls = torch.empty((max_nodes, 4, 3), pin_memory=True)
        self.points = torch.empty((max_nodes, case.samples, 3), pin_memory=True)
        self.stream = torch.cuda.Stream()
        self.ready = torch.cuda.Event()
        self.future = None
        self.refs = None
        self.offset = self.count = self.level = self.nodes = 0


class Runner:
    """One invocation at a time. No mutable buffers are shared across slots.

    run() returns reusable runner-owned storage, valid until the next run().
    Call clone() if retaining results across invocations.
    """
    def __init__(self, cpu_model, gpu_model, inputs, case, schedule):
        self.case, self.schedule = case, schedule
        self.cpu_model, self.gpu_model = cpu_model, gpu_model
        self.inputs = inputs
        times = (torch.arange(case.samples, dtype=torch.float32) + 0.5) / case.samples
        self.basis = bernstein(times)
        self.gpu_basis = self.basis.cuda()
        self.output = torch.empty((case.roots, case.leaves_per_root, 3), pin_memory=True)
        self.slots = []
        self.pool = None
        if schedule.mode in ("gpu_chunks", "hybrid"):
            self.slots = [Slot(case, schedule.chunk) for _ in range(schedule.slots)]
            self.pool = ThreadPoolExecutor(max_workers=schedule.workers)
        torch.cuda.synchronize()

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True)
        torch.cuda.synchronize()

    @torch.inference_mode()
    def _finish(self, slot):
        # D2H must complete before any CPU read. The same event also ensures
        # a preceding H2D has finished reading this slot's pinned point buffer.
        slot.ready.synchronize()
        if self.schedule.mode == "hybrid":
            evaluate(slot.controls[:slot.nodes], self.basis, out=slot.points[:slot.nodes])
        terminal = self.schedule.mode == "gpu_chunks" or slot.level + 1 == self.case.depth
        if terminal:
            value = slot.points[:slot.nodes].reshape(slot.count, self.case.leaves_per_root, 3)
            self.output[slot.offset:slot.offset + slot.count].copy_(value)

    @torch.inference_mode()
    def _launch(self, slot, continuation=False):
        with torch.cuda.stream(slot.stream):
            if continuation:
                points = slot.points[:slot.nodes].to("cuda", non_blocking=True)
                state = self.gpu_model.child(points.reshape(-1, 3))
                slot.level += 1
            else:
                state = self.inputs[slot.offset:slot.offset + slot.count].to("cuda", non_blocking=True)
                slot.level = 0
            if self.schedule.mode == "gpu_chunks":
                for level in range(self.case.depth):
                    controls = self.gpu_model.route(state)
                    points = evaluate(controls, self.gpu_basis)
                    if level + 1 < self.case.depth:
                        state = self.gpu_model.child(points.reshape(-1, 3))
                slot.nodes = points.shape[0]
                slot.points[:slot.nodes].copy_(points, non_blocking=True)
            else:
                controls = self.gpu_model.route(state)
                slot.nodes = controls.shape[0]
                slot.controls[:slot.nodes].copy_(controls, non_blocking=True)
            # Keep producer temporaries alive until its D2H completes. Stream
            # allocations use the creating stream; pinned host slots are never
            # overwritten until their event and worker have both completed.
            slot.refs = (state, controls, points if self.schedule.mode == "gpu_chunks" or continuation else None)
            slot.ready.record(slot.stream)
        slot.future = self.pool.submit(self._finish, slot)

    @torch.inference_mode()
    def run(self):
        mode = self.schedule.mode
        if mode == "cpu_full":
            self.output.copy_(full_frontier(self.cpu_model, self.inputs, self.basis, self.case))
            return self.output
        if mode == "gpu_full":
            result = full_frontier(self.gpu_model, self.inputs.to("cuda", non_blocking=True), self.gpu_basis, self.case)
            self.output.copy_(result, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            return self.output
        offset = 0
        active = {}
        for slot in self.slots:
            if offset == self.case.roots:
                break
            slot.offset = offset
            slot.count = min(self.schedule.chunk, self.case.roots - offset)
            offset += slot.count
            self._launch(slot)
            active[slot.future] = slot
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                slot = active.pop(future)
                future.result()
                slot.refs = None
                if mode == "hybrid" and slot.level + 1 < self.case.depth:
                    self._launch(slot, continuation=True)
                    active[slot.future] = slot
                elif offset < self.case.roots:
                    slot.offset = offset
                    slot.count = min(self.schedule.chunk, self.case.roots - offset)
                    offset += slot.count
                    self._launch(slot)
                    active[slot.future] = slot
        return self.output


def gpu_status():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        return result.stdout.strip() or result.stderr.strip()
    except Exception as error:
        return str(error)


def schedules(chunks, quick=False):
    result = [Schedule("cpu_full", native_threads=n) for n in ([1] if quick else [1, 4, 8])]
    result.append(Schedule("gpu_full"))
    for chunk in chunks:
        for slots in ([1, 2] if quick else [1, 2, 4, 8]):
            result.append(Schedule("gpu_chunks", chunk, slots, slots))
            for workers in ([1] if quick else [1, 2, 4, 8]):
                if workers <= slots:
                    result.append(Schedule("hybrid", chunk, slots, workers))
    return result


def effective_schedules(case, chunks, quick=False):
    """Avoid retiming equivalent capacities when there are too few chunks."""
    result = []
    for schedule in schedules(chunks, quick):
        if schedule.chunk:
            chunk = min(schedule.chunk, case.roots)
            jobs = (case.roots + chunk - 1) // chunk
            slots = min(schedule.slots, jobs)
            schedule = Schedule(schedule.mode, chunk, slots, min(schedule.workers, slots), schedule.native_threads)
        if schedule not in result:
            result.append(schedule)
    return result


def transfer_payload(case, schedule):
    initial = case.roots * case.width * 4
    final = case.roots * case.leaves_per_root * 3 * 4
    if schedule.mode == "cpu_full":
        return {"h2d_bytes": 0, "d2h_bytes": 0}
    if schedule.mode == "hybrid":
        intermediate_points = case.roots * sum(case.samples ** (level + 1) for level in range(case.depth - 1))
        return {"h2d_bytes": initial + intermediate_points * 3 * 4, "d2h_bytes": case.nodes * 4 * 3 * 4}
    return {"h2d_bytes": initial, "d2h_bytes": final}


def run_experiment(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    configure(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")
    raw = {"created_utc": datetime.now(timezone.utc).isoformat(), "settings": vars(args),
           "environment": environment(), "gpu_before": gpu_status(),
           "status": "running", "timing_scope": "CPU inputs to CPU final outputs; all copies, fill and drain included; setup and warmup excluded",
           "contention_note": {
               "contended_pilot": "External GPU workload active: timings are a provisional contended pilot.",
               "operator_confirmed_idle": "Operator reports other compute workloads idle; this is not automatic proof of isolation.",
               "unverified": "External load not independently verified; snapshots do not establish isolation."
           }[args.load_context],
           "rows": [], "round_observations": []}
    source_files = [Path(__file__), PROJECT / "bezier_sphere/model.py", PROJECT / "bezier_sphere/geometry.py",
                    Path(__file__).parents[1] / "docs/curve_pipeline_protocol.md",
                    Path(__file__).with_name("common.py"), Path(__file__).with_name("__init__.py")]
    depth_protocol = Path(__file__).parents[1] / "docs/curve_pipeline_depth.md"
    if depth_protocol.exists():
        source_files.append(depth_protocol)
    raw["sources"] = {}
    for source in source_files:
        data = source.read_bytes()
        raw["sources"][str(source)] = hashlib.sha256(data).hexdigest()
        target = output / "source" / source.name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(data)
    write_json(output / "results.json", raw)
    profiles = [(64, 32)] if args.quick else [(64, 32), (512, 512)]
    for approach in ("independent", "dependent_tree"):
        for width, router_width in profiles:
            roots = (37 if args.quick else args.independent_roots) if approach == "independent" else (9 if args.quick else args.tree_roots)
            depth, samples = (1, 4) if approach == "independent" else (4, 2)
            case = Case(approach, width, router_width, roots, depth, samples, args.hidden_layers)
            torch.manual_seed(20261125)
            model = Producer(case).eval()
            # Match inputs across shallow/deep models despite their different
            # parameter-initialization RNG consumption.
            input_generator = torch.Generator(device="cpu").manual_seed(20261126)
            inputs = torch.randn(roots, width, generator=input_generator).pin_memory()
            gpu_model = copy.deepcopy(model).cuda().eval()
            times = (torch.arange(samples, dtype=torch.float32) + 0.5) / samples
            basis = bernstein(times)
            with torch.inference_mode():
                controls = model.route(inputs[:min(17, roots)])
                torch.testing.assert_close(evaluate(controls, basis), bezier(controls, times), rtol=1e-6, atol=1e-7)
                expected = full_frontier(model, inputs, basis, case)
            specs = effective_schedules(case, [4] if args.quick else args.chunks, args.quick)
            architecture = {
                "control_hidden_layers": case.hidden_layers,
                "control_layer_units": [case.width] + [case.router_width] * case.hidden_layers + [12],
                "control_parameters": sum(p.numel() for p in model.route.parameters()),
                "child_embedding_parameters": sum(p.numel() for p in model.child.parameters()),
                "input_seed": 20261126,
                "weight_seed": 20261125,
            }
            rows = []
            for spec in specs:
                rows.append({"case": asdict(case), "nodes": case.nodes, "schedule": asdict(spec),
                             "architecture": architecture,
                             "logical_transfer_payload": transfer_payload(case, spec),
                             "wall_ms": [], "process_cpu_ms": [], "max_abs_error": [], "verified": False,
                             "verified_rounds": 0})
            raw["rows"].extend(rows)
            for round_index in range(args.rounds):
                order = list(range(len(specs)))
                random.Random(8128 + round_index).shuffle(order)
                for index in order:
                    spec = specs[index]
                    configure(spec.native_threads)
                    runner = Runner(model, gpu_model, inputs, case, spec)
                    try:
                        # First invocation is verification + warmup, outside timing.
                        actual = runner.run()
                        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
                        rows[index]["max_abs_error"].append(float((actual - expected).abs().max()))
                        torch.cuda.synchronize()
                        cpu_start = time.process_time()
                        start = time.perf_counter()
                        for _ in range(args.iterations):
                            runner.run()
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - start) * 1000 / args.iterations
                        cpu_elapsed = (time.process_time() - cpu_start) * 1000 / args.iterations
                        rows[index]["wall_ms"].append(elapsed)
                        rows[index]["process_cpu_ms"].append(cpu_elapsed)
                        # Also check final timed output: catches stale/reused buffers.
                        torch.testing.assert_close(runner.output, expected, rtol=1e-4, atol=2e-5)
                        rows[index]["verified"] = True
                        rows[index]["verified_rounds"] += 1
                    finally:
                        runner.close()
                observation = {"approach": approach, "width": width, "hidden_layers": case.hidden_layers,
                               "round": round_index + 1,
                               "configurations": len(specs), "schedule_order": order, "gpu": gpu_status()}
                raw["round_observations"].append(observation)
                write_json(output / "results.json", raw)
                print(json.dumps(observation), flush=True)
            for row in rows:
                row["median_ms"] = statistics.median(row["wall_ms"])
                row["nodes_per_second"] = case.nodes * 1000 / row["median_ms"]
            del gpu_model, model, inputs, expected, runner
            torch.cuda.empty_cache()
    raw["gpu_after"] = gpu_status()
    raw["status"] = "complete"
    write_json(output / "results.json", raw)
    print(json.dumps({"output": str(output.resolve()), "status": "complete", "configurations": len(raw["rows"])}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--independent-roots", type=int, default=8192)
    parser.add_argument("--tree-roots", type=int, default=512)
    parser.add_argument("--chunks", type=int, nargs="+", default=[32, 256])
    parser.add_argument("--hidden-layers", type=int, default=4,
                        help="Hidden layers in the control-point network; separate from the four tree levels (default: 4)")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--load-context", choices=["unverified", "contended_pilot", "operator_confirmed_idle"], default="unverified")
    args = parser.parse_args()
    if min(args.rounds, args.iterations, args.independent_roots, args.tree_roots, args.hidden_layers, *args.chunks) < 1:
        parser.error("All counts must be positive")
    run_experiment(args)


if __name__ == "__main__":
    main()
