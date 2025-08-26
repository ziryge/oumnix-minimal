import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict, deque
import pickle

class ReasoningPrimitive(Enum):
    DEDUCE = "deduce"
    INDUCE = "induce"
    ABDUCE = "abduce"
    ANALOGY = "analogy"
    DECOMPOSE = "decompose"
    VERIFY = "verify"

@dataclass
class TaskContext:
    description: str
    domain: str
    complexity: float
    constraints: List[str]
    expected_output_type: str
    metadata: Dict[str, Any]

@dataclass
class PrimitiveSpec:
    name: ReasoningPrimitive
    preconditions: List[str]
    effects: List[str]
    cost_model: Callable[[TaskContext], float]
    verifier: Optional[Callable[[Any, Any], bool]]
    io_contract: Dict[str, str]

@dataclass
class ReasoningProgram:
    primitives: List[ReasoningPrimitive]
    expected_cost: float
    confidence: float
    metadata: Dict[str, Any]

class StrategyOptimizer:
    def __init__(self, max_program_length: int = 4):
        self.max_program_length = max_program_length
        self.primitive_lib = self._init_primitive_library()
        self.domain_priors = defaultdict(
            lambda: {p.value: 1.0 / len(ReasoningPrimitive) for p in ReasoningPrimitive}
        )
        self.primitive_stats = defaultdict(lambda: {'success': 0, 'total': 0, 'avg_cost': 0.0})
        self.program_stats = defaultdict(lambda: {'success': 0, 'total': 0, 'avg_reward': 0.0})
        self.task_history = deque(maxlen=1000)

    def _init_primitive_library(self) -> Dict[ReasoningPrimitive, PrimitiveSpec]:
        lib = {}

        lib[ReasoningPrimitive.DEDUCE] = PrimitiveSpec(
            name=ReasoningPrimitive.DEDUCE,
            preconditions=["has_premises", "logical_structure"],
            effects=["conclusions", "contradictions_found"],
            cost_model=lambda ctx: 0.1 + 0.05 * ctx.complexity,
            verifier=self._verify_deduction,
            io_contract={"input": "premises", "output": "conclusions"}
        )

        lib[ReasoningPrimitive.INDUCE] = PrimitiveSpec(
            name=ReasoningPrimitive.INDUCE,
            preconditions=["has_examples", "pattern_detectable"],
            effects=["general_rule", "confidence_score"],
            cost_model=lambda ctx: 0.2 + 0.1 * ctx.complexity,
            verifier=self._verify_induction,
            io_contract={"input": "examples", "output": "pattern"}
        )

        lib[ReasoningPrimitive.ABDUCE] = PrimitiveSpec(
            name=ReasoningPrimitive.ABDUCE,
            preconditions=["has_observations", "possible_explanations"],
            effects=["best_explanation", "explanation_score"],
            cost_model=lambda ctx: 0.15 + 0.08 * ctx.complexity,
            verifier=self._verify_abduction,
            io_contract={"input": "observations", "output": "explanation"}
        )

        lib[ReasoningPrimitive.ANALOGY] = PrimitiveSpec(
            name=ReasoningPrimitive.ANALOGY,
            preconditions=["source_domain", "target_domain"],
            effects=["mapping", "transferred_knowledge"],
            cost_model=lambda ctx: 0.25 + 0.12 * ctx.complexity,
            verifier=self._verify_analogy,
            io_contract={"input": "domains", "output": "mapping"}
        )

        lib[ReasoningPrimitive.DECOMPOSE] = PrimitiveSpec(
            name=ReasoningPrimitive.DECOMPOSE,
            preconditions=["complex_task", "decomposable"],
            effects=["subtasks", "dependencies"],
            cost_model=lambda ctx: 0.1 + 0.03 * ctx.complexity,
            verifier=self._verify_decomposition,
            io_contract={"input": "task", "output": "subtasks"}
        )

        lib[ReasoningPrimitive.VERIFY] = PrimitiveSpec(
            name=ReasoningPrimitive.VERIFY,
            preconditions=["has_claim", "verification_method"],
            effects=["verification_result", "confidence"],
            cost_model=lambda ctx: 0.05 + 0.02 * ctx.complexity,
            verifier=self._verify_verification,
            io_contract={"input": "claim", "output": "result"}
        )

        return lib

    def select_program(self, task: TaskContext, beam_size: int = 3) -> ReasoningProgram:
        candidates = []
        for length in range(1, self.max_program_length + 1):
            programs = self._generate_programs(task, length)
            candidates.extend(programs)

        scored_programs = []
        for program in candidates:
            score = self._evaluate_program(program, task)
            scored_programs.append((program, score))

        scored_programs.sort(key=lambda x: x[1], reverse=True)

        if scored_programs:
            best_program, best_score = scored_programs[0]
            return best_program
        else:
            return ReasoningProgram(
                primitives=[ReasoningPrimitive.VERIFY],
                expected_cost=0.1,
                confidence=0.5,
                metadata={"fallback": True}
            )

    def _generate_programs(self, task: TaskContext, length: int) -> List[ReasoningProgram]:
        if length == 1:
            programs = []
            for primitive in ReasoningPrimitive:
                if self._check_preconditions(primitive, task):
                    cost = self.primitive_lib[primitive].cost_model(task)
                    confidence = self._estimate_confidence(primitive, task)
                    program = ReasoningProgram(
                        primitives=[primitive],
                        expected_cost=cost,
                        confidence=confidence,
                        metadata={"length": 1}
                    )
                    programs.append(program)
            return programs

        programs = []
        shorter_programs = self._generate_programs(task, length - 1)
        for base_program in shorter_programs:
            for primitive in ReasoningPrimitive:
                if self._can_extend_program(base_program, primitive, task):
                    new_primitives = base_program.primitives + [primitive]
                    new_cost = base_program.expected_cost + self.primitive_lib[primitive].cost_model(task)
                    new_confidence = min(base_program.confidence, self._estimate_confidence(primitive, task))
                    program = ReasoningProgram(
                        primitives=new_primitives,
                        expected_cost=new_cost,
                        confidence=new_confidence,
                        metadata={"length": length}
                    )
                    programs.append(program)

        return programs

    def _check_preconditions(self, primitive: ReasoningPrimitive, task: TaskContext) -> bool:
        if primitive == ReasoningPrimitive.DEDUCE:
            return "logical" in task.description.lower() or "premises" in task.description.lower()
        elif primitive == ReasoningPrimitive.INDUCE:
            return "pattern" in task.description.lower() or "examples" in task.description.lower()
        elif primitive == ReasoningPrimitive.ABDUCE:
            return "explain" in task.description.lower() or "why" in task.description.lower()
        elif primitive == ReasoningPrimitive.ANALOGY:
            return "similar" in task.description.lower() or "like" in task.description.lower()
        elif primitive == ReasoningPrimitive.DECOMPOSE:
            return task.complexity > 0.5 or "complex" in task.description.lower()
        elif primitive == ReasoningPrimitive.VERIFY:
            return True
        return True

    def _can_extend_program(self, program: ReasoningProgram, primitive: ReasoningPrimitive, task: TaskContext) -> bool:
        if primitive in program.primitives:
            return False

        last_primitive = program.primitives[-1]

        if last_primitive == ReasoningPrimitive.DECOMPOSE and primitive != ReasoningPrimitive.VERIFY:
            return True
        if last_primitive == ReasoningPrimitive.VERIFY:
            return False

        return self._check_preconditions(primitive, task)

    def _evaluate_program(self, program: ReasoningProgram, task: TaskContext) -> float:
        cost_penalty = -program.expected_cost
        confidence_bonus = program.confidence

        domain_prior = self.domain_priors[task.domain]
        prior_score = 0.0
        for primitive in program.primitives:
            key = primitive.value if hasattr(primitive, 'value') else str(primitive)
            prior_score += float(domain_prior.get(key, 0.1))

        program_key = "_".join([p.value for p in program.primitives])
        history_score = self.program_stats[program_key]['avg_reward']

        total_score = (
            0.3 * confidence_bonus +
            0.2 * cost_penalty +
            0.3 * prior_score +
            0.2 * history_score
        )
        return total_score

    def _estimate_confidence(self, primitive: ReasoningPrimitive, task: TaskContext) -> float:
        stats = self.primitive_stats[primitive.value]
        if stats['total'] == 0:
            return 0.5
        success_rate = stats['success'] / stats['total']
        complexity_penalty = 0.1 * task.complexity
        return max(0.1, min(0.9, success_rate - complexity_penalty))

    def update_performance(self, program: ReasoningProgram, task: TaskContext, success: bool, actual_cost: float, reward: float) -> None:
        for primitive in program.primitives:
            stats = self.primitive_stats[primitive.value]
            stats['total'] += 1
            if success:
                stats['success'] += 1
            alpha = 0.1
            stats['avg_cost'] = (1 - alpha) * stats['avg_cost'] + alpha * actual_cost

        program_key = "_".join([p.value for p in program.primitives])
        prog_stats = self.program_stats[program_key]
        prog_stats['total'] += 1
        if success:
            prog_stats['success'] += 1
        alpha = 0.1
        prog_stats['avg_reward'] = (1 - alpha) * prog_stats['avg_reward'] + alpha * reward

        domain = task.domain
        for primitive in program.primitives:
            key = primitive.value
            if success:
                self.domain_priors[domain][key] = self.domain_priors[domain].get(key, 0.01) + 0.1 * reward
            else:
                self.domain_priors[domain][key] = max(0.01, self.domain_priors[domain].get(key, 0.05) - 0.05)

        total = sum(self.domain_priors[domain].values())
        if total > 0:
            for k in self.domain_priors[domain].keys():
                self.domain_priors[domain][k] = self.domain_priors[domain][k] / total

        self.task_history.append({
            'task': task,
            'program': program,
            'success': success,
            'cost': actual_cost,
            'reward': reward,
            'timestamp': time.time()
        })

    def _verify_deduction(self, premises: Any, conclusion: Any) -> bool:
        return True

    def _verify_induction(self, examples: Any, pattern: Any) -> bool:
        return True

    def _verify_abduction(self, observations: Any, explanation: Any) -> bool:
        return True

    def _verify_analogy(self, mapping: Any, result: Any) -> bool:
        return True

    def _verify_decomposition(self, task: Any, subtasks: Any) -> bool:
        return True

    def _verify_verification(self, claim: Any, result: Any) -> bool:
        return True

    def save_state(self, path: str) -> None:
        state = {
            'domain_priors': dict(self.domain_priors),
            'primitive_stats': dict(self.primitive_stats),
            'program_stats': dict(self.program_stats),
            'task_history': list(self.task_history)
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, path: str) -> None:
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.domain_priors = defaultdict(lambda: np.ones(len(ReasoningPrimitive)) / len(ReasoningPrimitive))
        self.domain_priors.update(state['domain_priors'])
        self.primitive_stats = defaultdict(lambda: {'success': 0, 'total': 0, 'avg_cost': 0.0})
        self.primitive_stats.update(state['primitive_stats'])
        self.program_stats = defaultdict(lambda: {'success': 0, 'total': 0, 'avg_reward': 0.0})
        self.program_stats.update(state['program_stats'])
        self.task_history = deque(state['task_history'], maxlen=1000)


__all__ = [
    'StrategyOptimizer',
    'ReasoningPrimitive',
    'TaskContext',
    'ReasoningProgram'
]
