"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import time
import random
import heapq
from pathlib import Path
import pickle

@dataclass
class Experience:
    """
"""
    timestamp: float
    input_data: torch.Tensor
    target_data: torch.Tensor
    context: Dict[str, Any]
    reward: float
    surprise: float
    importance: float = 0.0
    replay_count: int = 0
    last_replay: float = 0.0
    
    def update_importance(self, td_error: float, novelty: float):
        """
"""
        
        age_factor = np.exp(-(time.time() - self.timestamp) / 86400)  
        self.importance = (
            0.4 * abs(td_error) +
            0.3 * novelty +
            0.2 * self.surprise +
            0.1 * age_factor
        )

@dataclass
class ConsolidationTask:
    """
"""
    task_type: str  
    priority: float
    data: Any
    estimated_time: float
    created_time: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        return self.priority > other.priority  

class PrioritizedReplayBuffer:
    """
"""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  
        self.beta = beta    
        
        self.experiences: List[Experience] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        
        self.tree_capacity = 1
        while self.tree_capacity < capacity:
            self.tree_capacity *= 2
        
        self.sum_tree = np.zeros(2 * self.tree_capacity - 1)
        self.min_tree = np.full(2 * self.tree_capacity - 1, float('inf'))
    
    def add(self, experience: Experience) -> None:
        """
"""
        
        priority = self.priorities.max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.experiences.append(experience)
            self.size += 1
        else:
            self.experiences[self.position] = experience
        
        self.priorities[self.position] = priority
        self._update_tree(self.position, priority)
        
        self.position = (self.position + 1) % self.capacity
    
    def _update_tree(self, idx: int, priority: float) -> None:
        """
"""
        tree_idx = idx + self.tree_capacity - 1
        
        self.sum_tree[tree_idx] = priority ** self.alpha
        self.min_tree[tree_idx] = priority ** self.alpha
        
        
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            left_child = 2 * tree_idx + 1
            right_child = 2 * tree_idx + 2
            
            self.sum_tree[tree_idx] = self.sum_tree[left_child] + self.sum_tree[right_child]
            self.min_tree[tree_idx] = min(self.min_tree[left_child], self.min_tree[right_child])
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
"""
        if self.size == 0:
            return [], np.array([]), np.array([])
        
        indices = []
        weights = []
        
        
        min_prob = self.min_tree[0] / self.sum_tree[0]
        max_weight = (min_prob * self.size) ** (-self.beta)
        
        
        segment_size = self.sum_tree[0] / batch_size
        
        for i in range(batch_size):
            
            value = random.uniform(i * segment_size, (i + 1) * segment_size)
            idx = self._retrieve(0, value)
            
            if idx < self.size:
                indices.append(idx)
                
                
                prob = self.sum_tree[idx + self.tree_capacity - 1] / self.sum_tree[0]
                weight = (prob * self.size) ** (-self.beta)
                weights.append(weight / max_weight)
        
        experiences = [self.experiences[i] for i in indices]
        weights = np.array(weights, dtype=np.float32)
        indices = np.array(indices)
        
        return experiences, weights, indices
    
    def _retrieve(self, idx: int, value: float) -> int:
        """
"""
        left_child = 2 * idx + 1
        right_child = 2 * idx + 2
        
        if left_child >= len(self.sum_tree):
            return idx - self.tree_capacity + 1
        
        if value <= self.sum_tree[left_child]:
            return self._retrieve(left_child, value)
        else:
            return self._retrieve(right_child, value - self.sum_tree[left_child])
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
"""
        for idx, priority in zip(indices, priorities):
            if idx < self.size:
                self.priorities[idx] = priority
                self._update_tree(idx, priority)
                
                
                if hasattr(self.experiences[idx], 'importance'):
                    self.experiences[idx].importance = priority

class AdvancedEWC:
    """
"""
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 0.4):
        self.model = model
        self.lambda_ewc = lambda_ewc
        
        
        self.task_params: List[Dict[str, torch.Tensor]] = []
        self.task_fishers: List[Dict[str, torch.Tensor]] = []
        self.task_weights: List[float] = []
        
        
        self.consolidated_fisher: Dict[str, torch.Tensor] = {}
        self.consolidated_params: Dict[str, torch.Tensor] = {}
    
    def compute_fisher_information(self, dataloader, sample_size: int = 1000) -> Dict[str, torch.Tensor]:
        """
"""
        fisher = {}
        
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)
        
        self.model.eval()
        sample_count = 0
        
        for batch in dataloader:
            if sample_count >= sample_size:
                break
            
            
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch
            else:
                inputs = batch
                targets = batch  
            
            self.model.zero_grad()
            
            
            outputs = self.model(inputs)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2
            
            sample_count += inputs.size(0)
        
        
        for name in fisher:
            fisher[name] /= sample_count
        
        return fisher
    
    def consolidate_task(self, dataloader, task_weight: float = 1.0) -> None:
        """
"""
        
        fisher = self.compute_fisher_information(dataloader)
        
        
        params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params[name] = param.detach().clone()
        
        
        self.task_params.append(params)
        self.task_fishers.append(fisher)
        self.task_weights.append(task_weight)
        
        
        self._update_consolidated_fisher()
    
    def _update_consolidated_fisher(self) -> None:
        """
"""
        if not self.task_fishers:
            return
        
        
        total_weight = sum(self.task_weights)
        
        for name in self.task_fishers[0].keys():
            weighted_fisher = torch.zeros_like(self.task_fishers[0][name])
            weighted_params = torch.zeros_like(self.task_params[0][name])
            
            for fisher, params, weight in zip(self.task_fishers, self.task_params, self.task_weights):
                normalized_weight = weight / total_weight
                weighted_fisher += normalized_weight * fisher[name]
                weighted_params += normalized_weight * params[name]
            
            self.consolidated_fisher[name] = weighted_fisher
            self.consolidated_params[name] = weighted_params
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """
"""
        if not self.consolidated_fisher:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        ewc_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for name, param in self.model.named_parameters():
            if name in self.consolidated_fisher and param.requires_grad:
                fisher = self.consolidated_fisher[name]
                old_param = self.consolidated_params[name]
                
                
                loss_term = fisher * (param - old_param) ** 2
                ewc_loss += loss_term.sum()
        
        return self.lambda_ewc * ewc_loss
    
    def importance_weights(self) -> Dict[str, torch.Tensor]:
        """
"""
        if not self.consolidated_fisher:
            return {}
        
        weights = {}
        for name, fisher in self.consolidated_fisher.items():
            
            max_fisher = fisher.max()
            if max_fisher > 0:
                weights[name] = fisher / max_fisher
            else:
                weights[name] = torch.zeros_like(fisher)
        
        return weights

class KnowledgeSynthesizer:
    """
"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.knowledge_base: Dict[str, Any] = {}
        self.synthesis_history: List[Dict[str, Any]] = []
    
    def extract_patterns(self, experiences: List[Experience]) -> Dict[str, Any]:
        """
"""
        if not experiences:
            return {}
        
        patterns = {
            'input_clusters': [],
            'output_patterns': [],
            'context_rules': [],
            'temporal_sequences': []
        }
        
        
        context_groups = defaultdict(list)
        for exp in experiences:
            context_key = str(sorted(exp.context.items()))
            context_groups[context_key].append(exp)
        
        
        for context, group_exps in context_groups.items():
            if len(group_exps) < 3:
                continue
            
            
            inputs = torch.stack([exp.input_data for exp in group_exps])
            targets = torch.stack([exp.target_data for exp in group_exps])
            
            
            input_mean = inputs.mean(dim=0)
            input_std = inputs.std(dim=0)
            
            patterns['input_clusters'].append({
                'context': context,
                'mean': input_mean,
                'std': input_std,
                'count': len(group_exps)
            })
            
            
            target_mean = targets.mean(dim=0)
            target_std = targets.std(dim=0)
            
            patterns['output_patterns'].append({
                'context': context,
                'input_mean': input_mean,
                'target_mean': target_mean,
                'target_std': target_std,
                'confidence': 1.0 / (1.0 + target_std.mean().item())
            })
        
        return patterns
    
    def synthesize_rules(self, patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
"""
        rules = []
        
        
        for cluster in patterns.get('input_clusters', []):
            if cluster['count'] >= 5:  
                rule = {
                    'type': 'input_cluster',
                    'condition': f"input similar to {cluster['mean'][:5].tolist()}...",
                    'context': cluster['context'],
                    'confidence': min(1.0, cluster['count'] / 10.0),
                    'evidence_count': cluster['count']
                }
                rules.append(rule)
        
        
        for pattern in patterns.get('output_patterns', []):
            if pattern['confidence'] > 0.7:
                rule = {
                    'type': 'input_output_mapping',
                    'condition': f"input {pattern['input_mean'][:3].tolist()}...",
                    'prediction': f"output {pattern['target_mean'][:3].tolist()}...",
                    'context': pattern['context'],
                    'confidence': pattern['confidence']
                }
                rules.append(rule)
        
        return rules
    
    def compress_knowledge(self, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
"""
        compressed = {
            'rule_count': len(rules),
            'high_confidence_rules': [r for r in rules if r['confidence'] > 0.8],
            'context_summary': {},
            'pattern_types': defaultdict(int)
        }
        
        
        for rule in rules:
            context = rule['context']
            if context not in compressed['context_summary']:
                compressed['context_summary'][context] = {
                    'rule_count': 0,
                    'avg_confidence': 0.0,
                    'types': set()
                }
            
            summary = compressed['context_summary'][context]
            summary['rule_count'] += 1
            summary['avg_confidence'] += rule['confidence']
            summary['types'].add(rule['type'])
            
            compressed['pattern_types'][rule['type']] += 1
        
        
        for context_summary in compressed['context_summary'].values():
            if context_summary['rule_count'] > 0:
                context_summary['avg_confidence'] /= context_summary['rule_count']
            context_summary['types'] = list(context_summary['types'])
        
        return compressed

class AdvancedConsolidator:
    """
"""
    
    def __init__(self, model: nn.Module, memory_system: Any):
        self.model = model
        self.memory_system = memory_system
        
        
        self.replay_buffer = PrioritizedReplayBuffer(capacity=10000)
        self.ewc = AdvancedEWC(model)
        self.synthesizer = KnowledgeSynthesizer(model)
        
        
        self.consolidation_interval = 3600  
        self.dream_duration = 300  
        self.replay_batch_size = 32
        self.synthesis_threshold = 100  
        
        
        self.is_dreaming = False
        self.last_consolidation = 0
        self.dream_thread: Optional[threading.Thread] = None
        self.consolidation_queue = []  
        
        
        self.stats = {
            'consolidations_performed': 0,
            'experiences_replayed': 0,
            'patterns_extracted': 0,
            'rules_synthesized': 0,
            'knowledge_compressed': 0
        }
    
    def add_experience(self, input_data: torch.Tensor, target_data: torch.Tensor,
                      context: Dict[str, Any], reward: float, surprise: float) -> None:
        """
"""
        experience = Experience(
            timestamp=time.time(),
            input_data=input_data.detach().cpu(),
            target_data=target_data.detach().cpu(),
            context=context,
            reward=reward,
            surprise=surprise
        )
        
        
        novelty = self._compute_novelty(input_data)
        experience.update_importance(td_error=abs(reward), novelty=novelty)
        
        self.replay_buffer.add(experience)
    
    def _compute_novelty(self, input_data: torch.Tensor) -> float:
        """
"""
        
        if self.replay_buffer.size == 0:
            return 1.0
        
        
        recent_experiences, _, _ = self.replay_buffer.sample(min(10, self.replay_buffer.size))
        
        if not recent_experiences:
            return 1.0
        
        recent_inputs = torch.stack([exp.input_data for exp in recent_experiences])
        mean_input = recent_inputs.mean(dim=0)
        
        
        distance = torch.norm(input_data.cpu() - mean_input).item()
        max_distance = torch.norm(mean_input).item() + 1e-8
        
        return min(1.0, distance / max_distance)
    
    def should_consolidate(self) -> bool:
        """
"""
        time_since_last = time.time() - self.last_consolidation
        has_enough_experiences = self.replay_buffer.size >= self.synthesis_threshold
        
        return (time_since_last >= self.consolidation_interval and 
                has_enough_experiences and 
                not self.is_dreaming)
    
    def start_dream_cycle(self) -> None:
        """
"""
        if self.is_dreaming:
            return
        
        self.is_dreaming = True
        self.dream_thread = threading.Thread(target=self._dream_cycle, daemon=True)
        self.dream_thread.start()
    
    def _dream_cycle(self) -> None:
        """
"""
        try:
            start_time = time.time()
            
            print(" Starting dream cycle...")
            
            
            self._replay_phase()
            
            
            self._synthesis_phase()
            
            
            self._consolidation_phase()
            
            
            self._compression_phase()
            
            duration = time.time() - start_time
            print(f" Dream completed in {duration:.1f}s")
            
            self.last_consolidation = time.time()
            self.stats['consolidations_performed'] += 1
            
        except Exception as e:
            print(f" Error during dream: {e}")
        finally:
            self.is_dreaming = False
    
    def _replay_phase(self) -> None:
        """
"""
        print("   Replay phase...")
        
        replay_steps = min(100, self.replay_buffer.size // self.replay_batch_size)
        
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)  
        
        for step in range(replay_steps):
            
            experiences, weights, indices = self.replay_buffer.sample(self.replay_batch_size)
            
            if not experiences:
                break
            
            
            inputs = torch.stack([exp.input_data for exp in experiences])
            targets = torch.stack([exp.target_data for exp in experiences])
            weights_tensor = torch.from_numpy(weights)
            
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            
            
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), 
                                 targets.view(-1), reduction='none')
            loss = loss.view(self.replay_batch_size, -1).mean(dim=1)
            weighted_loss = (loss * weights_tensor).mean()
            
            
            ewc_loss = self.ewc.compute_ewc_loss()
            total_loss = weighted_loss + ewc_loss
            
            total_loss.backward()
            optimizer.step()
            
            
            td_errors = loss.detach().numpy()
            self.replay_buffer.update_priorities(indices, td_errors + 1e-6)
            
            
            for i, exp in enumerate(experiences):
                exp.replay_count += 1
                exp.last_replay = time.time()
            
            self.stats['experiences_replayed'] += len(experiences)
    
    def _synthesis_phase(self) -> None:
        """
"""
        print("   Synthesis phase...")
        
        
        all_experiences, _, _ = self.replay_buffer.sample(
            min(500, self.replay_buffer.size)
        )
        
        if len(all_experiences) < self.synthesis_threshold:
            return
        
        
        patterns = self.synthesizer.extract_patterns(all_experiences)
        self.stats['patterns_extracted'] += len(patterns.get('input_clusters', []))
        
        
        rules = self.synthesizer.synthesize_rules(patterns)
        self.stats['rules_synthesized'] += len(rules)
        
        
        compressed_knowledge = self.synthesizer.compress_knowledge(rules)
        self.stats['knowledge_compressed'] += 1
        
        
        if hasattr(self.memory_system, 'add_synthesized_knowledge'):
            self.memory_system.add_synthesized_knowledge(compressed_knowledge)
    
    def _consolidation_phase(self) -> None:
        """
"""
        print("   Consolidation phase...")
        
        
        experiences, _, _ = self.replay_buffer.sample(
            min(200, self.replay_buffer.size)
        )
        
        if not experiences:
            return
        
        
        class ExperienceDataset:
            def __init__(self, experiences):
                self.experiences = experiences
            
            def __iter__(self):
                for exp in self.experiences:
                    yield exp.input_data.unsqueeze(0), exp.target_data.unsqueeze(0)
        
        dataset = ExperienceDataset(experiences)
        
        
        self.ewc.consolidate_task(dataset, task_weight=1.0)
    
    def _compression_phase(self) -> None:
        """
"""
        print("   Compression phase...")
        
        
        current_time = time.time()
        old_threshold = current_time - 7 * 24 * 3600  
        
        
        to_remove = []
        for i, exp in enumerate(self.replay_buffer.experiences[:self.replay_buffer.size]):
            if (exp.timestamp < old_threshold and 
                exp.importance < 0.1 and 
                exp.replay_count < 2):
                to_remove.append(i)
        
        
        
        print(f"    Identified {len(to_remove)} experiences for removal")
    
    def get_consolidation_stats(self) -> Dict[str, Any]:
        """
"""
        return {
            **self.stats,
            'buffer_size': self.replay_buffer.size,
            'is_dreaming': self.is_dreaming,
            'time_since_last_consolidation': time.time() - self.last_consolidation,
            'ewc_tasks': len(self.ewc.task_params),
            'knowledge_base_size': len(self.synthesizer.knowledge_base)
        }
    
    def force_consolidation(self) -> None:
        """
"""
        if not self.is_dreaming:
            self.start_dream_cycle()
    
    def save_state(self, path: str) -> None:
        """
"""
        state = {
            'replay_buffer': self.replay_buffer,
            'ewc': self.ewc,
            'synthesizer': self.synthesizer,
            'stats': self.stats,
            'last_consolidation': self.last_consolidation
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, path: str) -> None:
        """
"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.replay_buffer = state['replay_buffer']
        self.ewc = state['ewc']
        self.synthesizer = state['synthesizer']
        self.stats = state['stats']
        self.last_consolidation = state['last_consolidation']

__all__ = [
    'AdvancedConsolidator',
    'PrioritizedReplayBuffer',
    'AdvancedEWC',
    'KnowledgeSynthesizer',
    'Experience'
]