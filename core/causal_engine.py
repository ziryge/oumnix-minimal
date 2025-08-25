"""
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import networkx as nx
from scipy import stats
import time
import pickle

@dataclass
class CausalRelation:
    """
"""
    cause: str
    effect: str
    strength: float  
    confidence: float  
    contexts: List[str]  
    equation: Optional[str] = None  
    discovered_time: float = field(default_factory=time.time)
    test_count: int = 0
    success_count: int = 0

@dataclass
class CausalEvent:
    """
"""
    timestamp: float
    variables: Dict[str, float]
    action: Optional[str] = None  
    context: str = "default"
    outcome: Optional[Dict[str, float]] = None

class StructuralCausalModel:
    """
"""
    def __init__(self):
        self.graph = nx.DiGraph()  
        self.relations: Dict[Tuple[str, str], CausalRelation] = {}
        self.variables: Set[str] = set()
        self.contexts: Set[str] = set()
        
        
        self.equations: Dict[str, Dict[str, float]] = {}  
        self.noise_terms: Dict[str, float] = {}  
        
    def add_relation(self, relation: CausalRelation) -> None:
        """
"""
        key = (relation.cause, relation.effect)
        self.relations[key] = relation
        
        
        self.graph.add_edge(relation.cause, relation.effect, 
                           weight=relation.strength, 
                           confidence=relation.confidence)
        
        self.variables.add(relation.cause)
        self.variables.add(relation.effect)
        self.contexts.update(relation.contexts)
    
    def remove_relation(self, cause: str, effect: str) -> None:
        """
"""
        key = (cause, effect)
        if key in self.relations:
            del self.relations[key]
            if self.graph.has_edge(cause, effect):
                self.graph.remove_edge(cause, effect)
    
    def get_parents(self, variable: str) -> List[str]:
        """
"""
        return list(self.graph.predecessors(variable))
    
    def get_children(self, variable: str) -> List[str]:
        """
"""
        return list(self.graph.successors(variable))
    
    def simulate_intervention(self, interventions: Dict[str, float], 
                            context: str = "default") -> Dict[str, float]:
        """
"""
        
        result = interventions.copy()
        
        
        try:
            topo_order = list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            
            topo_order = list(self.variables)
        
        
        for var in topo_order:
            if var in interventions:
                continue  
            
            parents = self.get_parents(var)
            if not parents:
                continue
            
            
            value = 0.0
            for parent in parents:
                if parent in result:
                    relation_key = (parent, var)
                    if relation_key in self.relations:
                        relation = self.relations[relation_key]
                        if context in relation.contexts or not relation.contexts:
                            value += relation.strength * result[parent]
            
            
            if var in self.noise_terms:
                noise = np.random.normal(0, self.noise_terms[var])
                value += noise
            
            result[var] = value
        
        return result
    
    def estimate_effect(self, cause: str, effect: str, 
                       intervention_value: float = 1.0,
                       context: str = "default") -> float:
        """
"""
        
        baseline = self.simulate_intervention({}, context)
        intervention = self.simulate_intervention({cause: intervention_value}, context)
        
        if effect in baseline and effect in intervention:
            return intervention[effect] - baseline[effect]
        
        return 0.0
    
    def get_confounders(self, cause: str, effect: str) -> List[str]:
        """
"""
        confounders = []
        
        for var in self.variables:
            if var != cause and var != effect:
                
                if (self.graph.has_edge(var, cause) and 
                    self.graph.has_edge(var, effect)):
                    confounders.append(var)
        
        return confounders
    
    def test_invariance(self, relation: CausalRelation, 
                       new_context: str, events: List[CausalEvent]) -> bool:
        """
"""
        
        context_events = [e for e in events if e.context == new_context]
        
        if len(context_events) < 5:  
            return False
        
        
        cause_values = []
        effect_values = []
        
        for event in context_events:
            if (relation.cause in event.variables and 
                relation.effect in event.variables):
                cause_values.append(event.variables[relation.cause])
                effect_values.append(event.variables[relation.effect])
        
        if len(cause_values) < 3:
            return False
        
        
        correlation, p_value = stats.pearsonr(cause_values, effect_values)
        
        
        expected_strength = relation.strength
        tolerance = 0.3  
        
        return abs(correlation - expected_strength) < tolerance and p_value < 0.05
    
    def prune_weak_relations(self, min_confidence: float = 0.3) -> None:
        """
"""
        to_remove = []
        
        for key, relation in self.relations.items():
            if relation.confidence < min_confidence:
                to_remove.append(key)
        
        for key in to_remove:
            cause, effect = key
            self.remove_relation(cause, effect)
    
    def get_summary(self) -> Dict[str, Any]:
        """
"""
        return {
            'n_variables': len(self.variables),
            'n_relations': len(self.relations),
            'n_contexts': len(self.contexts),
            'avg_confidence': np.mean([r.confidence for r in self.relations.values()]) if self.relations else 0.0,
            'graph_density': nx.density(self.graph),
            'has_cycles': not nx.is_directed_acyclic_graph(self.graph)
        }

class CausalDiscovery:
    """
"""
    
    @staticmethod
    def granger_causality(events: List[CausalEvent], 
                         var1: str, var2: str,
                         max_lag: int = 3) -> Tuple[float, float]:
        """
"""
        
        times = []
        values1 = []
        values2 = []
        
        for event in sorted(events, key=lambda x: x.timestamp):
            if var1 in event.variables and var2 in event.variables:
                times.append(event.timestamp)
                values1.append(event.variables[var1])
                values2.append(event.variables[var2])
        
        if len(values1) < max_lag * 2:
            return 0.0, 1.0  
        
        
        
        
        
        n = len(values2)
        
        
        X1 = []  
        X2 = []  
        y = []
        
        for i in range(max_lag, n):
            
            y.append(values2[i])
            
            
            row1 = [1.0]  
            row2 = [1.0]
            
            for lag in range(1, max_lag + 1):
                row1.append(values2[i - lag])
                row2.append(values2[i - lag])
            
            
            for lag in range(1, max_lag + 1):
                row2.append(values1[i - lag])
            
            X1.append(row1)
            X2.append(row2)
        
        if len(y) < 5:
            return 0.0, 1.0
        
        
        X1 = np.array(X1)
        X2 = np.array(X2)
        y = np.array(y)
        
        try:
            
            beta1 = np.linalg.lstsq(X1, y, rcond=None)[0]
            pred1 = X1 @ beta1
            rss1 = np.sum((y - pred1) ** 2)
            
            
            beta2 = np.linalg.lstsq(X2, y, rcond=None)[0]
            pred2 = X2 @ beta2
            rss2 = np.sum((y - pred2) ** 2)
            
            
            df1 = X2.shape[1] - X1.shape[1]  
            df2 = len(y) - X2.shape[1]
            
            if df2 <= 0 or rss2 <= 0:
                return 0.0, 1.0
            
            f_stat = ((rss1 - rss2) / df1) / (rss2 / df2)
            
            
            p_value = 1.0 / (1.0 + f_stat)  
            
            return f_stat, p_value
            
        except np.linalg.LinAlgError:
            return 0.0, 1.0
    
    @staticmethod
    def invariance_test(events: List[CausalEvent],
                       var1: str, var2: str,
                       contexts: List[str]) -> Dict[str, float]:
        """
"""
        correlations = {}
        
        for context in contexts:
            context_events = [e for e in events if e.context == context]
            
            values1 = []
            values2 = []
            
            for event in context_events:
                if var1 in event.variables and var2 in event.variables:
                    values1.append(event.variables[var1])
                    values2.append(event.variables[var2])
            
            if len(values1) >= 3:
                corr, _ = stats.pearsonr(values1, values2)
                correlations[context] = corr
            else:
                correlations[context] = 0.0
        
        return correlations
    
    @staticmethod
    def detect_confounders(events: List[CausalEvent],
                          cause: str, effect: str,
                          candidate_vars: List[str]) -> List[str]:
        """
"""
        confounders = []
        
        
        cause_vals = []
        effect_vals = []
        confounder_vals = {var: [] for var in candidate_vars}
        
        for event in events:
            if (cause in event.variables and effect in event.variables and
                all(var in event.variables for var in candidate_vars)):
                
                cause_vals.append(event.variables[cause])
                effect_vals.append(event.variables[effect])
                
                for var in candidate_vars:
                    confounder_vals[var].append(event.variables[var])
        
        if len(cause_vals) < 5:
            return confounders
        
        
        orig_corr, _ = stats.pearsonr(cause_vals, effect_vals)
        
        
        for var in candidate_vars:
            conf_vals = confounder_vals[var]
            
            
            
            try:
                
                X = np.column_stack([conf_vals])
                
                
                beta_cause = np.linalg.lstsq(X, cause_vals, rcond=None)[0]
                cause_residual = np.array(cause_vals) - X @ beta_cause
                
                
                beta_effect = np.linalg.lstsq(X, effect_vals, rcond=None)[0]
                effect_residual = np.array(effect_vals) - X @ beta_effect
                
                
                partial_corr, _ = stats.pearsonr(cause_residual, effect_residual)
                
                
                if abs(orig_corr - partial_corr) > 0.2:
                    confounders.append(var)
                    
            except (np.linalg.LinAlgError, ValueError):
                continue
        
        return confounders

class CausalEngine:
    """
"""
    def __init__(self, discovery_threshold: float = 0.3):
        self.scm = StructuralCausalModel()
        self.event_buffer = deque(maxlen=1000)
        self.discovery_threshold = discovery_threshold
        
        
        self.min_events_for_discovery = 10
        self.max_lag_granger = 3
        self.confidence_decay = 0.95  
        
        
        self.stats = {
            'relations_discovered': 0,
            'relations_pruned': 0,
            'interventions_simulated': 0,
            'last_discovery_time': 0.0
        }
    
    def add_event(self, event: CausalEvent) -> None:
        """
"""
        self.event_buffer.append(event)
        
        
        if len(self.event_buffer) % 50 == 0:  
            self._discover_relations()
    
    def _discover_relations(self) -> None:
        """
"""
        events = list(self.event_buffer)
        
        if len(events) < self.min_events_for_discovery:
            return
        
        
        all_vars = set()
        for event in events:
            all_vars.update(event.variables.keys())
        
        all_vars = list(all_vars)
        
        
        for i, var1 in enumerate(all_vars):
            for j, var2 in enumerate(all_vars):
                if i != j:
                    self._test_causal_relation(events, var1, var2)
        
        
        self.stats['last_discovery_time'] = time.time()
    
    def _test_causal_relation(self, events: List[CausalEvent], 
                             cause: str, effect: str) -> None:
        """
"""
        
        f_stat, p_value = CausalDiscovery.granger_causality(
            events, cause, effect, self.max_lag_granger
        )
        
        if p_value > 0.05:  
            return
        
        
        strength = min(1.0, f_stat / 10.0)  
        
        if strength < self.discovery_threshold:
            return
        
        
        contexts = list(set(e.context for e in events))
        invariance_scores = CausalDiscovery.invariance_test(
            events, cause, effect, contexts
        )
        
        
        if len(invariance_scores) > 1:
            invariance_std = np.std(list(invariance_scores.values()))
            confidence = max(0.1, 1.0 - invariance_std)
        else:
            confidence = 0.5  
        
        
        other_vars = [v for v in set().union(*[e.variables.keys() for e in events])
                     if v != cause and v != effect]
        confounders = CausalDiscovery.detect_confounders(
            events, cause, effect, other_vars[:5]  
        )
        
        
        if confounders:
            confidence *= 0.8  
        
        
        relation = CausalRelation(
            cause=cause,
            effect=effect,
            strength=strength,
            confidence=confidence,
            contexts=list(invariance_scores.keys()),
            discovered_time=time.time()
        )
        
        
        existing_key = (cause, effect)
        if existing_key in self.scm.relations:
            
            old_relation = self.scm.relations[existing_key]
            
            alpha = 0.3  
            relation.strength = (1 - alpha) * old_relation.strength + alpha * strength
            relation.confidence = (1 - alpha) * old_relation.confidence + alpha * confidence
            relation.test_count = old_relation.test_count + 1
        else:
            self.stats['relations_discovered'] += 1
        
        self.scm.add_relation(relation)
    
    def plan_intervention(self, goal_var: str, target_value: float,
                         context: str = "default") -> Dict[str, float]:
        """
"""
        
        parents = self.scm.get_parents(goal_var)
        
        if not parents:
            return {}  
        
        
        best_intervention = {}
        best_score = float('-inf')
        
        for parent in parents:
            
            for value in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                intervention = {parent: value}
                result = self.scm.simulate_intervention(intervention, context)
                
                if goal_var in result:
                    
                    score = -abs(result[goal_var] - target_value)
                    
                    if score > best_score:
                        best_score = score
                        best_intervention = intervention
        
        self.stats['interventions_simulated'] += 1
        return best_intervention
    
    def explain_outcome(self, event: CausalEvent) -> Dict[str, Any]:
        """
"""
        explanations = {}
        
        for var, value in event.variables.items():
            
            parents = self.scm.get_parents(var)
            
            if parents:
                contributions = {}
                for parent in parents:
                    if parent in event.variables:
                        relation_key = (parent, var)
                        if relation_key in self.scm.relations:
                            relation = self.scm.relations[relation_key]
                            contribution = relation.strength * event.variables[parent]
                            contributions[parent] = contribution
                
                explanations[var] = {
                    'value': value,
                    'contributions': contributions,
                    'main_cause': max(contributions.items(), key=lambda x: abs(x[1]))[0] if contributions else None
                }
        
        return explanations
    
    def consolidate_model(self) -> None:
        """
"""
        
        for relation in self.scm.relations.values():
            time_since_discovery = time.time() - relation.discovered_time
            decay_factor = self.confidence_decay ** (time_since_discovery / 86400)  
            relation.confidence *= decay_factor
        
        
        old_count = len(self.scm.relations)
        self.scm.prune_weak_relations(min_confidence=0.2)
        new_count = len(self.scm.relations)
        
        self.stats['relations_pruned'] += (old_count - new_count)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
"""
        return {
            'scm_summary': self.scm.get_summary(),
            'stats': self.stats,
            'buffer_size': len(self.event_buffer),
            'strongest_relations': self._get_strongest_relations(5)
        }
    
    def _get_strongest_relations(self, top_k: int) -> List[Dict[str, Any]]:
        """
"""
        relations = list(self.scm.relations.values())
        relations.sort(key=lambda r: r.strength * r.confidence, reverse=True)
        
        return [
            {
                'cause': r.cause,
                'effect': r.effect,
                'strength': r.strength,
                'confidence': r.confidence,
                'contexts': r.contexts
            }
            for r in relations[:top_k]
        ]
    
    def save_state(self, path: str) -> None:
        """
"""
        state = {
            'scm': self.scm,
            'event_buffer': list(self.event_buffer),
            'stats': self.stats,
            'discovery_threshold': self.discovery_threshold
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, path: str) -> None:
        """
"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.scm = state['scm']
        self.event_buffer = deque(state['event_buffer'], maxlen=1000)
        self.stats = state['stats']
        self.discovery_threshold = state['discovery_threshold']

__all__ = [
    'CausalEngine',
    'StructuralCausalModel',
    'CausalRelation',
    'CausalEvent',
    'CausalDiscovery'
]