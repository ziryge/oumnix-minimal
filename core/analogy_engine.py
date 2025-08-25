"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import time
import pickle
import hashlib

@dataclass
class TaskStructure:
    """
"""
    task_id: str
    domain: str
    entities: List[str]
    relations: List[Tuple[str, str, str]]  
    constraints: List[str]
    goal_pattern: str
    complexity_metrics: Dict[str, float]
    embedding: Optional[np.ndarray] = None

@dataclass
class IsletSeed:
    """
"""
    seed_id: str
    seed_vector: np.ndarray  
    task_structure: TaskStructure
    success_count: int = 0
    usage_count: int = 0
    creation_time: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    promotion_score: float = 0.0
    
    def update_usage(self, success: bool):
        """
"""
        self.usage_count += 1
        if success:
            self.success_count += 1
        self.last_used = time.time()
        
        
        success_rate = self.success_count / self.usage_count if self.usage_count > 0 else 0
        recency = 1.0 / (1.0 + (time.time() - self.last_used) / 86400)  
        self.promotion_score = success_rate * recency * np.log(1 + self.usage_count)

@dataclass
class AnalogicalMapping:
    """
"""
    source_task: str
    target_task: str
    entity_mapping: Dict[str, str]  
    relation_mapping: Dict[str, str]  
    transformation_matrix: np.ndarray  
    confidence: float
    structural_similarity: float
    semantic_similarity: float

class StructuralEncoder(nn.Module):
    """
"""
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        
        self.entity_embed = nn.Embedding(vocab_size, embed_dim)
        self.relation_embed = nn.Embedding(vocab_size, embed_dim)
        
        
        self.graph_encoder = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),  
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        
        self.final_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        
        self.vocab = {}
        self.next_id = 0
    
    def _get_token_id(self, token: str) -> int:
        """
"""
        if token not in self.vocab:
            if self.next_id >= self.vocab_size:
                
                return hash(token) % self.vocab_size
            self.vocab[token] = self.next_id
            self.next_id += 1
        return self.vocab[token]
    
    def encode_task(self, task: TaskStructure) -> torch.Tensor:
        """
"""
        if not task.relations:
            
            return torch.zeros(self.embed_dim)
        
        
        relation_embeddings = []
        
        for entity1, relation, entity2 in task.relations:
            e1_id = self._get_token_id(entity1)
            rel_id = self._get_token_id(relation)
            e2_id = self._get_token_id(entity2)
            
            e1_emb = self.entity_embed(torch.tensor(e1_id))
            rel_emb = self.relation_embed(torch.tensor(rel_id))
            e2_emb = self.entity_embed(torch.tensor(e2_id))
            
            
            triple_emb = torch.cat([e1_emb, rel_emb, e2_emb])
            encoded = self.graph_encoder(triple_emb)
            relation_embeddings.append(encoded)
        
        
        if relation_embeddings:
            stacked = torch.stack(relation_embeddings)
            aggregated = torch.mean(stacked, dim=0)  
        else:
            aggregated = torch.zeros(self.embed_dim)
        
        
        final_embedding = self.final_encoder(aggregated)
        
        return final_embedding

class IsletManifold:
    """
"""
    def __init__(self, seed_dim: int = 16, max_seeds: int = 1000):
        self.seed_dim = seed_dim
        self.max_seeds = max_seeds
        
        
        self.seeds: Dict[str, IsletSeed] = {}
        self.seed_embeddings: Optional[np.ndarray] = None
        
        
        self.similarity_graph = nx.Graph()
        self.similarity_threshold = 0.7
        
        
        self.structure_encoder = StructuralEncoder()
        
        
        self.n_clusters = 50
        self.kmeans = None
        self.cluster_assignments = None
        
        
        self.stats = {
            'seeds_created': 0,
            'seeds_promoted': 0,
            'transfers_attempted': 0,
            'transfers_successful': 0,
            'last_update': time.time()
        }
    
    def add_seed(self, task: TaskStructure, seed_vector: np.ndarray) -> str:
        """
"""
        
        seed_id = hashlib.sha256(f"{task.task_id}_{time.time()}".encode()).hexdigest()[:16]
        
        
        seed = IsletSeed(
            seed_id=seed_id,
            seed_vector=seed_vector,
            task_structure=task
        )
        
        self.seeds[seed_id] = seed
        self.stats['seeds_created'] += 1
        
        
        self._update_structural_embeddings()
        
        
        self._update_similarity_graph(seed_id)
        
        
        if len(self.seeds) % 20 == 0:
            self._update_clusters()
        
        return seed_id
    
    def _update_structural_embeddings(self):
        """
"""
        if not self.seeds:
            return
        
        embeddings = []
        seed_ids = []
        
        for seed_id, seed in self.seeds.items():
            
            with torch.no_grad():
                embedding = self.structure_encoder.encode_task(seed.task_structure)
                embeddings.append(embedding.numpy())
                seed_ids.append(seed_id)
        
        self.seed_embeddings = np.array(embeddings)
        self.seed_ids_ordered = seed_ids
    
    def _update_similarity_graph(self, new_seed_id: str):
        """
"""
        if self.seed_embeddings is None or len(self.seeds) < 2:
            return
        
        new_seed = self.seeds[new_seed_id]
        new_embedding = self.structure_encoder.encode_task(new_seed.task_structure).numpy()
        
        
        for i, (seed_id, seed) in enumerate(self.seeds.items()):
            if seed_id == new_seed_id:
                continue
            
            
            structural_sim = np.dot(new_embedding, self.seed_embeddings[i])
            structural_sim = structural_sim / (np.linalg.norm(new_embedding) * np.linalg.norm(self.seed_embeddings[i]) + 1e-8)
            
            
            domain_sim = 1.0 if new_seed.task_structure.domain == seed.task_structure.domain else 0.3
            
            
            combined_sim = 0.7 * structural_sim + 0.3 * domain_sim
            
            
            if combined_sim > self.similarity_threshold:
                self.similarity_graph.add_edge(new_seed_id, seed_id, weight=combined_sim)
    
    def _update_clusters(self):
        """
"""
        if self.seed_embeddings is None or len(self.seeds) < self.n_clusters:
            return
        
        n_clusters = min(self.n_clusters, len(self.seeds) // 2)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_assignments = self.kmeans.fit_predict(self.seed_embeddings)
    
    def find_similar_seeds(self, target_task: TaskStructure, k: int = 5) -> List[Tuple[str, float]]:
        """
"""
        if not self.seeds:
            return []
        
        
        with torch.no_grad():
            target_embedding = self.structure_encoder.encode_task(target_task).numpy()
        
        
        if self.kmeans is not None and self.cluster_assignments is not None:
            
            target_cluster = self.kmeans.predict(target_embedding.reshape(1, -1))[0]
            
            
            candidate_indices = np.where(self.cluster_assignments == target_cluster)[0]
            
            
            if len(candidate_indices) < k * 2:
                candidate_indices = np.arange(len(self.seeds))
        else:
            candidate_indices = np.arange(len(self.seeds))
        
        
        similarities = []
        
        for idx in candidate_indices:
            if idx >= len(self.seed_ids_ordered):
                continue
                
            seed_id = self.seed_ids_ordered[idx]
            seed = self.seeds[seed_id]
            
            
            structural_sim = np.dot(target_embedding, self.seed_embeddings[idx])
            structural_sim = structural_sim / (np.linalg.norm(target_embedding) * np.linalg.norm(self.seed_embeddings[idx]) + 1e-8)
            
            
            domain_sim = 1.0 if target_task.domain == seed.task_structure.domain else 0.3
            
            
            success_score = seed.success_count / max(1, seed.usage_count)
            
            
            combined_sim = 0.5 * structural_sim + 0.2 * domain_sim + 0.3 * success_score
            
            similarities.append((seed_id, combined_sim))
        
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def compute_transformation(self, source_seed_id: str, target_task: TaskStructure) -> Optional[np.ndarray]:
        """
"""
        if source_seed_id not in self.seeds:
            return None
        
        source_seed = self.seeds[source_seed_id]
        
        
        source_structure = source_seed.task_structure
        
        
        source_entities = set(source_structure.entities)
        target_entities = set(target_task.entities)
        
        source_relations = set([rel for _, rel, _ in source_structure.relations])
        target_relations = set([rel for _, rel, _ in target_task.relations])
        
        
        entity_overlap = len(source_entities & target_entities) / max(len(source_entities | target_entities), 1)
        relation_overlap = len(source_relations & target_relations) / max(len(source_relations | target_relations), 1)
        
        if entity_overlap + relation_overlap < 0.3:  
            return None
        
        
        transformation = np.eye(self.seed_dim)
        
        
        noise_scale = 0.1 * (1.0 - (entity_overlap + relation_overlap) / 2.0)
        noise = np.random.normal(0, noise_scale, (self.seed_dim, self.seed_dim))
        transformation += noise
        
        return transformation
    
    def create_ephemeral_islet(self, source_seed_id: str, target_task: TaskStructure) -> Optional[Tuple[str, np.ndarray]]:
        """
"""
        if source_seed_id not in self.seeds:
            return None
        
        source_seed = self.seeds[source_seed_id]
        transformation = self.compute_transformation(source_seed_id, target_task)
        
        if transformation is None:
            return None
        
        
        new_seed_vector = transformation @ source_seed.seed_vector
        
        
        ephemeral_id = f"ephemeral_{hashlib.sha256(f'{source_seed_id}_{target_task.task_id}'.encode()).hexdigest()[:12]}"
        
        self.stats['transfers_attempted'] += 1
        
        return ephemeral_id, new_seed_vector
    
    def promote_ephemeral(self, ephemeral_id: str, ephemeral_vector: np.ndarray, 
                         task: TaskStructure, success: bool) -> Optional[str]:
        """
"""
        if not success:
            return None
        
        
        similar_seeds = self.find_similar_seeds(task, k=3)
        
        for seed_id, similarity in similar_seeds:
            if similarity > 0.9:  
                
                self.seeds[seed_id].update_usage(True)
                return seed_id
        
        
        permanent_id = self.add_seed(task, ephemeral_vector)
        self.seeds[permanent_id].update_usage(True)
        
        self.stats['seeds_promoted'] += 1
        self.stats['transfers_successful'] += 1
        
        return permanent_id
    
    def prune_weak_seeds(self, min_promotion_score: float = 0.1):
        """
"""
        to_remove = []
        
        for seed_id, seed in self.seeds.items():
            
            age_days = (time.time() - seed.creation_time) / 86400
            
            
            if age_days > 30 and seed.promotion_score < min_promotion_score:
                to_remove.append(seed_id)
        
        
        for seed_id in to_remove:
            del self.seeds[seed_id]
            if self.similarity_graph.has_node(seed_id):
                self.similarity_graph.remove_node(seed_id)
        
        
        if to_remove:
            self._update_structural_embeddings()
            self._update_clusters()
    
    def get_manifold_stats(self) -> Dict[str, Any]:
        """
"""
        if not self.seeds:
            return self.stats
        
        promotion_scores = [seed.promotion_score for seed in self.seeds.values()]
        usage_counts = [seed.usage_count for seed in self.seeds.values()]
        
        return {
            **self.stats,
            'total_seeds': len(self.seeds),
            'avg_promotion_score': np.mean(promotion_scores),
            'avg_usage_count': np.mean(usage_counts),
            'graph_edges': self.similarity_graph.number_of_edges(),
            'graph_density': nx.density(self.similarity_graph) if self.seeds else 0.0
        }

class AnalogyEngine:
    """
"""
    def __init__(self, seed_dim: int = 16, max_seeds: int = 1000):
        self.manifold = IsletManifold(seed_dim, max_seeds)
        self.active_ephemeral: Dict[str, Tuple[str, np.ndarray]] = {}  
        
        
        self.transfer_threshold = 0.5
        self.promotion_threshold = 0.7
        
        
        self.transfer_history = deque(maxlen=500)
        
    def register_task_solution(self, task: TaskStructure, seed_vector: np.ndarray, 
                              success: bool) -> str:
        """
"""
        
        if task.task_id in self.active_ephemeral:
            ephemeral_id, ephemeral_vector = self.active_ephemeral[task.task_id]
            
            
            promoted_id = self.manifold.promote_ephemeral(
                ephemeral_id, ephemeral_vector, task, success
            )
            
            
            del self.active_ephemeral[task.task_id]
            
            if promoted_id:
                return promoted_id
        
        
        if success:
            return self.manifold.add_seed(task, seed_vector)
        
        return ""
    
    def attempt_transfer(self, target_task: TaskStructure) -> Optional[Tuple[str, np.ndarray]]:
        """
"""
        
        similar_seeds = self.manifold.find_similar_seeds(target_task, k=5)
        
        if not similar_seeds:
            return None
        
        
        best_seed_id, best_similarity = similar_seeds[0]
        
        if best_similarity < self.transfer_threshold:
            return None
        
        
        result = self.manifold.create_ephemeral_islet(best_seed_id, target_task)
        
        if result:
            ephemeral_id, ephemeral_vector = result
            
            
            self.active_ephemeral[target_task.task_id] = (ephemeral_id, ephemeral_vector)
            
            
            self.transfer_history.append({
                'source_seed': best_seed_id,
                'target_task': target_task.task_id,
                'similarity': best_similarity,
                'timestamp': time.time()
            })
            
            return result
        
        return None
    
    def get_transfer_suggestions(self, target_task: TaskStructure, k: int = 3) -> List[Dict[str, Any]]:
        """
"""
        similar_seeds = self.manifold.find_similar_seeds(target_task, k=k)
        
        suggestions = []
        for seed_id, similarity in similar_seeds:
            seed = self.manifold.seeds[seed_id]
            
            suggestion = {
                'seed_id': seed_id,
                'source_domain': seed.task_structure.domain,
                'source_task': seed.task_structure.task_id,
                'similarity': similarity,
                'success_rate': seed.success_count / max(1, seed.usage_count),
                'usage_count': seed.usage_count,
                'entities_overlap': len(set(seed.task_structure.entities) & set(target_task.entities)),
                'relations_overlap': len(set([r[1] for r in seed.task_structure.relations]) & 
                                       set([r[1] for r in target_task.relations]))
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    def analyze_transfer_patterns(self) -> Dict[str, Any]:
        """
"""
        if not self.transfer_history:
            return {}
        
        
        domain_transfers = defaultdict(list)
        
        for transfer in self.transfer_history:
            source_seed = self.manifold.seeds.get(transfer['source_seed'])
            if source_seed:
                source_domain = source_seed.task_structure.domain
                
                target_domain = transfer['target_task'].split('_')[0] if '_' in transfer['target_task'] else 'unknown'
                
                domain_pair = f"{source_domain} -> {target_domain}"
                domain_transfers[domain_pair].append(transfer['similarity'])
        
        
        domain_stats = {}
        for domain_pair, similarities in domain_transfers.items():
            domain_stats[domain_pair] = {
                'count': len(similarities),
                'avg_similarity': np.mean(similarities),
                'success_rate': len([s for s in similarities if s > self.promotion_threshold]) / len(similarities)
            }
        
        return {
            'total_transfers': len(self.transfer_history),
            'domain_patterns': domain_stats,
            'avg_similarity': np.mean([t['similarity'] for t in self.transfer_history]),
            'recent_transfers': len([t for t in self.transfer_history 
                                   if time.time() - t['timestamp'] < 86400])  
        }
    
    def consolidate(self):
        """
"""
        self.manifold.prune_weak_seeds()
        
        
        current_time = time.time()
        to_remove = []
        
        for task_id, (ephemeral_id, _) in self.active_ephemeral.items():
            
            if current_time - time.time() > 3600:  
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.active_ephemeral[task_id]
    
    def save_state(self, path: str) -> None:
        """
"""
        state = {
            'manifold': self.manifold,
            'active_ephemeral': self.active_ephemeral,
            'transfer_history': list(self.transfer_history),
            'transfer_threshold': self.transfer_threshold,
            'promotion_threshold': self.promotion_threshold
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, path: str) -> None:
        """
"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.manifold = state['manifold']
        self.active_ephemeral = state['active_ephemeral']
        self.transfer_history = deque(state['transfer_history'], maxlen=500)
        self.transfer_threshold = state['transfer_threshold']
        self.promotion_threshold = state['promotion_threshold']

__all__ = [
    'AnalogyEngine',
    'IsletManifold', 
    'TaskStructure',
    'IsletSeed',
    'AnalogicalMapping'
]