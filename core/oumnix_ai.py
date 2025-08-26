"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
import re
import threading
# from pathlib import Path


from core.oumnix_minimal import OumnixMinimal, OumnixMinimalConfig
from core.metacognition import StrategyOptimizer, TaskContext, ReasoningProgram
from core.causal_engine import CausalEngine, CausalEvent
from core.analogy_engine import AnalogyEngine, TaskStructure
from memory.infinity_window import InfinityWindow, MemoryConfig
from memory.advanced_consolidator import AdvancedConsolidator
from memory.persistence import PersistenceManager
from neuro.advanced_chemistry import global_neuro_chemistry
from utils.tokenizer import tokenizer

@dataclass
class OumnixAIConfig:
    """
"""
    
    vocab_size: int = 32000
    model_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    
    
    hot_kv_size: int = 4096
    warm_kv_windows: int = 32
    context_tree_fanout: int = 8
    
    
    max_reasoning_steps: int = 4
    strategy_beam_size: int = 3
    
    
    use_neurochemistry: bool = True
    neuro_update_frequency: int = 1
    
    
    consolidation_interval: int = 3600  
    dream_duration: int = 300  
    
    
    auto_save_interval: int = 600  
    state_dir: str = ".ai_state"
    encryption_password: Optional[str] = None
    
    
    max_sequence_length: int = 8192
    batch_size: int = 1
    device: str = "auto"

class OumnixAI(nn.Module):
    """
"""
    
    def __init__(self, config: OumnixAIConfig):
        super().__init__()
        self.config = config
        
        
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        
        self._init_core_model()
        self._init_memory_system()
        self._init_metacognition()
        self._init_neurochemistry()
        self._init_consolidation()
        self._init_persistence()
        
        
        self.is_active = False
        self.interaction_count = 0
        self.last_input_time = 0.0
        self.current_context = {}
        
        
        self.background_threads: List[threading.Thread] = []
        self.shutdown_event = threading.Event()
        
        
        self.stats = {
            'total_interactions': 0,
            'successful_transfers': 0,
            'consolidations_performed': 0,
            'avg_response_time': 0.0,
            'memory_usage_mb': 0.0
        }
        
        print("[INIT] Oumnix agent initialized successfully")
        print(f"   Device: {self.device}")
        print(f"   Parameters: ~{self._count_parameters()/1e6:.1f}M")
        print(f"   Memory: {config.hot_kv_size} hot tokens")
    
    def _init_core_model(self):
        """
"""
        insight_config = OumnixMinimalConfig(
            vocab_size=self.config.vocab_size,
            dim=self.config.model_dim,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            hot_kv_size=self.config.hot_kv_size
        )
        
        self.core_model = OumnixMinimal(insight_config).to(self.device)
        print("[INIT] OumnixMinimal model initialized")
    
    def _init_memory_system(self):
        """
"""
        memory_config = MemoryConfig(
            hot_kv_size=self.config.hot_kv_size,
            warm_max_windows=self.config.warm_kv_windows,
            tree_fanout=self.config.context_tree_fanout
        )
        
        self.memory_system = InfinityWindow(
            config=memory_config,
            dim=self.config.model_dim,
            n_heads=self.config.n_heads,
            head_dim=self.config.model_dim // self.config.n_heads
        )
        print(" Window memory system initialized")
    
    def _init_metacognition(self):
        """
"""
        self.strategy_optimizer = StrategyOptimizer(
            max_program_length=self.config.max_reasoning_steps
        )
        
        self.causal_engine = CausalEngine()
        self.analogy_engine = AnalogyEngine()
        
        print(" Metacognitive system initialized")
    
    def _init_neurochemistry(self):
        """
"""
        if self.config.use_neurochemistry:
            self.neurochemistry = global_neuro_chemistry
        else:
            self.neurochemistry = None
        
        print(" Neurochemical system initialized")
    
    def _init_consolidation(self):
        """
"""
        self.consolidator = AdvancedConsolidator(
            model=self.core_model,
            memory_system=self.memory_system
        )
        
        self.consolidator.consolidation_interval = self.config.consolidation_interval
        self.consolidator.dream_duration = self.config.dream_duration
        
        print("[INIT] Consolidation system initialized")
    
    def _init_persistence(self):
        """
"""
        self.persistence = PersistenceManager(
            base_dir=self.config.state_dir,
            password=self.config.encryption_password
        )
        
        self.persistence.auto_save_interval = self.config.auto_save_interval
        
        print(" Persistence system initialized")
    
    def _count_parameters(self) -> int:
        """
"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def activate(self):
        """
"""
        if self.is_active:
            return
        
        self.is_active = True
        self.shutdown_event.clear()
        
        
        consolidation_thread = threading.Thread(
            target=self._consolidation_loop,
            daemon=True,
            name="ConsolidationThread"
        )
        consolidation_thread.start()
        self.background_threads.append(consolidation_thread)
        
        
        autosave_thread = threading.Thread(
            target=self._autosave_loop,
            daemon=True,
            name="AutoSaveThread"
        )
        autosave_thread.start()
        self.background_threads.append(autosave_thread)
        
        print("[RUN] Oumnix agent activated - background threads started")
    
    def deactivate(self):
        """
"""
        if not self.is_active:
            return
        
        print(" Disabling Oumnix Agent...")
        
        self.is_active = False
        self.shutdown_event.set()
        
        
        for thread in self.background_threads:
            thread.join(timeout=5.0)
        
        
        self.save_state()
        
        print(" Oumnix Agent disabled")
    
    def _consolidation_loop(self):
        """
"""
        while not self.shutdown_event.is_set():
            try:
                if self.consolidator.should_consolidate():
                    print("[BG] Auto-consolidation cycle started")
                    self.consolidator.start_dream_cycle()
                
                
                self.shutdown_event.wait(timeout=300)  
                
            except Exception as e:
                print(f"[BG][WARN] Consolidation error: {e}")
                self.shutdown_event.wait(timeout=60)  
    
    def _autosave_loop(self):
        """
"""
        while not self.shutdown_event.is_set():
            try:
                if self.persistence.should_auto_save():
                    print("[BG] Auto-saving state...")
                    self.save_state()
                
                
                self.shutdown_event.wait(timeout=60)  
                
            except Exception as e:
                print(f"[BG][WARN] Auto-save error: {e}")
                self.shutdown_event.wait(timeout=60)
    
    def process_input(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
"""
        start_time = time.time()
        
        if context is None:
            context = {}
        
        
        self.current_context.update(context)
        self.current_context['timestamp'] = start_time
        self.current_context['interaction_id'] = self.interaction_count
        
        try:
            
            input_ids = torch.tensor([tokenizer.encode(text)], device=self.device)
            
            
            task_context = self._create_task_context(text, context)
            reasoning_program = self.strategy_optimizer.select_program(task_context)
            
            
            response = self._execute_reasoning_program(
                input_ids, reasoning_program, task_context
            )
            
            
            self._update_systems(input_ids, response, context)
            
            
            processing_time = time.time() - start_time
            
            final_response = {
                'text': response['text'],
                'confidence': response['confidence'],
                'reasoning_program': [p.value for p in reasoning_program.primitives],
                'neurochemistry': self._get_neuro_state(),
                'memory_stats': self.memory_system.get_stats(),
                'processing_time': processing_time,
                'interaction_id': self.interaction_count
            }
            
            
            self.stats['total_interactions'] += 1
            self.stats['avg_response_time'] = (
                (self.stats['avg_response_time'] * (self.stats['total_interactions'] - 1) + processing_time) /
                self.stats['total_interactions']
            )
            
            self.interaction_count += 1
            self.last_input_time = start_time
            
            return final_response
            
        except Exception as e:
            import traceback
            print(f" Processing error: {e}")
            traceback.print_exc()
            return {
                'text': "Sorry, an internal error occurred.",
                'confidence': 0.0,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _create_task_context(self, text: str, context: Dict[str, Any]) -> TaskContext:
        """
"""
        
        complexity = min(1.0, len(text.split()) / 100.0)
        
        
        domain = context.get('domain', 'general')
        
        
        constraints = context.get('constraints', [])
        
        return TaskContext(
            description=text,
            domain=domain,
            complexity=complexity,
            constraints=constraints,
            expected_output_type='text',
            metadata=context
        )
    
    def _execute_reasoning_program(self, input_ids: torch.Tensor, 
                                 program: ReasoningProgram,
                                 task_context: TaskContext,
                                 max_new_tokens: int = 32) -> Dict[str, Any]:
        """
"""
        
        neuro_state = self._get_neuro_state()
        generation_params = self._get_generation_params(neuro_state)
        
        generated_ids: List[int] = []
        confidence_scores: List[float] = []
        past_kv = None
        
        with torch.no_grad():
            memory_vectors = None
            if any(p.value == 'analogia' for p in program.primitives):
                task_structure = self._extract_task_structure(task_context)
                transfer_result = self.analogy_engine.attempt_transfer(task_structure)
                if transfer_result:
                    print(" Applied analog transfer")
            
            current_ids = input_ids
            for _ in range(max_new_tokens):
                outputs = self.core_model(
                    input_ids=current_ids,
                    neuro_state=neuro_state,
                    memory_vectors=memory_vectors,
                    attention_mask=torch.ones_like(current_ids, dtype=torch.bool),
                    use_cache=True,
                    past_key_values=past_kv
                )
                logits = outputs['logits']
                past_kv = outputs.get('past_key_values', past_kv)
                
                next_token_logits = logits[0, -1, :]
                scaled_logits = next_token_logits / max(1e-6, generation_params['temperature'])
                
                top_k = int(self._get_generation_params(neuro_state).get('top_k', 40))
                if top_k > 0 and top_k < scaled_logits.numel():
                    topk_vals, topk_idx = torch.topk(scaled_logits, top_k)
                    mask = torch.ones_like(scaled_logits, dtype=torch.bool)
                    mask[topk_idx] = False
                    scaled_logits[mask] = float('-inf')
                probs = F.softmax(scaled_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > generation_params['top_p']
                if sorted_indices_to_remove.numel() > 1:
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                scaled_logits[indices_to_remove] = float('-inf')
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                token_id = next_token.item()
                generated_ids.append(token_id)
                confidence_scores.append(probs[next_token].item())
                
                
                current_ids = torch.tensor([[token_id]], device=self.device, dtype=torch.long)
                
                
                tok = tokenizer.decode([token_id])
                if tok in ['.', '!', '?']:
                    break
        
        response_text = tokenizer.decode(generated_ids) if generated_ids else ''
        
        response_text = self._clean_text(response_text)
        confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0
        
        return {
            'text': response_text,
            'confidence': confidence,
            'logits': None,
            'aux_info': []
        }
    
    def _get_neuro_state(self) -> Dict[str, Any]:
        """
"""
        if self.neurochemistry:
            analysis = self.neurochemistry.analyze_state()
            return analysis['modulated_params']
        else:
            return {
                'temperature': 0.7,
                'depth_factor': 1.0,
                'lora_rank': 4,
                'moop_top_k': 2
            }
    
    def _get_generation_params(self, neuro_state: Dict[str, Any]) -> Dict[str, float]:
        """
"""
        temp = float(neuro_state.get('temperature', 0.6))
        temp = max(0.2, min(0.9, temp))
        return {
            'temperature': temp,
            'top_p': 0.9,
            'top_k': 40,
            'repetition_penalty': 1.05
        }

    def _clean_text(self, text: str) -> str:
        """
"""
        if not text:
            return text
        
        text = re.sub(r"</?[^>]{1,20}>", "", text)
        
        text = text.replace('\\n', ' ').replace('\\t', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.strip('"\' )')
        return text
    
    def _extract_task_structure(self, task_context: TaskContext) -> TaskStructure:
        """
"""
        
        words = task_context.description.split()
        entities = [w for w in words if w.isalpha() and len(w) > 3][:10]
        
        
        relations = []
        for i in range(len(entities) - 1):
            relations.append((entities[i], "follows", entities[i+1]))
        
        return TaskStructure(
            task_id=f"task_{self.interaction_count}",
            domain=task_context.domain,
            entities=entities,
            relations=relations,
            constraints=task_context.constraints,
            goal_pattern="generate_response",
            complexity_metrics={'word_count': len(words)}
        )
    
    def _update_systems(self, input_ids: torch.Tensor, response: Dict[str, Any], 
                       context: Dict[str, Any]):
        """
"""
        
        
        with torch.no_grad():
            input_embeddings = self.core_model.token_embed(input_ids)  
            input_embeddings = input_embeddings.detach()
            head_dim = self.config.model_dim // self.config.n_heads
            kv = input_embeddings.view(input_embeddings.size(0), input_embeddings.size(1), self.config.n_heads, head_dim)
            self.memory_system.add_tokens(
                k=kv,
                v=kv,
                text=tokenizer.decode(input_ids[0].tolist()),
                embeddings=input_embeddings
            )
        
        
        if self.neurochemistry and self.interaction_count % self.config.neuro_update_frequency == 0:
            reward = response['confidence']  
            surprise = 1.0 - response['confidence']  
            error = 0.0  
            
            self.neurochemistry.update(reward, surprise, error, context)
        
        
        if hasattr(response, 'logits'):
            target_ids = input_ids  
            self.consolidator.add_experience(
                input_data=input_ids[0],
                target_data=target_ids[0],
                context=context,
                reward=response['confidence'],
                surprise=1.0 - response['confidence']
            )
        
        
        causal_event = CausalEvent(
            timestamp=time.time(),
            variables={
                'input_length': input_ids.shape[1],
                'confidence': response['confidence'],
                'processing_time': context.get('processing_time', 0.0)
            },
            context=context.get('domain', 'general')
        )
        self.causal_engine.add_event(causal_event)
    
    def save_state(self) -> None:
        """
"""
        try:
            
            model_state = self.core_model.state_dict()
            
            memory_state = {
                'infinity_window': self.memory_system,
                'consolidator': self.consolidator
            }
            
            neuro_state = self.neurochemistry if self.neurochemistry else None
            
            metacognition_state = {
                'strategy_optimizer': self.strategy_optimizer,
                'causal_engine': self.causal_engine,
                'analogy_engine': self.analogy_engine
            }
            
            config_state = {
                'config': self.config,
                'stats': self.stats,
                'interaction_count': self.interaction_count,
                'current_context': self.current_context
            }
            
            
            self.persistence.save_complete_state(
                model_state=model_state,
                memory_state=memory_state,
                neuro_state=neuro_state,
                metacognition_state=metacognition_state,
                config=config_state
            )
            
            print(" AI state saved successfully")
            
        except Exception as e:
            print(f" Error saving state: {e}")
    
    def load_state(self) -> bool:
        """
"""
        try:
            state = self.persistence.load_complete_state()
            
            
            self.core_model.load_state_dict(state['model_weights'])
            
            
            memory_state = state['memory_state']
            self.memory_system = memory_state['infinity_window']
            self.consolidator = memory_state['consolidator']
            
            if state['neuro_state']:
                self.neurochemistry = state['neuro_state']
            
            metacognition_state = state['metacognition_state']
            self.strategy_optimizer = metacognition_state['strategy_optimizer']
            self.causal_engine = metacognition_state['causal_engine']
            self.analogy_engine = metacognition_state['analogy_engine']
            
            
            config_state = state['config']
            self.stats = config_state['stats']
            self.interaction_count = config_state['interaction_count']
            self.current_context = config_state['current_context']
            
            print(" AI state loaded successfully")
            return True
            
        except Exception as e:
            print(f" Error loading state: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """
"""
        return {
            'is_active': self.is_active,
            'interaction_count': self.interaction_count,
            'stats': self.stats,
            'model_params': self._count_parameters(),
            'memory_stats': self.memory_system.get_stats(),
            'consolidation_stats': self.consolidator.get_consolidation_stats(),
            'neurochemistry': self._get_neuro_state() if self.neurochemistry else None,
            'metacognition': {
                'causal_relations': len(self.causal_engine.scm.relations),
                'analogy_seeds': len(self.analogy_engine.manifold.seeds),
                'strategy_stats': len(self.strategy_optimizer.program_stats)
            },
            'background_threads': len([t for t in self.background_threads if t.is_alive()]),
            'device': str(self.device)
        }
    
    def force_consolidation(self):
        """
"""
        self.consolidator.force_consolidation()
    
    def reset_neurochemistry(self):
        """
"""
        if self.neurochemistry:
            self.neurochemistry.reset_to_baseline()
    
    def __del__(self):
        """
"""
        if hasattr(self, 'is_active') and self.is_active:
            self.deactivate()

def create_oumnix_ai(config: Optional[OumnixAIConfig] = None) -> OumnixAI:
    """
"""
    if config is None:
        config = OumnixAIConfig(
            vocab_size=tokenizer.vocab_size,
            model_dim=768,  
            n_layers=12,
            n_heads=12
        )
    
    return OumnixAI(config)

__all__ = [
    'OumnixAI',
    'OumnixAIConfig', 
    'create_oumnix_ai'
]
