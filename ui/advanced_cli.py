import sys
from utils.logging_utils import get_logger
logger = get_logger("advanced_cli")
import time
import json
from typing import Dict, Any, Optional
from datetime import datetime
import threading

def print_banner():
    logger.info("Oumnix Advanced CLI")

def print_help():
    logger.info("/help, /status, /neuro, /memory, /consolidate, /save, /load, /reset-neuro, /stats, /causal, /analogies, /strategies, /set <param> <value>, /get <param>")

def format_neuro_state(neuro_data: Dict[str, Any]) -> str:
    if not neuro_data:
        return "Neurochemical system disabled"
    current = neuro_data.get('current_state', {})
    modulated = neuro_data.get('modulated_params', {})
    mood = neuro_data.get('interpreted_mood', 'unknown')
    def progress_bar(value: float, width: int = 20) -> str:
        v = max(0.0, min(1.0, float(value)))
        filled = int(v * width)
        bar = '█' * filled + '░' * (width - filled)
        return f"{bar} {v:.2f}"
    lines = [f"Mood: {mood}"]
    if current:
        for k, v in sorted(current.items()):
            try:
                val = float(v)
                lines.append(f"{k}: {progress_bar(val)}")
            except Exception:
                lines.append(f"{k}: {v}")
    if modulated:
        lines.append("Modulated parameters:")
        for k, v in sorted(modulated.items()):
            try:
                val = float(v)
                lines.append(f"  {k}: {progress_bar(val)}")
            except Exception:
                lines.append(f"  {k}: {v}")
    return "\n".join(lines)

def format_memory_stats(memory_data: Dict[str, Any]) -> str:
    used = memory_data.get('used', 0)
    capacity = memory_data.get('capacity', 0)
    ratio = (used / max(capacity, 1)) if capacity else 0.0
    return f"Memory used: {used}/{capacity} ({ratio:.2%})"

def format_system_status(status: Dict[str, Any]) -> str:
    threads = status.get('background_threads', 0)
    params = status.get('model_params', 0)
    return f"Threads: {threads}, Params: {params}"

def format_response_analysis(response: Dict[str, Any]) -> str:
    confidence = float(response.get('confidence', 0.0))
    processing_time = float(response.get('processing_time', 0.0))
    reasoning = response.get('reasoning_program', []) or []
    width = 20
    v = max(0.0, min(1.0, confidence))
    bar = '█' * int(v * width) + '░' * (width - int(v * width))
    level = 'HIGH' if v > 0.8 else ('MEDIUM' if v > 0.5 else 'LOW')
    lines = [
        f"Confidence: {bar} {v:.2f} ({level})",
        f"Processing time: {processing_time:.3f}s",
    ]
    if reasoning:
        head = reasoning if isinstance(reasoning, list) else [reasoning]
        head = head[:3]
        for i, step in enumerate(head, 1):
            lines.append(f"Step {i}: {step}")
    return "\n".join(lines)

class AdvancedCLI:
    """
    """
    
    def __init__(self, ai_instance):
        self.ai = ai_instance
        self.running = True
        self.conversation_history = []
        self.settings = {
            'show_analysis': True,
            'show_neuro': False,
            'auto_save_interval': 300  
        }
        
        self.last_save_time = time.time()
        self.auto_save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self.auto_save_thread.start()
        self._io_lock = threading.Lock()
    
    def _auto_save_loop(self):
        """
        """
        while self.running:
            time.sleep(60)  
            
            if (time.time() - self.last_save_time > self.settings['auto_save_interval'] and
                len(self.conversation_history) > 0):
                self._save_conversation()
    
    def _save_conversation(self):
        """
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Conversation saved to {filename}")
            self.last_save_time = time.time()
            
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
    
    def run(self):
        """
        """
        print_banner()
        logger.info("Type /help to see available commands")
        logger.info("=" * 60)
        
        while self.running:
            try:
                if self.ai.neurochemistry:
                    neuro_analysis = self.ai.neurochemistry.analyze_state()
                    mood_label = neuro_analysis.get('interpreted_mood', 'neutral')
                else:
                    mood_label = 'neutral'
                
                user_input = input(f"\n[{mood_label}] You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                else:
                    self._handle_message(user_input)
                    
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except EOFError:
                logger.info("Exiting...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
        
        self.running = False
        if len(self.conversation_history) > 0:
            self._save_conversation()
    
    def _handle_command(self, command: str):
        """
        """
        parts = command[1:].split()
        cmd = parts[0].lower() if parts else ""
        
        if cmd in ['quit', 'exit']:
            self.running = False
            return
        
        elif cmd == 'help':
            print_help()
        
        elif cmd == 'status':
            status = self.ai.get_system_status()
            logger.info(format_system_status(status))
        
        elif cmd == 'neuro':
            if self.ai.neurochemistry:
                neuro_analysis = self.ai.neurochemistry.analyze_state()
                logger.info(format_neuro_state(neuro_analysis))
            else:
                logger.info("Neurochemical system disabled")
        
        elif cmd == 'memory':
            memory_stats = self.ai.memory_system.get_stats()
            logger.info(format_memory_stats(memory_stats))
        
        elif cmd == 'consolidate':
            logger.info("Starting forced consolidation...")
            self.ai.force_consolidation()
            logger.info("Consolidation started")
        
        elif cmd == 'save':
            def _save():
                with self._io_lock:
                    logger.info("[save] started")
                self.ai.save_state()
                with self._io_lock:
                    logger.info("[save] completed")
            threading.Thread(target=_save, daemon=True).start()
        elif cmd == 'load':
            def _load():
                with self._io_lock:
                    logger.info("[load] started")
                ok = self.ai.load_state()
                with self._io_lock:
                    logger.info("[load] completed" if ok else "[load] failed")
            threading.Thread(target=_load, daemon=True).start()
        
        elif cmd == 'reset-neuro':
            if self.ai.neurochemistry:
                self.ai.reset_neurochemistry()
                logger.info("Neurochemistry reset to baseline")
            else:
                logger.info("Neurochemical system disabled")
        
        elif cmd == 'stats':
            status = self.ai.get_system_status()
            stats = status.get('stats', {})
            msg = f"Tokens: {stats.get('tokens_processed', 0)} | Interactions: {status.get('interaction_count', 0)} | Device: {status.get('device')}"
            logger.info(msg)
        
        elif cmd == 'causal':
            summary = self.ai.causal_engine.get_model_summary()
            scm_summary = summary.get('scm_summary', {})
            strongest = summary.get('strongest_relations', [])
            header = (
                f"Causal Model: vars={scm_summary.get('n_variables',0)} "
                f"rels={scm_summary.get('n_relations',0)} "
                f"density={scm_summary.get('graph_density',0.0):.3f} "
                f"cycles={scm_summary.get('has_cycles',False)}"
            )
            logger.info(header)
            
            for i, rel in enumerate(strongest[:5], 1):
                logger.info(f"  {i}. {rel['cause']} → {rel['effect']} (strength: {rel['strength']:.2f}, conf: {rel['confidence']:.2f})")
        
        elif cmd == 'analogies':
            manifold_stats = self.ai.analogy_engine.manifold.get_manifold_stats()
            transfer_patterns = self.ai.analogy_engine.analyze_transfer_patterns()
            logger.info(
                f"Analogy seeds={manifold_stats.get('total_seeds',0)} "
                f"edges={manifold_stats.get('graph_edges',0)} density={manifold_stats.get('graph_density',0.0):.3f}"
            )
            if transfer_patterns:
                dom = transfer_patterns.get('domain_patterns', {})
                logger.info(f"Transfers: {transfer_patterns.get('total_transfers',0)} patterns={len(dom)}")
        
        elif cmd == 'strategies':
            s = self.ai.strategy_optimizer.program_stats
            logger.info(f"Strategies tracked: {len(s)}")
        
        elif cmd == 'set':
            if len(parts) >= 3:
                param = parts[1]
                value = parts[2]
                
                if param in self.settings:
                    try:
                        if param.endswith('_interval'):
                            self.settings[param] = int(value)
                        elif value.lower() in ['true', 'false']:
                            self.settings[param] = value.lower() == 'true'
                        else:
                            self.settings[param] = value
                        
                        logger.info(f"{param} = {self.settings[param]}")
                    except ValueError:
                        logger.info(f"Invalid value for {param}")
                else:
                    logger.info(f"Unknown parameter: {param}")
            else:
                logger.info("Usage: /set <parameter> <value>")
        
        elif cmd == 'get':
            if len(parts) >= 2:
                param = parts[1]
                if param in self.settings:
                    logger.info(f"{param} = {self.settings[param]}")
                else:
                    logger.info(f"Unknown parameter: {param}")
            else:
                logger.info("Usage: /get <parameter>")
        
        else:
            logger.info(f"Unknown command: {cmd}")
            logger.info("Type /help to see available commands")
    
    def _handle_message(self, message: str):
        """
        """
        logger.info("Processing...")
        
        start_time = time.time()
        
        response = self.ai.process_input(message)
        
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_input': message,
            'ai_response': response['text'],
            'confidence': response['confidence'],
            'processing_time': response['processing_time'],
            'reasoning_program': response.get('reasoning_program', [])
        }
        self.conversation_history.append(interaction)
        
        logger.info(f"\nAI: {response['text']}")
        
        if self.settings['show_analysis']:
            logger.info(format_response_analysis(response))
        
        if self.settings['show_neuro'] and response.get('neurochemistry'):
            logger.info(format_neuro_state(response['neurochemistry']))

def start_advanced_cli(ai_instance):
    """
    """
    cli = AdvancedCLI(ai_instance)
    cli.run()

__all__ = ['start_advanced_cli']
