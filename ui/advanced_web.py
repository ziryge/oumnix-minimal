import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import time
import json
from datetime import datetime, timedelta

class AdvancedWebInterface:
    def __init__(self, ai_instance):
        self.ai = ai_instance
        self.conversation_history = []
        self.performance_history = []
        self.neuro_history = []
        
        self.update_interval = 5  
        self.max_history_points = 100
        
    def chat_response(self, message: str, history: List[List[str]]) -> Tuple[str, List[List[str]], str]:
        if not message.strip():
            return "", history, ""
        
        response = self.ai.process_input(message)
        
        history.append([message, response['text']])
        
        self.conversation_history.append({
            'timestamp': time.time(),
            'user': message,
            'ai': response['text'],
            'confidence': response['confidence'],
            'processing_time': response['processing_time']
        })
        
        analysis = self._format_response_analysis(response)
        
        return "", history, analysis
    
    def _format_response_analysis(self, response: Dict[str, Any]) -> str:
        confidence = response.get('confidence', 0.0)
        processing_time = response.get('processing_time', 0.0)
        reasoning = response.get('reasoning_program', [])
        
        analysis = f"""
        """
        return analysis
    
    def get_system_status(self) -> str:
        status = self.ai.get_system_status()
        status_text = f"""
        """
        return status_text
    
    def get_neuro_state(self) -> Tuple[str, go.Figure]:
        if not self.ai.neurochemistry:
            return "Neurochemical system disabled", go.Figure()
        
        analysis = self.ai.neurochemistry.analyze_state()
        current = analysis.get('current_state', {})
        modulated = analysis.get('modulated_params', {})
        mood = analysis.get('interpreted_mood', 'neutral')
        
        state_text = f"""
        """
        
        neurotransmitters = ['Dopamine', 'Serotonin', 'Noradrenaline', 'Acetylcholine']
        values = [
            current.get('dopamine', 0.5),
            current.get('serotonin', 0.5),
            current.get('noradrenaline', 0.5),
            current.get('acetylcholine', 0.5)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  
            theta=neurotransmitters + [neurotransmitters[0]],
            fill='toself',
            name='Current Levels',
            line_color='rgb(0, 123, 255)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Neurochemical State",
            height=400
        )
        
        self.neuro_history.append({
            'timestamp': time.time(),
            **current
        })
        
        if len(self.neuro_history) > self.max_history_points:
            self.neuro_history = self.neuro_history[-self.max_history_points:]
        
        return state_text, fig
    
    def get_memory_stats(self) -> Tuple[str, go.Figure]:
        memory_stats = self.ai.memory_system.get_stats()
        
        stats_text = f"""
        """
        
        labels = ['Hot Cache', 'Warm Cache', 'Cold Cache']
        values = [
            memory_stats.get('hot_hits', 0),
            memory_stats.get('warm_hits', 0),
            memory_stats.get('cold_hits', 0)
        ]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3
        )])
        
        fig.update_layout(
            title="Cache Hit Distribution",
            height=400
        )
        
        return stats_text, fig
    
    def get_performance_chart(self) -> go.Figure:
        if len(self.conversation_history) < 2:
            fig = go.Figure()
            fig.update_layout(title="Performance Over Time (insufficient data)")
            return fig
        
        recent_history = self.conversation_history[-50:]  
        
        timestamps = [datetime.fromtimestamp(h['timestamp']) for h in recent_history]
        processing_times = [h['processing_time'] * 1000 for h in recent_history]  
        confidences = [h['confidence'] for h in recent_history]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=processing_times,
            mode='lines+markers',
            name='Processing Time (ms)'
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=[c * 100 for c in confidences],  
            mode='lines+markers',
            name='Confidence (%)',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Performance Over Time',
            xaxis_title='Time',
            yaxis=dict(
                title='Processing Time (ms)',
                side='left'
            ),
            yaxis2=dict(
                title='Confidence (%)',
                side='right',
                overlaying='y'
            ),
            height=400
        )
        
        return fig
    
    def get_neuro_timeline(self) -> go.Figure:
        if len(self.neuro_history) < 2:
            fig = go.Figure()
            fig.update_layout(title="Neurochemical Evolution (insufficient data)")
            return fig
        
        timestamps = [datetime.fromtimestamp(h['timestamp']) for h in self.neuro_history]
        
        fig = go.Figure()
        
        neurotransmitters = {
            'dopamine': ('Dopamine', 'blue'),
            'serotonin': ('Serotonin', 'green'),
            'noradrenaline': ('Noradrenaline', 'red'),
            'acetylcholine': ('Acetylcholine', 'purple')
        }
        
        for key, (name, color) in neurotransmitters.items():
            values = [h.get(key, 0.5) for h in self.neuro_history]
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=values,
                mode='lines',
                name=name,
                line=dict(color=color, width=2)
            ))
        
        fig.update_layout(
            title='Neurotransmitter Evolution',
            xaxis_title='Time',
            yaxis_title='Level (0-1)',
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        return fig
    
    def force_consolidation(self) -> str:
        self.ai.force_consolidation()
        return "Consolidation successfully started!"
    
    def save_state(self) -> str:
        try:
            self.ai.save_state()
            return "State successfully saved!"
        except Exception as e:
            return f"Error while saving: {e}"
    
    def reset_neurochemistry(self) -> str:
        if self.ai.neurochemistry:
            self.ai.reset_neurochemistry()
            return "Neurochemistry reset to baseline!"
        else:
            return "Neurochemical system disabled"
    
    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(
            title="Oumnix Agent",
            theme=gr.themes.Soft(),
            css=""
        ) as interface:
            
            gr.Markdown("")
            
            with gr.Tabs():
                with gr.TabItem("Chat"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            chatbot = gr.Chatbot(
                                label="Conversation",
                                height=500,
                                show_label=True
                            )
                            
                            with gr.Row():
                                msg_input = gr.Textbox(
                                    placeholder="Type your message...",
                                    label="Message",
                                    scale=4
                                )
                                send_btn = gr.Button("Send", scale=1, variant="primary")
                        
                        with gr.Column(scale=1):
                            analysis_output = gr.Markdown(
                                label="Response Analysis",
                                value="Waiting for message..."
                            )
                    
                    def chat_fn(message, history):
                        return self.chat_response(message, history)
                    
                    send_btn.click(
                        chat_fn,
                        inputs=[msg_input, chatbot],
                        outputs=[msg_input, chatbot, analysis_output]
                    )
                    
                    msg_input.submit(
                        chat_fn,
                        inputs=[msg_input, chatbot],
                        outputs=[msg_input, chatbot, analysis_output]
                    )
                
                with gr.TabItem("System"):
                    with gr.Row():
                        with gr.Column():
                            status_output = gr.Markdown(
                                label="System Status",
                                value=self.get_system_status()
                            )
                            
                            with gr.Row():
                                refresh_btn = gr.Button("Refresh", variant="secondary")
                                consolidate_btn = gr.Button("Consolidate", variant="primary")
                                save_btn = gr.Button("Save", variant="primary")
                        
                        with gr.Column():
                            performance_chart = gr.Plot(
                                label="Performance",
                                value=self.get_performance_chart()
                            )
                    
                    refresh_btn.click(
                        self.get_system_status,
                        outputs=[status_output]
                    )
                    
                    consolidate_btn.click(
                        self.force_consolidation,
                        outputs=[gr.Textbox(visible=False)]  
                    )
                    
                    save_btn.click(
                        self.save_state,
                        outputs=[gr.Textbox(visible=False)]  
                    )
                
                with gr.TabItem("Neurochemistry"):
                    with gr.Row():
                        with gr.Column():
                            neuro_status = gr.Markdown(
                                label="Neurochemical State",
                                value="Loading..."
                            )
                            
                            reset_neuro_btn = gr.Button("Reset Neurochemistry", variant="secondary")
                        
                        with gr.Column():
                            neuro_radar = gr.Plot(
                                label="Neuro Radar",
                                value=go.Figure()
                            )
                    
                    with gr.Row():
                        neuro_timeline = gr.Plot(
                            label="Timeline",
                            value=go.Figure()
                        )
                    
                    def update_neuro():
                        status, radar = self.get_neuro_state()
                        timeline = self.get_neuro_timeline()
                        return status, radar, timeline
                    
                    reset_neuro_btn.click(
                        self.reset_neurochemistry,
                        outputs=[gr.Textbox(visible=False)]  
                    )
                
                with gr.TabItem("Memory"):
                    with gr.Row():
                        with gr.Column():
                            memory_status = gr.Markdown(
                                label="Memory Stats",
                                value="Loading..."
                            )
                        
                        with gr.Column():
                            memory_chart = gr.Plot(
                                label="Cache Hits",
                                value=go.Figure()
                            )
                    
                    def update_memory():
                        return self.get_memory_stats()
            
            interface.load(
                lambda: (
                    self.get_system_status(),
                    *self.get_neuro_state(),
                    self.get_neuro_timeline(),
                    *self.get_memory_stats(),
                    self.get_performance_chart()
                ),
                outputs=[
                    status_output,
                    neuro_status,
                    neuro_radar,
                    neuro_timeline,
                    memory_status,
                    memory_chart,
                    performance_chart
                ]
            )
        
        return interface

def start_advanced_web(ai_instance, port: int = 7860, share: bool = False):
    web_interface = AdvancedWebInterface(ai_instance)
    interface = web_interface.create_interface()
    
    print(f"Starting web interface on port {port}")
    print(f"   URL: http://localhost:{port}")
    
    interface.launch(
        server_port=port,
        share=share,
        show_error=True,
        quiet=False
    )

__all__ = ['start_advanced_web']
