import torch
import time
from typing import Dict, Optional
import gc

class PerformanceMetrics:
    """Lightweight performance metrics tracker for VRAM, timing, throughput, tensor Counter"""
    
    def __init__(self, track_vram=True, track_tflops=True, track_tensors=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.track_vram = track_vram
        self.track_tflops = track_tflops
        self.track_tensors = track_tensors
        
        self.start_time = None
        self.tokens_processed = 0
        self.total_flops = 0
        
        # Tensor tracking
        self.tensor_count_start = 0
        self.tensor_memory_start = 0
        
    def start(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            if self.track_vram:
                torch.cuda.reset_peak_memory_stats()
        
        self.start_time = time.time()
        self.tokens_processed = 0
        self.total_flops = 0
        
        # Track tensor count at start
        if self.track_tensors:
            if self.device.type == 'cuda':
                self.tensor_count_start = len([obj for obj in gc.get_objects() 
                                              if torch.is_tensor(obj) and obj.is_cuda])
                self.tensor_memory_start = sum(obj.element_size() * obj.nelement() 
                                               for obj in gc.get_objects() 
                                               if torch.is_tensor(obj) and obj.is_cuda) / (1024**3)
            else:
                self.tensor_count_start = len([obj for obj in gc.get_objects() 
                                              if torch.is_tensor(obj)])
                self.tensor_memory_start = 0
    
    def stop(self) -> Dict:
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - self.start_time
        
        metrics = {'elapsed_time_s': elapsed}
        
        # VRAM metrics
        if self.track_vram and self.device.type == 'cuda':
            metrics['vram_peak_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
            metrics['vram_current_gb'] = torch.cuda.memory_allocated() / (1024**3)
            metrics['vram_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
            metrics['vram_fragmentation_gb'] = (torch.cuda.memory_reserved() - 
                                                torch.cuda.memory_allocated()) / (1024**3)
        
        # Throughput metrics
        if self.tokens_processed > 0:
            metrics['tokens_per_second'] = self.tokens_processed / elapsed
            metrics['samples_per_second'] = self.tokens_processed / elapsed
            metrics['ms_per_token'] = (elapsed * 1000) / self.tokens_processed
        
        # TFLOPs calculation
        if self.track_tflops and self.total_flops > 0:
            # TFLOPs = total FLOPs / (time * 10^12)
            metrics['tflops'] = self.total_flops / (elapsed * 1e12)
            metrics['total_flops'] = self.total_flops
        
        # Tensor tracking
        if self.track_tensors:
            if self.device.type == 'cuda':
                current_tensor_count = len([obj for obj in gc.get_objects() 
                                           if torch.is_tensor(obj) and obj.is_cuda])
                current_tensor_memory = sum(obj.element_size() * obj.nelement() 
                                           for obj in gc.get_objects() 
                                           if torch.is_tensor(obj) and obj.is_cuda) / (1024**3)
            else:
                current_tensor_count = len([obj for obj in gc.get_objects() 
                                           if torch.is_tensor(obj)])
                current_tensor_memory = 0
            
            metrics['tensor_count'] = current_tensor_count
            metrics['tensors_created'] = current_tensor_count - self.tensor_count_start
            metrics['tensor_memory_gb'] = current_tensor_memory
            metrics['tensor_memory_delta_gb'] = current_tensor_memory - self.tensor_memory_start
        
        return metrics
    
    def log_tokens(self, num_tokens: int):
        self.tokens_processed += num_tokens
    
    def estimate_flops(self, model, batch_size: int, seq_len: int):
        # Estimate FLOPs for a forward + backward pass. Call this after each training step.
        if not self.track_tflops:
            return
        
        config = model.config
        
        # Get model dimensions - handle different model architectures
        hidden_size = getattr(config, 'hidden_size', getattr(config, 'n_embd', 768))
        num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 12))
        num_heads = getattr(config, 'num_attention_heads', getattr(config, 'n_head', 12))
        vocab_size = getattr(config, 'vocab_size', 50257)
        
        # FFN size - handle None case
        ffn_size = getattr(config, 'intermediate_size', None)
        if ffn_size is None:
            ffn_size = 4 * hidden_size  # Standard transformer ratio
        
        # Per-layer FLOPs calculation
        # 1. Attention: Q, K, V, O projections
        attention_flops = 4 * (2 * batch_size * seq_len * hidden_size * hidden_size)
        
        # 2. Attention scores: Q @ K^T
        attention_flops += 2 * batch_size * num_heads * seq_len * seq_len * (hidden_size // num_heads)
        
        # 3. Attention output: scores @ V
        attention_flops += 2 * batch_size * num_heads * seq_len * seq_len * (hidden_size // num_heads)
        
        # 4. FFN: two linear layers (up and down projection)
        ffn_flops = 2 * (2 * batch_size * seq_len * hidden_size * ffn_size)
        
        # Total per layer
        layer_flops = attention_flops + ffn_flops
        
        # All layers
        forward_flops = num_layers * layer_flops
        
        # Add embedding and final projection
        forward_flops += 2 * batch_size * seq_len * hidden_size * vocab_size
        
        # Backward pass is approximately 2x forward pass
        total_flops = 3 * forward_flops
        
        self.total_flops += total_flops
    
    @staticmethod
    def get_model_stats(model) -> Dict:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        # Memory size
        model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
        trainable_size_gb = sum(
            p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
        ) / (1024**3)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'trainable_percent': 100 * trainable_params / total_params,
            'model_size_gb': model_size_gb,
            'trainable_size_gb': trainable_size_gb,
            'parameter_reduction': f"{100 * (1 - trainable_params/total_params):.2f}%"
        }
    
    def print_summary(self, epoch: int, metrics: Dict, model_stats: Dict = None):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch} Performance Summary")
        print(f"{'='*70}")
        
        if model_stats:
            print(f"Trainable Params: {model_stats['trainable_params']:,} "
                  f"({model_stats['trainable_percent']:.4f}%)")
            print(f"Memory Savings: {model_stats['parameter_reduction']}")
        
        print(f"Training Time: {metrics['elapsed_time_s']:.2f}s")
        
        if 'vram_peak_gb' in metrics:
            print(f"VRAM Peak: {metrics['vram_peak_gb']:.2f} GB")
            print(f"VRAM Current: {metrics['vram_current_gb']:.2f} GB")
            print(f"VRAM Fragmentation: {metrics['vram_fragmentation_gb']:.2f} GB")
        
        if 'tokens_per_second' in metrics:
            print(f"Throughput: {metrics['tokens_per_second']:.2f} tokens/s")
        
        if 'tflops' in metrics:
            print(f"TFLOPs: {metrics['tflops']:.4f}")
        
        if 'tensor_count' in metrics:
            print(f"Tensors: {metrics['tensor_count']} "
                  f"(Δ{metrics['tensors_created']:+d})")
            print(f"Tensor Memory: {metrics['tensor_memory_gb']:.2f} GB "
                  f"(Δ{metrics['tensor_memory_delta_gb']:+.2f} GB)")
        
        print(f"{'='*70}\n")


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m").cuda()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
    
    model_stats = PerformanceMetrics.get_model_stats(model)
    print("\n=== Model Configuration ===")
    for key, value in model_stats.items():
        print(f"{key}: {value}")
    
    tracker = PerformanceMetrics(
        track_vram=True,
        track_tflops=True,
        track_tensors=True
    )
    
    EPOCHS = 3
    BATCH_SIZE = 2
    
    for epoch in range(EPOCHS):
        tracker.start()
        
        for step in range(100):
            text = "Hello world " * 20
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
            
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            loss.backward()
            
            seq_len = inputs.input_ids.shape[1]
            tracker.log_tokens(BATCH_SIZE * seq_len)
            tracker.estimate_flops(model, BATCH_SIZE, seq_len)
        
        metrics = tracker.stop()
        tracker.print_summary(epoch, metrics, model_stats)