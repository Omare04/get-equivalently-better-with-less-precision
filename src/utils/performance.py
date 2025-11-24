import torch
import time
from typing import Dict

class PerformanceMetrics:
    """Lightweight performance metrics tracker for VRAM, timing, and throughput"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_time = None
        self.tokens_processed = 0
        
    def start(self):
        """Start timing and reset peak memory stats"""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.start_time = time.time()
        self.tokens_processed = 0
    
    def stop(self) -> Dict:
        """Stop timing and return metrics"""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - self.start_time
        
        metrics = {
            'elapsed_time_s': elapsed,
        }
        
        # VRAM metrics (only on CUDA)
        if self.device.type == 'cuda':
            metrics['vram_peak_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
            metrics['vram_current_gb'] = torch.cuda.memory_allocated() / (1024**3)
            metrics['vram_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
        
        # Throughput metrics (if tokens were tracked)
        if self.tokens_processed > 0:
            metrics['tokens_per_second'] = self.tokens_processed / elapsed
            metrics['ms_per_token'] = (elapsed * 1000) / self.tokens_processed
        
        return metrics
    
    def log_tokens(self, num_tokens: int):
        """Track number of tokens processed"""
        self.tokens_processed += num_tokens
    
    @staticmethod
    def get_model_memory(model) -> Dict:
        """Get model size in memory"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'model_size_gb': param_size,
            'total_params': param_count,
            'trainable_params': trainable_params,
            'trainable_percent': 100 * trainable_params / param_count if param_count > 0 else 0
        }


# Example usage with your training code
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m").cuda()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
    
    # Get model stats
    model_stats = PerformanceMetrics.get_model_memory(model)
    print("Model Stats:", model_stats)
    
    # Track training step
    metrics = PerformanceMetrics()
    metrics.start()
    
    # Simulate training
    text = "Hello world, this is a test"
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    for _ in range(10):
        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss
        loss.backward()
        metrics.log_tokens(inputs.input_ids.shape[1])
    
    results = metrics.stop()
    print("\nPerformance Metrics:", results)