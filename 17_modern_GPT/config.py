"""
Configuration for Modern GPT
"""

class ModelConfig:
    """Model configuration"""

    def __init__(self):
        # Model dimensions
        self.vocab_size = 5000        # Vocabulary size
        self.d_model = 128            # Model dimension (small for CPU)
        self.num_heads = 4            # Number of attention heads
        self.d_ff = 512              # FFN expansion dimension
        self.num_layers = 3          # Number of decoder blocks
        self.num_kv_heads = 2        # Number of KV heads (GQA)
        self.max_len = 64            # Maximum sequence length
        self.dropout = 0.1           # Dropout rate

        # Training
        self.batch_size = 16
        self.epochs = 50
        self.learning_rate = 0.001

        # Generation
        self.max_new_tokens = 40
        self.temperature = 0.8
        self.top_k = 40


class DataConfig:
    """Data configuration"""

    def __init__(self):
        self.seq_len = 32
        self.num_samples = 10000
        self.split_ratio = 0.8


def get_default_config():
    """Get default configuration"""
    return ModelConfig()