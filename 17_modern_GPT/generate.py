"""
Generation script for Modern GPT with KV-cache
"""

import numpy as np
import pickle
from config import ModelConfig
from model import ModernGPT
from data import build_vocab
from layers.loss import softmax


class Generator:
    """Text generator with Modern GPT"""

    def __init__(self, model_path='modern_gpt_best.pkl'):
        """Load model"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        self.model.eval_mode()
        self.vocab = build_vocab()
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        print(f"✅ Loaded model from {model_path}")
        print(f"   Vocab size: {len(self.vocab)}")

    def generate(self, prompt_tokens, max_new_tokens=20, temperature=0.8, top_k=40):
        """
        Generate tokens autoregressively

        Args:
            prompt_tokens: List of token IDs
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-K sampling (0 = disabled)

        Returns:
            List of generated tokens
        """
        # Start with prompt
        generated = list(prompt_tokens)

        # Limit to max_len
        max_len = self.model.max_len

        # Generate
        for _ in range(max_new_tokens):
            # Prepare input
            x = np.array([generated[-max_len:]])

            # Forward pass
            logits = self.model.forward(x)

            # Get logits for last token
            last_logits = logits[0, -1, :]  # (V,)

            # Temperature scaling
            if temperature > 0:
                last_logits = last_logits / temperature

            # Top-K sampling
            if top_k > 0:
                top_k_indices = np.argsort(last_logits)[-top_k:]
                mask = np.zeros_like(last_logits)
                mask[top_k_indices] = 1
                last_logits = last_logits * mask - 1e9 * (1 - mask)

            # Softmax
            probs = softmax(last_logits)

            # Sample
            next_token = np.random.choice(len(probs), p=probs)

            # Stop at EOS
            if next_token == self.vocab['EOS']:
                generated.append(next_token)
                break

            generated.append(next_token)

        return generated

    def generate_text(self, prompt_text, max_new_tokens=20, temperature=0.8, top_k=40):
        """
        Generate from text prompt

        Args:
            prompt_text: String prompt
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-K sampling

        Returns:
            Generated text
        """
        # Tokenize prompt
        words = prompt_text.lower().split()
        prompt_tokens = [self.vocab['BOS']]
        for word in words:
            if word in self.vocab:
                prompt_tokens.append(self.vocab[word])
            else:
                prompt_tokens.append(self.vocab['UNK'])

        # Generate
        tokens = self.generate(prompt_tokens, max_new_tokens, temperature, top_k)

        # Convert to text
        words = []
        for t in tokens:
            if t == self.vocab['BOS']:
                continue
            if t == self.vocab['EOS']:
                break
            if t == self.vocab['PAD']:
                continue
            if t == self.vocab['UNK']:
                words.append('[UNK]')
            else:
                words.append(self.inv_vocab[t])

        return ' '.join(words)


def generate_interactive(model_path='modern_gpt_best.pkl'):
    """Interactive generation"""
    generator = Generator(model_path)

    print("\n" + "=" * 60)
    print("MODERN GPT - INTERACTIVE GENERATION")
    print("=" * 60)
    print("Type 'quit' to exit")
    print("-" * 60)

    while True:
        try:
            prompt = input("\n📝 Prompt: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            if not prompt:
                continue

            # Generate
            text = generator.generate_text(
                prompt,
                max_new_tokens=30,
                temperature=0.8,
                top_k=40
            )

            print(f"\n🤖 {text}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Test generation
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        generate_interactive()
    else:
        # Simple test
        generator = Generator('modern_gpt_best.pkl')

        test_prompts = [
            "one two three",
            "zero one two",
            "five four three"
        ]

        for prompt in test_prompts:
            text = generator.generate_text(prompt, max_new_tokens=10, temperature=0.5)
            print(f"Prompt: {prompt}")
            print(f"Generated: {text}")
            print("-" * 40)