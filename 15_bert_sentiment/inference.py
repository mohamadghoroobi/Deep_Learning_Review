import numpy as np
import pickle
from data import build_vocab
from model import BERTEncoder
from layers.loss import softmax


# ============================================
# Load the trained model
# ============================================
def load_model(model_path='bert_sentiment.pkl'):
    """Load the trained BERT model"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully from {model_path}")

        # Set to evaluation mode (disables dropout)
        model.training = False
        for layer in model.layers:
            layer.dropout1.training = False
            layer.dropout2.training = False

        return model
    except FileNotFoundError:
        print(f"❌ Error: Model file '{model_path}' not found!")
        print("   Please train the model first with train.py or temp.py")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


# ============================================
# Tokenize and prepare input text
# ============================================
def prepare_input(text, vocab, max_len=20):
    """
    Convert raw text to token IDs ready for BERT

    Args:
        text: String review
        vocab: Vocabulary dictionary
        max_len: Maximum sequence length

    Returns:
        token_ids: List of token IDs (padded to max_len)
    """
    # Convert to lowercase and split into words
    words = text.lower().split()

    # Start with [CLS] token
    token_ids = [vocab['CLS']]  # 1

    # Convert each word to token ID
    for word in words:
        # Remove punctuation
        word = word.strip('.,!?;:()"\'')

        if word in vocab:
            token_ids.append(vocab[word])
        else:
            token_ids.append(vocab['UNK'])  # Unknown token

    # Pad to max_len
    if len(token_ids) < max_len:
        token_ids += [vocab['PAD']] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]

    return token_ids


# ============================================
# Predict sentiment
# ============================================
def predict_sentiment(model, text, vocab, max_len=20, return_probs=False):
    """
    Predict sentiment of a text using trained BERT model

    Args:
        model: Trained BERTEncoder model
        text: String review to analyze
        vocab: Vocabulary dictionary
        max_len: Maximum sequence length
        return_probs: If True, return full probability distribution

    Returns:
        sentiment: "Positive" or "Negative"
        confidence: Confidence score (0-1)
        probs: (Optional) Full probability distribution
    """
    # Prepare input
    token_ids = prepare_input(text, vocab, max_len)

    # Convert to batch (1, max_len)
    x = np.array([token_ids], dtype=np.int32)

    # Forward pass
    logits, _ = model.forward(x)

    # Get probabilities
    probs = softmax(logits[0])  # (2,)

    # Determine sentiment
    positive_prob = probs[1]
    if positive_prob > 0.5:
        sentiment = "Positive"
        confidence = positive_prob
    else:
        sentiment = "Negative"
        confidence = 1 - positive_prob  # Probability of negative

    if return_probs:
        return sentiment, confidence, probs
    return sentiment, confidence


# ============================================
# Analyze multiple reviews
# ============================================
def batch_predict(model, texts, vocab, max_len=20):
    """Predict sentiment for multiple texts"""
    results = []
    for text in texts:
        sentiment, confidence = predict_sentiment(model, text, vocab, max_len)
        results.append({
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence
        })
    return results


# ============================================
# Main execution
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("BERT SENTIMENT ANALYSIS - INFERENCE")
    print("=" * 60)

    # Load the model
    model = load_model('bert_sentiment.pkl')
    if model is None:
        exit(1)

    # Build vocabulary
    vocab = build_vocab()
    print(f"📚 Vocabulary size: {len(vocab)} tokens")
    print("")

    # ============================================
    # Test the model on reviews
    # ============================================
    test_reviews = [
        "This movie is absolutely amazing and wonderful",
        "The plot is boring and the acting is terrible",
        "Good acting, great story, I really enjoyed it",
        "The worst film I have ever seen, complete waste of time",
        "Beautiful cinematography and excellent performances",
        "Poor writing, disappointing ending, avoid this movie",
        "I loved this film, the characters were brilliant",
        "Hated it, waste of money and time",
        "The movie was okay, nothing special",
        "Fantastic performances from the entire cast",
        "Disappointing and dull, I fell asleep",
        "A masterpiece of modern cinema"
    ]

    print("📊 Analyzing Reviews...")
    print("-" * 60)

    # Analyze all reviews
    results = batch_predict(model, test_reviews, vocab, max_len=20)

    # Display results with colors (if supported)
    print("RESULTS:")
    print("-" * 60)

    for i, result in enumerate(results, 1):
        text = result['text']
        sentiment = result['sentiment']
        confidence = result['confidence']

        # Emoji based on sentiment
        emoji = "😊" if sentiment == "Positive" else "😞"

        # Format confidence as percentage
        conf_pct = f"{confidence:.2%}"

        # Truncate long reviews for display
        if len(text) > 50:
            display_text = text[:47] + "..."
        else:
            display_text = text

        print(f"{i:2d}. {emoji} {sentiment:8s} [{conf_pct:6s}]  {display_text}")

    print("-" * 60)

    # ============================================
    # Interactive mode
    # ============================================
    print("\n🔄 Interactive Mode")
    print("   Type a review to analyze, or 'quit' to exit")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n📝 Enter a review: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            if not user_input:
                continue

            # Predict
            sentiment, confidence, probs = predict_sentiment(
                model, user_input, vocab, max_len=20, return_probs=True
            )

            # Display result
            emoji = "😊" if sentiment == "Positive" else "😞"
            print(f"\n{emoji} Sentiment: {sentiment}")
            print(f"   Confidence: {confidence:.2%}")
            print(f"   Raw scores: Positive={probs[1]:.3f}, Negative={probs[0]:.3f}")
            print("   " + "=" * 30)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")