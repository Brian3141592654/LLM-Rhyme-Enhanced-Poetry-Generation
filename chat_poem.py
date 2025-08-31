import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re

# ---------------- Settings ----------------
BASE_MODEL_PATH = "microsoft/phi-2"
LORA_PATH = "microsoft/lora/poem_lora2"
TEST_PROMPTS = [
    "Write a poem about love with perfect rhymes",
    "Create a rhyming poem about nature",
    "Compose a poem with an AABB rhyme scheme about the ocean",
    "Write a rhyming verse about friendship",
    "Create a poem with ABAB rhyme pattern about seasons"
]

# ---------------- Load model and tokenizer ----------------
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()
print("Model loaded successfully!")

# ---------------- Rhyme evaluation functions ----------------
def is_rhyme(word1, word2):
    """Check if two words rhyme"""
    if not word1 or not word2:
        return False
    
    # Clean words
    word1 = ''.join(c for c in word1.lower() if c.isalnum())
    word2 = ''.join(c for c in word2.lower() if c.isalnum())
    
    if len(word1) < 2 or len(word2) < 2:
        return False
    
    # Check last 2-3 characters for rhyme
    return word1[-2:] == word2[-2:] or word1[-3:] == word2[-3:]

def extract_last_words(text):
    """Extract last words from each line of a poem"""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    last_words = []
    
    for line in lines:
        words = line.split()
        if words:
            # Remove punctuation from last word
            last_word = words[-1].rstrip('.,!?;:()[]{}"\'')
            last_words.append(last_word.lower())
        else:
            last_words.append("")
    
    return last_words

def calculate_rhyme_score(text):
    """Calculate rhyme quality score (0-1)"""
    last_words = extract_last_words(text)
    
    if len(last_words) < 2:
        return 0.0
    
    rhyme_pairs = 0
    total_possible_pairs = 0
    
    # Check adjacent lines (AABB pattern)
    for i in range(0, len(last_words) - 1, 2):
        if i + 1 < len(last_words):
            if is_rhyme(last_words[i], last_words[i+1]):
                rhyme_pairs += 1
            total_possible_pairs += 1
    
    # Check alternating lines (ABAB pattern)
    for i in range(0, len(last_words) - 2, 2):
        if i + 2 < len(last_words):
            if is_rhyme(last_words[i], last_words[i+2]):
                rhyme_pairs += 0.8  # Slightly lower weight for alternating
            total_possible_pairs += 0.8
    
    # Check every other pair
    for i in range(0, len(last_words) - 1):
        if i + 1 < len(last_words):
            if is_rhyme(last_words[i], last_words[i+1]):
                rhyme_pairs += 0.6
            total_possible_pairs += 0.6
    
    return rhyme_pairs / total_possible_pairs if total_possible_pairs > 0 else 0.0

def detect_rhyme_scheme(text):
    """Detect and return the rhyme scheme"""
    last_words = extract_last_words(text)
    
    if len(last_words) < 2:
        return "No rhyme detected"
    
    scheme = []
    rhyme_map = {}
    current_char = 'A'
    
    for i, word in enumerate(last_words):
        found = False
        for char, (pattern_word, pattern_index) in rhyme_map.items():
            if is_rhyme(word, pattern_word) and abs(i - pattern_index) <= 3:
                scheme.append(char)
                found = True
                break
        
        if not found:
            scheme.append(current_char)
            rhyme_map[current_char] = (word, i)
            current_char = chr(ord(current_char) + 1)
    
    return ''.join(scheme)

def count_syllables(word):
    """Approximate syllable count"""
    word = word.lower()
    if len(word) <= 3:
        return 1
    
    vowels = "aeiouy"
    count = 0
    prev_char_vowel = False
    
    for char in word:
        if char in vowels:
            if not prev_char_vowel:
                count += 1
            prev_char_vowel = True
        else:
            prev_char_vowel = False
    
    return max(1, count)

def calculate_rhythm_consistency(text):
    """Check rhythm consistency by syllable count"""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    if len(lines) < 2:
        return 0.0
    
    syllable_counts = []
    for line in lines:
        words = line.split()
        line_syllables = sum(count_syllables(word) for word in words)
        syllable_counts.append(line_syllables)
    
    # Check if lines have similar syllable counts
    avg_syllables = sum(syllable_counts) / len(syllable_counts)
    variance = sum((count - avg_syllables) ** 2 for count in syllable_counts) / len(syllable_counts)
    
    # Lower variance = better rhythm consistency
    return 1 / (1 + variance)

# ---------------- Generation function ----------------
def generate_poem(prompt, max_length=150, temperature=0.8, top_p=0.9):
    """Generate a poem from prompt"""
    full_prompt = f"###Instruction: {prompt}\n###Response:"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    if "###Response:" in generated_text:
        response = generated_text.split("###Response:")[1].strip()
    else:
        response = generated_text
    
    return response

# ---------------- Test function ----------------
def test_model():
    """Test the model with various prompts"""
    print("=" * 60)
    print("TESTING RHYME-ENHANCED POEM GENERATION")
    print("=" * 60)
    
    results = []
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n\n{'='*50}")
        print(f"TEST {i}: {prompt}")
        print(f"{'='*50}")
        
        # Generate poem
        poem = generate_poem(prompt)
        print(f"\nGENERATED POEM:\n")
        print(poem)
        print("\n" + "-" * 40)
        
        # Analyze rhyme
        rhyme_score = calculate_rhyme_score(poem)
        rhyme_scheme = detect_rhyme_scheme(poem)
        rhythm_score = calculate_rhythm_consistency(poem)
        
        print(f"Rhyme Scheme: {rhyme_scheme}")
        print(f"Rhyme Quality Score: {rhyme_score:.3f}/1.0")
        print(f"Rhythm Consistency: {rhythm_score:.3f}/1.0")
        print(f"Overall Quality: {(rhyme_score * 0.7 + rhythm_score * 0.3):.3f}/1.0")
        
        # Show rhyming words
        last_words = extract_last_words(poem)
        print(f"Last words: {last_words}")
        
        results.append({
            'prompt': prompt,
            'poem': poem,
            'rhyme_score': rhyme_score,
            'rhyme_scheme': rhyme_scheme,
            'rhythm_score': rhythm_score,
            'last_words': last_words
        })
    
    return results

# ---------------- Batch testing ----------------
def batch_test(num_tests=3):
    """Run multiple tests and provide statistics"""
    print("Running batch test...")
    
    all_results = []
    for test_run in range(num_tests):
        print(f"\n\n{'#'*60}")
        print(f"BATCH TEST RUN {test_run + 1}/{num_tests}")
        print(f"{'#'*60}")
        
        results = test_model()
        all_results.extend(results)
    
    # Calculate statistics
    rhyme_scores = [r['rhyme_score'] for r in all_results]
    rhythm_scores = [r['rhythm_score'] for r in all_results]
    overall_scores = [r['rhyme_score'] * 0.7 + r['rhythm_score'] * 0.3 for r in all_results]
    
    print(f"\n\n{'*'*60}")
    print("BATCH TEST SUMMARY")
    print(f"{'*'*60}")
    print(f"Total tests: {len(all_results)}")
    print(f"Average Rhyme Score: {sum(rhyme_scores)/len(rhyme_scores):.3f}/1.0")
    print(f"Average Rhythm Score: {sum(rhythm_scores)/len(rhythm_scores):.3f}/1.0")
    print(f"Average Overall Score: {sum(overall_scores)/len(overall_scores):.3f}/1.0")
    print(f"Best Rhyme Score: {max(rhyme_scores):.3f}")
    print(f"Worst Rhyme Score: {min(rhyme_scores):.3f}")
    
    # Show best and worst examples
    best_idx = overall_scores.index(max(overall_scores))
    worst_idx = overall_scores.index(min(overall_scores))
    
    print(f"\nBEST EXAMPLE (Score: {overall_scores[best_idx]:.3f}):")
    print(f"Prompt: {all_results[best_idx]['prompt']}")
    print(f"Poem:\n{all_results[best_idx]['poem']}")
    print(f"Rhyme Scheme: {all_results[best_idx]['rhyme_scheme']}")
    
    print(f"\nWORST EXAMPLE (Score: {overall_scores[worst_idx]:.3f}):")
    print(f"Prompt: {all_results[worst_idx]['prompt']}")
    print(f"Poem:\n{all_results[worst_idx]['poem']}")
    print(f"Rhyme Scheme: {all_results[worst_idx]['rhyme_scheme']}")

# ---------------- Interactive testing ----------------
def interactive_test():
    """Interactive testing mode"""
    print("Interactive testing mode. Type 'quit' to exit.")
    print("Enter your poem prompts:")
    
    while True:
        prompt = input("\nYour prompt: ").strip()
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if prompt:
            poem = generate_poem(prompt)
            print(f"\nGENERATED POEM:\n")
            print(poem)
            
            # Analyze
            rhyme_score = calculate_rhyme_score(poem)
            rhyme_scheme = detect_rhyme_scheme(poem)
            rhythm_score = calculate_rhythm_consistency(poem)
            
            print(f"\nANALYSIS:")
            print(f"Rhyme Scheme: {rhyme_scheme}")
            print(f"Rhyme Quality: {rhyme_score:.3f}/1.0")
            print(f"Rhythm Consistency: {rhythm_score:.3f}/1.0")
            print(f"Overall: {(rhyme_score * 0.7 + rhythm_score * 0.3):.3f}/1.0")

# ---------------- Main execution ----------------
if __name__ == "__main__":
    print("Choose testing mode:")
    print("1. Standard test with predefined prompts")
    print("2. Batch test (multiple runs)")
    print("3. Interactive test")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        test_model()
    elif choice == "2":
        batch_test(num_tests=2)
    elif choice == "3":
        interactive_test()
    else:
        print("Running standard test...")
        test_model()
