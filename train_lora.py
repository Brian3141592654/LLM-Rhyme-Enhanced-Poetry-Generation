import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import gc
import random

# ---------------- Settings ----------------
BASE_MODEL_PATH = "microsoft/phi-2"
DATASET_PATH = "lyrics2.txt"
OUTPUT_DIR = "microsoft/lora/poem_lora2"
NUM_EPOCHS = 5
BATCH_SIZE = 2
LR = 1e-4
MAX_LENGTH = 128

# ---------------- Rhyme-focused functions ----------------
def is_rhyme(word1, word2):
    """Simple rhyme detection"""
    if not word1 or not word2:
        return False
    
    # Remove punctuation and make lowercase
    word1 = ''.join(c for c in word1.lower() if c.isalnum())
    word2 = ''.join(c for c in word2.lower() if c.isalnum())
    
    if len(word1) < 2 or len(word2) < 2:
        return False
    
    # Check last 2-3 characters for rhyme
    return word1[-2:] == word2[-2:] or word1[-3:] == word2[-3:]

def detect_rhyme_pattern(lines):
    """Simple rhyme pattern detection"""
    if len(lines) < 2:
        return "A"
    
    last_words = []
    for line in lines:
        words = line.split()
        if words:
            last_words.append(words[-1].lower())
        else:
            last_words.append("")
    
    rhyme_scheme = []
    rhyme_map = {}
    current_char = 'A'
    
    for i, word in enumerate(last_words):
        found_match = False
        for char, (pattern_word, pattern_index) in rhyme_map.items():
            if is_rhyme(word, pattern_word) and abs(i - pattern_index) <= 4:  # Limit distance for rhyme
                rhyme_scheme.append(char)
                found_match = True
                break
        
        if not found_match:
            rhyme_scheme.append(current_char)
            rhyme_map[current_char] = (word, i)
            current_char = chr(ord(current_char) + 1)
    
    return ''.join(rhyme_scheme)

def calculate_rhyme_quality(lines):
    """Calculate how well the lines rhyme"""
    if len(lines) < 2:
        return 0.0
    
    last_words = []
    for line in lines:
        words = line.split()
        if words:
            last_words.append(words[-1].lower())
        else:
            last_words.append("")
    
    rhyme_score = 0.0
    valid_pairs = 0
    
    # Check common rhyme patterns
    # AABB pattern (adjacent pairs)
    for i in range(0, len(last_words) - 1, 2):
        if i + 1 < len(last_words) and is_rhyme(last_words[i], last_words[i+1]):
            rhyme_score += 1.0
            valid_pairs += 1
    
    # ABAB pattern (alternating)
    for i in range(0, len(last_words) - 2, 2):
        if i + 2 < len(last_words) and is_rhyme(last_words[i], last_words[i+2]):
            rhyme_score += 0.8
            valid_pairs += 1
    
    return rhyme_score / valid_pairs if valid_pairs > 0 else 0.0

def create_rhyme_focused_dataset(existing_texts):
    """Create dataset with explicit rhyme instructions"""
    rhyme_examples = []
    
    # Explicit rhyme instruction examples
    rhyme_instructions = [
        "###Instruction: Write a poem with an AABB rhyme scheme\n###Response: The sun shines bright (A)\nGiving us light (A)\nThe birds take flight (B)\nThroughout the night (B)",
        "###Instruction: Create a rhyming poem about nature\n###Response: The trees sway high (A)\nBeneath the sky (A)\nThe rivers flow by (B)\nAs time drifts nigh (B)",
        "###Instruction: Write a poem with perfect rhymes\n###Response: The moon above (A)\nFilled with love (A)\nA gentle dove (A)\nFrom heaven above (A)",
        "###Instruction: Compose a rhyming verse about love\n###Response: My heart beats true (A)\nOnly for you (A)\nThrough morning dew (A)\nMy love renews (A)",
        "###Instruction: Write a poem with ABAB rhyme pattern\n###Response: The wind does blow (A)\nAcross the snow (B)\nThe rivers flow (A)\nWith gentle glow (B)"
    ]
    
    # Add rhyme pattern indicators to existing examples
    for text in existing_texts:
        if "###Response:" in text:
            try:
                response_part = text.split("###Response:")[1].strip()
                lines = [line.strip() for line in response_part.split('\n') if line.strip()]
                
                if len(lines) >= 2:
                    rhyme_pattern = detect_rhyme_pattern(lines)
                    rhyme_quality = calculate_rhyme_quality(lines)
                    
                    # Only enhance if there's some rhyme quality
                    if rhyme_quality > 0.3:
                        instruction_part = text.split("###Response:")[0]
                        enhanced_text = f"{instruction_part}[Rhyme: {rhyme_pattern}] ###Response: {response_part}"
                        rhyme_examples.append(enhanced_text)
                    else:
                        # Add basic rhyme instruction for non-rhyming examples
                        enhanced_text = text.replace(
                            "###Instruction:", 
                            "###Instruction: [Add rhyming] "
                        )
                        rhyme_examples.append(enhanced_text)
                else:
                    rhyme_examples.append(text)
            except:
                rhyme_examples.append(text)
        else:
            rhyme_examples.append(text)
    
    return rhyme_instructions + rhyme_examples

def augment_with_rhymes(texts, augmentation_factor=2):
    """Augment dataset with rhyme variations"""
    augmented = []
    for text in texts:
        augmented.append(text)
        
        if "###Instruction:" in text and "###Response:" in text and random.random() < 0.6:
            # Create rhyme-focused variations
            rhyme_patterns = ["AABB", "ABAB", "ABBA", "AAAA", "AABA"]
            rhyme_pattern = random.choice(rhyme_patterns)
            
            augmented_text = text.replace(
                "###Instruction:", 
                f"###Instruction: [Use {rhyme_pattern} rhyme scheme] "
            )
            augmented.append(augmented_text)
            
            # Add another variation with explicit rhyme request
            if random.random() < 0.4:
                augmented_text2 = text.replace(
                    "###Instruction:", 
                    "###Instruction: [Focus on end rhymes] "
                )
                augmented.append(augmented_text2)
    
    return augmented

# ---------------- Load tokenizer ----------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# ---------------- Load 4-bit model with memory optimizations ----------------
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# ---------------- LoRA config ----------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.train()

# ---------------- Enhanced dataset loading ----------------
def load_and_enhance_dataset(file_path):
    """Load and enhance dataset with rhyme focus"""
    examples = []
    
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                line = line.replace("###Instuction:", "###Instruction:")
                if "###Instruction:" in line and "###Response:" in line:
                    examples.append(line)
    
    print(f"Loaded {len(examples)} base examples")
    
    # Enhance with rhyme-focused examples
    enhanced_examples = create_rhyme_focused_dataset(examples)
    
    # Augment with rhyme variations
    final_examples = augment_with_rhymes(enhanced_examples)
    
    print(f"Enhanced to {len(final_examples)} rhyme-focused examples")
    return final_examples

# Load dataset
formatted_texts = load_and_enhance_dataset(DATASET_PATH)

if len(formatted_texts) == 0:
    raise ValueError("No valid examples found in the dataset!")

# ---------------- Tokenize in batches ----------------
def tokenize_in_batches(texts, batch_size=10):
    """Tokenize in batches to avoid memory issues"""
    all_input_ids = []
    all_attention_mask = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        
        all_input_ids.append(encodings["input_ids"])
        all_attention_mask.append(encodings["attention_mask"])
        
        del encodings
        gc.collect()
        torch.cuda.empty_cache()
    
    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_mask, dim=0)
    labels = input_ids.clone()
    
    return input_ids, attention_mask, labels

print("Tokenizing dataset in batches...")
input_ids, attention_mask, labels = tokenize_in_batches(formatted_texts, batch_size=8)

# Create dataset
dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)

# Create data loader
dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

# ---------------- Rhyme-aware loss function ----------------
def rhyme_aware_loss(loss, input_ids, labels, tokenizer, weight=1.3):
    """Increase loss weight for non-rhyming outputs"""
    try:
        texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        rhyme_penalty = 0
        
        for text in texts:
            if "###Response:" in text:
                response = text.split("###Response:")[1].strip()
                lines = [line.strip() for line in response.split('\n') if line.strip()]
                
                if len(lines) >= 2:
                    rhyme_score = calculate_rhyme_quality(lines)
                    # Higher penalty for lower rhyme scores
                    rhyme_penalty += (1 - rhyme_score) * weight
        
        return loss + (rhyme_penalty / len(texts))
    except:
        return loss  # Fallback to regular loss if error occurs

# ---------------- Optimizer ----------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
gradient_accumulation_steps = 4

# ---------------- Training loop ----------------
device = model.device
model.gradient_checkpointing_enable()

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    total_loss = 0
    num_batches = 0
    
    optimizer.zero_grad()
    
    loop = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, batch in enumerate(loop):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        
        # Forward pass
        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        loss = outputs.loss / gradient_accumulation_steps
        
        # Apply rhyme-aware loss weighting
        loss = rhyme_aware_loss(loss, input_ids, labels, tokenizer, weight=1.5)
        
        # Backward pass
        loss.backward()
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        avg_loss = total_loss / num_batches
        
        loop.set_postfix(loss=avg_loss, rhyme_focus="enabled")
        
        del outputs, loss
        gc.collect()
    
    print(f"Epoch {epoch+1} done. Average loss: {avg_loss:.4f}")
    
    # Save checkpoint
    model.save_pretrained(f"{OUTPUT_DIR}_epoch_{epoch+1}")
    print(f"Checkpoint saved for epoch {epoch+1}")

# ---------------- Final save ----------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA saved at {OUTPUT_DIR}")
print(f"Training completed with {len(formatted_texts)} rhyme-enhanced examples")
