# Rhyme-Enhanced Poetry Generation with Phi-2 and LoRA

A sophisticated fine-tuning pipeline that transforms Microsoft's Phi-2 model into a specialized poetry generator with enhanced rhyming capabilities using Low-Rank Adaptation (LoRA).

---

## 🎯 Features

### Core Capabilities
- Rhyme-Optimized Generation: Produces poems with structured rhyme schemes (AABB, ABAB, ABBA, etc.)
- Quality Assessment: Built-in rhyme and rhythm scoring system
- Multiple Testing Modes: Standard, batch, and interactive testing
- Memory Efficient: 4-bit quantization and gradient checkpointing

### Advanced Rhyme Features
- Automatic rhyme scheme detection and labeling
- Rhyme-aware loss function during training
- Rhyme pattern augmentation in dataset
- Rhythm consistency analysis
- Syllable counting for meter analysis

---

## 🏗️ Architecture

### Model Components
- Base Model: Microsoft Phi-2 (2.7B parameters)
- Adaptation Method: LoRA (Low-Rank Adaptation)
- Quantization: 4-bit for memory efficiency
- Target Modules: Query and Value projections

### Training Enhancements
- Rhyme pattern detection (AABB, ABAB, etc.)
- Rhyme quality scoring (0–1 scale)
- Dataset augmentation with rhyme instructions
- Rhyme-aware loss weighting

---

## 📊 Performance Metrics

- Rhyme Quality Score (0–1)
- Rhythm Consistency (0–1)
- Overall Quality: 70% rhyme + 30% rhythm

---

## 🚀 Quick Start

### Installation
```bash
pip install torch transformers peft accelerate tqdm
```

### Training
```bash
python train_rhyme_poem.py
```

### Testing
```bash
python test_rhyme_poem.py
```

---

## 💌 Example Poems

```
### Instruction: write a love poem
### Response:
A fragile bloom, our love takes hold,
A story whispered, brave and bold.

### Instruction: write a love poem
### Response:
With every breath, a silent vow,
Forever bound, then, here, and now.

### Instruction: write a love poem
### Response:
Your presence calms my restless mind,
A peace within your love I find.
```

