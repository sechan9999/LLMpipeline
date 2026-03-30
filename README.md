# 🚀 Senior NLP Pipeline — Production-Grade Scalable Architecture

> A **Senior Data Scientist III**-level NLP pipeline demonstrating production-grade scalability, modular architecture, and advanced deep learning patterns using **TensorFlow/Keras**.

---

## 🎯 Purpose

This repository is designed to demonstrate the architectural maturity expected at the **Senior Data Scientist III** level — moving well beyond tutorials into the realm of scalable, production-ready machine learning systems.

It is specifically tailored for demonstrating expertise in roles requiring:
- Large-scale NLP system design
- Performance-optimized data ingestion pipelines
- Custom model architectures and training loops

---

## 📁 Project Structure

```
nlp-pipeline/
├── data/                   # Raw and processed datasets
├── src/
│   ├── pipeline.py         # tf.data input pipeline with AUTOTUNE
│   ├── model.py            # Bi-LSTM + Custom Attention architecture
│   └── train.py            # Custom training loop with gradient clipping
├── app.py                  # Interactive Streamlit dashboard (local server)
├── requirements.txt        # Production dependencies
└── README.md               # You are here
```

---

## 🏗️ Architecture Overview

```
Text Input
    │
    ▼
TextVectorization Layer   ← In-graph preprocessing (no training-serving skew)
    │
    ▼
Embedding Layer           ← Dense vector representations
    │
    ▼
Bi-Directional LSTM       ← Captures temporal context in both directions
    │
    ▼
Custom Attention Layer    ← Learns to focus on key tokens
    │
    ▼
Dense Classifier
    │
    ▼
Sigmoid Output            ← Binary classification (e.g., Bullish / Bearish)
```

---

## ⚡ Senior-Level Scalability Features

### 1. `tf.data.AUTOTUNE` — Parallelized Data Pipeline
```python
dataset = dataset.map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```
**Why it matters:** Decouples CPU preprocessing from GPU training. The GPU never starves waiting for data — critical for large-scale production workloads.

### 2. `@tf.function` — Graph-Compiled Training Loop
```python
@tf.function
def train_step(model, x, y, optimizer, loss_fn, metric):
    with tf.GradientTape() as tape:
        ...
```
**Why it matters:** Compiles Python code into a TensorFlow static graph, yielding **10–50% faster execution** compared to eager mode. This is the production deployment pattern.

### 3. Global Norm Gradient Clipping — Numerical Stability
```python
grads, _ = tf.clip_by_global_norm(grads, 5.0)
```
**Why it matters:** Prevents the **exploding gradient** problem endemic to deep RNN/LSTM stacks. Essential for stable training on long sequences.

### 4. Custom Attention Mechanism
```python
class Attention(layers.Layer):
    def call(self, features):
        score = tf.nn.tanh(self.W1(query) + self.W2(features))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        ...
```
**Why it matters:** Allows the model to selectively weight critical tokens (e.g., *"surge"*, *"inflation"*) regardless of position, dramatically improving performance on longer documents.

### 5. In-Graph Vectorization — Zero Training-Serving Skew
```python
self.vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=max_seq_len
)
```
**Why it matters:** Preprocessing lives inside the model graph, ensuring **identical behavior at training and inference** — eliminating a critical class of production bugs.

---

## 🖥️ Interactive Dashboard

The project ships with a full **Streamlit dashboard** for live demonstration:

| Section | Description |
|---|---|
| **Pipeline Configuration** | Adjust batch size, vocab size, and epochs |
| **Training Loop** | Run the custom `tf.GradientTape` loop with live loss/accuracy metrics |
| **Real-time Inference** | Enter any text and get instant sentiment classification |
| **Technical Architecture** | Code snippets and visual model blueprint |

---

## 🛠️ Setup & Run

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/sechan9999/LLMpipeline.git
cd LLMpipeline
pip install -r requirements.txt
```

### Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard will be available at **http://localhost:8501**

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `tensorflow >= 2.15` | Model training, `tf.data` pipeline |
| `streamlit >= 1.30` | Interactive dashboard |
| `numpy` | Numerical operations |
| `pandas` | Data manipulation |

---

## 🎤 Interview Talking Points (RELX / Senior DS III)

When presenting this repository, emphasize:

1. **Scalability by design** — `AUTOTUNE` ensures the pipeline scales from a single GPU to multi-worker distributed training with zero code changes
2. **Production parity** — In-graph preprocessing eliminates training-serving skew, a common failure mode in deployed ML systems
3. **Numerical robustness** — Gradient clipping demonstrates awareness of deep sequence model instability, separating practitioners from researchers
4. **Architectural modularity** — Each concern (data, model, training) is fully separated, enabling independent testing and deployment

---

## 📄 License

MIT License — free to use for interview preparation and portfolio purposes.
