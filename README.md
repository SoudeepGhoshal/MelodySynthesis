# MelodySynthesis: Transformer-Based Melody Generation

## 📌 Research Overview

This repository is part of an ongoing research project exploring deep learning approaches for melody synthesis. The study compares different architectures—including **LSTM, Transformer, a Transformer-LSTM hybrid, and a GAN-based Transformer-LSTM hybrid model**—to determine which performs best in generating musically coherent melodies.

This branch (`transformer`) focuses specifically on **Transformer-based melody generation**, investigating how self-attention mechanisms capture musical dependencies in sequences more efficiently than recurrence-based approaches.

## 📂 Repository Structure

```
MelodySynthesis/
│── melodies/               # Performance evaluation files
│── processed_data/         # Preprocessed data formatted for model input
│── model/                  # Trained models & related artifacts
│   │── plots/              # Training analysis & model evaluation plots
│── train.py                # Transformer model training script
│── melody_generator.py     # Script to generate melodies using trained model
│── evaluate.py             # Performance evaluation metrics & visualization
│── utils.py                # Data preprocessing & helper functions
│── requirements.txt        # Dependencies
│── README.md               # Project documentation (this file)
│── LICENSE.txt             # License information
```

## 🎶 Model Architecture

The **Transformer model** uses self-attention layers to understand global patterns in musical sequences. The model's decoder stack comprises multiple layers of attention and feedforward blocks.

### Transformer Decoder Block

The decoder in this Transformer-based model processes musical sequences and predicts the next note. It consists of a stack of four identical decoder blocks, each designed to extract and refine temporal dependencies. Below is a detailed breakdown of the architecture:

**Components per Decoder Block:**
1. **Positional Encoding**
- Adds sinusoidal positional encodings to the input embeddings.
- Helps the model understand token order without recurrence.

2. **Layer Normalization**
- Normalizes inputs to stabilize and accelerate training.

3. **Multi-Head Self-Attention**
- Uses 8 parallel attention heads
- Captures contextual relationships across the sequence.

4. **Dropout Layer**
- Dropout rate: 0.2.
- Regularizes the attention output to prevent overfitting.

5. **Residual Connection**
- Adds the attention output back to the block's input.
- Preserves original features and supports gradient flow.

6. **Second Layer Normalization**
- Further normalizes the output from the residual addition.

7. **Position-wise Feedforward Network**
- First Dense layer: 1024 units with ReLU activation.
- Second Dense layer: 256 units (same as model dimension).

8. **Second Dropout Layer**
- Dropout rate: 0.2.
- Applied to the feedforward network output for additional regularization.

9. **Final Residual Connection**
- Adds the feedforward output to the previous residual output.

**Decoder Stack:**
- This block structure is repeated 4 times (i.e., 4 decoder layers).
- Enables the model to learn deep hierarchical representations of the input.

**Output Layer:**
- The last time step's output is extracted.
- Passed through a final Dense layer with softmax activation.
- Produces a probability distribution over 45 possible musical notes.

Model visualization:  
![Model Architecture](model/transformer_architecture.png)

## 📊 Training Details

**Training Parameters**:
- Optimizer: **Adam** with a **Custom Learning Rate Schedule**:  
  A custom learning rate schedule is used instead of a fixed rate. The learning rate increases linearly for the first few thousand steps (warm-up phase) and then decays proportionally to the inverse square root of the step number. This helps stabilize and accelerate training, especially in attention-based models. The schedule is configured with ```d_model = 256``` and ```warmup_steps = 4000```.
  
- Loss Function: **Sparse Categorical Cross-Entropy**
- Batch Size: **64**
- Epochs: **50** (with **early stopping** to prevent overfitting)


Training stopped at **epoch 18** (best model restored from **epoch 13**), achieving:
- **Training Accuracy**: **0.8557**
- **Validation Accuracy**: **0.8481**
- **Test Accuracy**: **0.8482**

**Training Progress** (accuracy & loss over epochs):  
![Training Analysis](model/plots/training_history.png)

## 📈 Model Performance & Evaluation

The trained Transformer model was evaluated using standard classification metrics:

✔️ **Confusion Matrix**:  
![Confusion Matrix](model/plots/confusion_matrix.png)

✔️ **Classification Counts**:  
![Classification Distribution](model/plots/classification_counts.png)

### Key Observations:
- The Transformer captures long-term dependencies more effectively than RNNs.
- Some note sequences still pose prediction challenges.
- Minimal overfitting observed with appropriate regularization.

## 🎼 Evaluation of Generated Melodies

To assess the quality of melodies generated by the Transformer model, 1000 melodies were sampled from random seed sequences and evaluated using various musical metrics. Below are key evaluation results:

| **Metric**                | **Mean** | **Min** | **Max** |
| ------------------------- | -------: | ------: | ------: |
| **Pitch Variance**        |   3.6332 |    0.25 | 13.9213 |
| **Pitch Range**           |   4.3260 |  0.5714 |    10.5 |
| **Rhythmic Variance**     |   0.1022 |       0 |  0.6600 |
| **Note Density**          |   1.1394 |  0.3333 |  3.0160 |
| **Rest Ratio**            |   0.0925 |       0 |  0.3750 |
| **Interval Variability**  |   1.6892 |       0 | 14.2550 |
| **Note Repetition**       |   1.8957 |     1.0 |  5.0476 |
| **Contour Stability**     |   0.6474 |       0 |     3.4 |
| **Syncopation**           |   1.6469 |       0 |     9.5 |
| **Harmonic Tension**      |   0.4351 |  0.1650 |  0.6897 |
| **KL Divergence**         |   0.7652 |  0.0079 |  1.9620 |
| **Pitch Entropy**         |   2.4306 |  0.8925 |  3.4053 |
| **Rhythmic Entropy**      |   1.2013 |       0 |  2.5029 |
| **Motif Diversity Index** |   0.7414 |  0.0690 |     1.0 |
| **Harmonic Complexity**   |   0.5081 |  0.0944 |  1.3302 |
| **Contour Variability**   |  24.1560 |       0 | 64.2857 |
| **Tonal Drift**           |   3.5770 |       0 |    12.0 |

## 🛠️ How to Use

### 1️⃣ Setup Environment
Install the required dependencies:
```bash
pip install -r requirements.txt
```
Download the pre-trained model:
```bash
gdown https://drive.google.com/uc?id=1BJf8oagKfX9rsiMxgkmf7uGM6IkUTIVZ --output model/transformer.keras
```

### 2️⃣ Train the Model
```bash
python train.py
```

### 3️⃣ Generate Melodies
```bash
python melody_generator.py
```

### 4️⃣ Evaluate the Model
```bash
python evaluate.py
```

## 💪 Research Significance

This study contributes to **AI-driven music generation** by comparing deep learning architectures for melody synthesis. Our findings will:
- Identify **which model best captures musical structure**.
- Explore how **attention-based models** compare with **recurrent networks** in sequence generation.
- Provide insights into **hybrid architectures (Transformer-LSTM, GAN-based models)** for music synthesis.

Each model is implemented in a separate branch:
- **LSTM Model** (`lstm` branch)
- **Transformer Model** (`transformer` branch) ⬅️ *(current branch)*
- **Hybrid Transformer-LSTM** (`hybrid-transformer-lstm` branch)
- **GAN-Based Model** (`gan-transformer-lstm` branch)

By analyzing these models, we aim to determine the most **effective approach for AI-generated melodies**.

## 🚀 Future Work

- 🔹 Comparing performance across all four architectures
- 🔹 Incorporating **music-theory-based constraints**
- 🔹 Real-time music generation and streaming

## 🤝 Contributors

- **Soudeep Ghoshal** ([@SoudeepGhoshal](https://github.com/SoudeepGhoshal))
- **Sandipan Chakraborty** ([@SChakraborty04](https://github.com/SChakraborty04))

## 🐝 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE.txt) for details.

---
📌 **Note**: This is a research-oriented repository. Please cite this work appropriately if you use it.