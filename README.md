# MelodySynthesis: Hybrid Transformer-LSTM Melody Generation

## 📌 Research Overview

This repository is part of an ongoing research project exploring deep learning approaches for melody synthesis. The study compares different architectures, including **LSTM, Transformer, a Transformer-LSTM hybrid, and a GAN-based Transformer-LSTM hybrid model**, to determine which performs best in generating musically coherent melodies.

This branch (`transformer-lstm`) focuses on a **Hybrid Transformer Encoder - LSTM Decoder model** for melody generation. This architecture leverages the **self-attention mechanism of Transformers** to model long-range dependencies in musical sequences while using **LSTM layers in the decoder** to capture local temporal patterns efficiently.

## 📂 Repository Structure

```
MelodySynthesis/
│── melodies/               # Performance evaluation files
│── processed_data/         # Preprocessed data formatted for model input
│── model/                  # Trained models & related artifacts
│   │── plots/              # Training analysis & model evaluation plots
│── train.py                # Training script for Hybrid Transformer-LSTM
│── melody_generator.py     # Script to generate melodies using trained model
│── evaluate.py             # Performance evaluation metrics & visualization
│── utils.py                # Data preprocessing & helper functions
│── requirements.txt        # Dependencies
│── README.md               # Project documentation (this file)
│── LICENSE.txt             # License information
```

## 🎶 Model Architecture

The **Hybrid Transformer Encoder - LSTM Decoder** model consists of:

### **Encoder (Transformer)**:
- **Embedding Layer**: Converts input sequences into dense representations.
- **Positional Encoding**: Adds sequence order information.
- **Multi-Head Self-Attention**: Captures relationships between different notes across the sequence.
- **Feed-Forward Layer**: Applies transformation for enhanced feature extraction.
- **Layer Normalization**: Stabilizes training.

### **Decoder (LSTM)**:
- **LSTM Layer 1**: 256 units, processes encoded musical context.
- **Dropout Layer**: (Rate = 0.2) to reduce overfitting.
- **LSTM Layer 2**: 256 units, generates sequential note outputs.
- **Dense Output Layer**: Predicts the next note in the sequence.

📊 **Total Trainable Parameters**: 8,176,941

Model visualization:  
![Model Architecture](model/hybrid_transformer_lstm_architecture.png)

## 📊 Training Details

### **Training Parameters**:
- Optimizer: **Adam** (`lr=0.001`, adaptive learning rate)
- Loss Function: **Sparse Categorical Cross-Entropy**
- Batch Size: **64**
- Epochs: **50** (with **early stopping** to prevent overfitting)

### **Final Training Results**:
- **Training stopped at**: **Epoch 7**
- **Best model saved from**: **Epoch 2**
- **Training Accuracy**: **77.27%**
- **Validation Accuracy**: **77.95%**
- **Training Loss**: **0.9947**
- **Validation Loss**: **0.9254**
- **Test Accuracy**: **77.55%**

**Training Progress** (accuracy & loss over epochs):  
![Training Analysis](model/plots/training_history.png)

## 📊 Model Performance & Evaluation  

The trained model was evaluated using classification metrics:

✔️ **Confusion Matrix**:  
![Confusion Matrix](model/plots/confusion_matrix.png)

✔️ **Classification Counts**:  
![Classification Distribution](model/plots/classification_counts.png)

### Key Observations:
- The Transformer encoder helps in capturing long-range dependencies.
- The LSTM decoder maintains sequential flow but struggles with some note transitions.
- Training performance suggests potential for further fine-tuning.

## 🎵 Evaluation of Generated Melodies

1000 generated melodies were analyzed using musical quality metrics. Here are the results:

| **Metric**                  | **Mean** | **Min** | **Max** |
|-----------------------------|---------|---------|---------|
| **Pitch Variance**          | 7.11    | 0.74    | 32.74   |
| **Pitch Range**             | 5.21    | 0.89    | 14.00   |
| **Rhythmic Variance**       | 0.25    | 0.01    | 0.98    |
| **Note Density**            | 0.92    | 0.25    | 2.50    |
| **Rest Ratio**              | 0.06    | 0.00    | 0.35    |
| **Interval Variability**    | 3.94    | 0.17    | 23.57   |
| **Note Repetition**         | 1.45    | 1.00    | 3.33    |
| **Contour Stability**       | 0.78    | 0.00    | 2.57    |
| **Syncopation**             | 1.27    | 0.00    | 6.20    |
| **Harmonic Tension**        | 0.38    | 0.20    | 0.53    |
| **KL Divergence**           | 1.05    | 0.00    | 1.79    |
| **Pitch Entropy**           | 2.70    | 1.37    | 3.44    |
| **Rhythmic Entropy**        | 1.94    | 0.32    | 3.45    |
| **Motif Diversity Index**   | 0.86    | 0.50    | 1.00    |
| **Harmonic Complexity**     | 0.90    | 0.26    | 1.90    |
| **Contour Variability**     | 45.51   | 0.00    | 80.00   |
| **Tonal Drift**             | 6.41    | 0.00    | 14.00   |

These results provide insights into the structural quality, variation, and coherence of the generated melodies.

## 🛠️ How to Use

### 1️⃣ Setup Environment  
Ensure dependencies are installed:  
```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model  
If you wish to retrain the model, run:  
```bash
python train.py
```

### 3️⃣ Generate Melodies  
Run the melody generation script:  
```bash
python melody_generator.py
```
Generated melodies will be saved in the `melodies/` folder.

### 4️⃣ Evaluate the Model  
Analyze the model's performance using:  
```bash
python evaluate.py
```

## 💪 Research Significance  

This study contributes to **AI-driven music generation** by comparing deep learning architectures for melody synthesis. Our findings will:
- Identify **which model best captures musical structure**.
- Explore how **attention mechanisms** compare with **recurrent networks** in sequence generation.
- Provide insights into **hybrid architectures (Transformer-LSTM, GAN-based models)** for music synthesis.

Each model is implemented in a separate branch:
- **LSTM Model** (`lstm` branch)  
- **Transformer Model** (`transformer` branch)  
- **Hybrid Transformer-LSTM** (`transformer-lstm` branch) ⬅️ _(current branch)_  
- **GAN-Based Model** (`gan-transformer-lstm` branch)  

By analyzing these models, we aim to determine the most **effective approach for AI-generated melodies**.

## 🚀 Future Work

- 🔹 Implementing **GAN-augmented Transformer-LSTM** models.
- 🔹 Exploring **music-theory-guided generation** (ensuring harmony & rhythm).
- 🔹 Enhancing **real-time generation capabilities**.

## 🤝 Contributors

- **Soudeep Ghoshal** ([@SoudeepGhoshal](https://github.com/SoudeepGhoshal))  

## 🐝 License  

This project is licensed under the **MIT License**. See [LICENSE](LICENSE.txt) for details.  

---  
📌 **Note**: This is a research-oriented repository, and findings will be documented in an upcoming paper. If you use this work, please cite appropriately.