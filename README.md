# Fusing Memory and Attention for Symbolic Music Generation

This repository accompanies the paper:

***Fusing Memory and Attention: A Study on LSTM, Transformer, and Hybrid Architectures for Symbolic Music Generation***

The project provides a systematic empirical analysis of **LSTM**, **Transformer**, and a **Hybrid Transformer–LSTM** architecture for symbolic melody generation, with an emphasis on the trade-off between **local melodic continuity** and **global structural coherence**.

---

## Repository Structure

This repository is intentionally organized **by research objective**, with each branch corresponding to a distinct experimental track reported in the paper.

```text
main/
├── dataset/
│   ├── processed/
│   │   ├── train_dataset
│   │   ├── train_mapping
│   │   ├── val_dataset
│   │   ├── val_mapping
│   │   ├── test_dataset
│   │   └── test_mapping
│   └── seeds.txt
└── preprocess.py

lstm/                # LSTM baseline experiments
transformer/         # Transformer baseline experiments
hybrid/              # Proposed Transformer–LSTM hybrid model
hybrid-ablation/     # Controlled ablation studies on the hybrid model
```

### Branch Overview

* **`main`** (this branch)

  * Contains the **fully preprocessed dataset** used across all experiments
  * Includes the **reproducible preprocessing pipeline** (`preprocess.py`)
  * Serves as the canonical entry point for the project

* **`lstm`**

  * Training and evaluation scripts for the LSTM baseline
  * Generated melodies, metric summaries, and logs

* **`transformer`**

  * Decoder-only Transformer model implementation
  * Generation outputs and evaluation results

* **`hybrid`**

  * Proposed Transformer Encoder + LSTM Decoder architecture
  * Full training, inference, and evaluation pipeline

* **`hybrid-ablation`**

  * Ablation experiments isolating architectural components
  * Encoder depth, attention width, decoder capacity, embeddings, positional encoding, and regularization studies

---

## Dataset

All experiments are conducted on the **Deutschl subset of the Essen Folk Song Collection** (available via the Humdrum **KernScores** repository: [Deutschl collection](https://kern.humdrum.org/cgi-bin/browse?l=essen/europa/deutschl)), consisting of monophonic symbolic melodies in **Kern (.krn)** format.

### Preprocessing Highlights

* Key normalization:

  * All major keys → **C major**
  * All minor keys → **A minor**
* Duration filtering to a fixed rhythmic vocabulary
* Symbolic encoding using:

  * MIDI pitch numbers
  * Explicit rest tokens
  * Duration continuation markers (`_`)
* Fixed sequence length: **64 tokens**
* Train/Validation/Test split: **80 / 6 / 14** (fixed random seed)

The processed datasets in this branch are **identical across all models**, ensuring fair architectural comparison.

---

## Experimental Summary

Across all experiments, **1,000 melodies per architecture** were generated from a shared set of fixed seed sequences and evaluated using **17 musically grounded metrics**, spanning both local and global musical properties.

### Key Findings

* **LSTM**

  * Strong local continuity and stylistic consistency
  * Weak long-range structure retention

* **Transformer**

  * Improved global dependency modeling
  * Occasional irregular phrasing and local instability

* **Hybrid Transformer–LSTM (Proposed)**

  * Best overall balance between:

    * Local melodic smoothness
    * Global structural coherence
  * Highest scores in pitch variance, motif diversity, entropy, and perceptual originality

A structured **human listening study** further confirmed that the hybrid model produces melodies rated higher in musicality, coherence, and overall quality.

---

## Reproducibility

* All preprocessing steps are deterministic
* Identical seed sequences are used across models
* Metric computation is shared across experimental branches

To regenerate the dataset from raw `.krn` files:

```bash
python preprocess.py
```

> **Note:** Raw `.krn` files are not redistributed due to dataset licensing. Please obtain the Essen Folk Song Collection separately.
<!--
---

## Preprint and Paper

A PDF version of the **author preprint** is provided for reference:

```
/ paper_preprint.pdf
```

This repository reflects the **exact experimental setup and results reported in the paper**.

---

## Citation

If you use this code or dataset in your research, please cite the paper:

### BibTeX

```bibtex
@article{ghoshal2025fusing,
  title={Fusing Memory and Attention: A Study on LSTM, Transformer, and Hybrid Architectures for Symbolic Music Generation},
  author={Ghoshal, Soudeep and Chakraborty, Sandipan and Chowdhury, Pradipto and Buckchash, Himanshu},
  journal={Expert Systems with Applications},
  year={2025}
}
```
-->
---

## License

This repository is released for **academic and research use only**.

Please refer to the individual dataset and library licenses for additional restrictions.

---

## Contact

For questions, clarifications, or collaboration inquiries:

**Soudeep Ghoshal**
Email: [soudeepghoshal.sg@gmail.com](mailto:soudeepghoshal.sg@gmail.com)

---

*This repository is structured to emphasize clarity, reproducibility, and architectural insight rather than leaderboard optimization.*
