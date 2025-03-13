# RankPO: Rank Preference Optimization

This repository contains the implementation of a two-stage framework for aligning models with AI preferences while retaining previously learned knowledge. The framework includes **contrastive learning** and **rank preference optimization (RankPO)**, along with supportive utilities for fine-tuning and evaluation.

---

## 📂 Repository Structure

### **Main Components**

- **Contrastive Learning**: 
  - `run_contrastive.py`: Script to train models using contrastive learning with rule-based data generation.
  - `contrastive_trainer.py`: Custom trainer leveraging Hugging Face's `Trainer` for contrastive learning.
- `modeling.py`: Defines the core model operations such as `forward` passes and inference mode.

- **Rank Preference Optimization (RankPO)**:
  - `run_rankpo.py`: Script to train models using RankPO for preference alignment.
  - `rankpo_trainer.py`: Custom trainer for RankPO, implementing specialized training objectives.

### **Supportive Files**

- **Arguments**:
  - `arguments.py`: Defines training, data, and model arguments for both stages.

- **Utilities**:
  - `utils/`: Contains helper functions for data processing, collators, and other operations.

- **Configs**:
  - `configs/`: Directory for configuration files for deepspeed.

---


### **Usage**

#### **1. Contrastive Learning**

The first stage uses **contrastive learning** with rule-based data generation to train models on limited data. This step builds robust and generalizable ranking capabilities.

Run the contrastive learning stage with:

```bash
python run_contrastive.py \
  --model_name_or_path <model-name> \
  --train_data <path-to-train-data> \
  --output_dir <output-directory> \
  --do_train \
  --per_device_train_batch_size 8 \
  --num_train_epochs 3
```

#### **2. Rank Preference Optimization (RankPO)**

The second stage fine-tunes the model to align with AI preferences using RankPO.

Run the RankPO stage with:

```bash
python run_rankpo.py \
  --model_name_or_path <model-name> \
  --train_data <path-to-train-data> \
  --output_dir <output-directory> \
  --do_train \
  --per_device_train_batch_size 16 \
  --num_train_epochs 3
```

---

### **Key Features**

- **Contrastive Learning**:
  - Rule-based data generation for scenarios with limited data.
  - Custom trainer (`contrastive_trainer.py`) for handling embeddings and contrastive objectives.

- **Rank Preference Optimization (RankPO)**:
  - Incorporates reference-free or reference-based comparisons.
  - Supports flexible loss functions (e.g., hinge, sigmoid).
  - Fine-tunes models for preference alignment.

---

## 📄 Example Data Format

### **Contrastive Learning Data**
```json
{
  "query": "Example query",
  "positives": ["Positive sample 1", "Positive sample 2"],
  "negatives": ["Negative sample 1", "Negative sample 2"]
}
```

### **RankPO Data**
```json
{
  "query": "Example query",
  "passage1": "Passage for comparison A",
  "passage2": "Passage for comparison B",
  "preferred": "A"  # or "B"
}
```

---

## 📜 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for more details.

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions, feature requests, or bug fixes, please open an issue or submit a pull request.

