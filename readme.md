# TAN-NILM Project

This repository contains the implementation of a **Temporal Attention Network (TAN)** for **Non-Intrusive Load Monitoring (NILM)** using the UK-DALE dataset. The project is designed to experiment with multiple NILM architectures and compare their performance on appliance-level energy disaggregation.  

## Project Structure
```
nilm/
├── dataset/ukdale/ # Preprocessed UK-DALE dataset (HDF5 format)
│ ├── house_1/
│ ├── house_2/
│ ├── ...
│ └── ukdale_tan.h5 # Combined dataset for TAN-NILM training
├── tan_nilm/ # TAN model implementation
│ ├── init.py
│ ├── dataset.py # NILMDataset class for handling UK-DALE data
│ ├── model.py # TAN model architecture
│ └── train.py # Training script
├── utils/ # Utility functions for preprocessing, evaluation, etc.
├── .venv/ # Python virtual environment
└── README.md
```
---

## Implemented and Planned Methods

The repository currently focuses on **four NILM methods** for comparison and experimentation:

1. **TAN (Temporal Attention Network)**
   - Uses LSTM layers with temporal attention to predict appliance-level energy usage from aggregate mains signals.
   - Designed to capture long-range temporal dependencies and emphasize relevant time steps.

2. **Seq2Point**
   - Predicts the appliance power consumption at a single time point using a sliding window of aggregate mains.
   - Efficient for real-time NILM applications due to reduced output size.

3. **Seq2Seq**
   - Predicts the appliance consumption for an entire sequence window.
   - Useful for capturing appliances with longer operating durations and complex usage patterns.

4. **Windowed CNN**
   - Convolutional Neural Network that operates on sliding windows of mains data.
   - Learns local temporal features and is suitable for appliances with repetitive consumption patterns.

> The TAN method is currently implemented and can be trained on the UK-DALE dataset using `train.py`.

---

## Getting Started

1. **Clone the repository:**

```
git clone https://github.com/yourusername/tan-nilm.git
cd tan-nilm
```

2**Create and activate a virtual environment:**

3.**Install required packages:**

```
pip install -r requirements.txt
```

4.**Prepare the UK-DALE dataset:**
Place the dataset in dataset/ukdale/ and generate the ukdale_tan.h5 file using the provided preprocessing scripts.

5.**Train the TAN model:**
```
python3 -m tan_nilm.train
```

---

Notes

- The training process uses mini-batches to reduce RAM usage; adjust batch_size and seq_len in CONFIG as needed for your hardware.

- Dataset preprocessing scripts should ensure all appliance channels are aligned and aggregated correctly for TAN input.

- Future work will integrate Seq2Point, Seq2Seq, and CNN methods into the same pipeline for systematic comparison.

---

## License

This project is licensed under the MIT License.