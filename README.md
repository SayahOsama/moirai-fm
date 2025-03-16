# Moirai Time Series Forecasting

## 📌 Overview
This project leverages **Moirai**, a universal time series forecasting transformer, to predict future trends in time series data. It utilizes **GluonTS, PyTorch, and Uni2TS** to process and analyze time series data efficiently.

## 🚀 Features
- Supports **zero-shot forecasting** with pre-trained Moirai models.
- Fine-tuning capabilities for improved accuracy.
- Handles multivariate and univariate time series forecasting.
- Generates and saves predictions along with visualizations.
- Automatically organizes results in a structured folder (`results/`).

## 📎 Directory Structure
```
moirai-fm/
│── moirai_example.py         # Main script for time series forecasting
│── Dockerfile                # Dockerfile to containerize the project
│── results/                  # Directory for storing model outputs
│── uni2ts/                   # Cloned repository of Uni2TS
│── README.md                 # Project documentation (this file)
```

## 🛠 Setup & Installation
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/SayahOsama/moirai-fm.git
cd moirai-fm
```

### **2️⃣ Set Up a Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate    # On Linux/Mac
venv\Scripts\activate       # On Windows
```

### **3️⃣ upgrade pip**
```bash
pip install --upgrade pip
```

### **4️⃣ Clone & Install Uni2TS**
```bash
git clone https://github.com/SalesforceAIResearch/uni2ts.git
cd uni2ts
pip install -e '.[notebook]'
cd ..
```

---

## 🏃 Running the Model
```bash
python moirai_example.py
```

### Expected Outputs:
- Forecasted values printed in the terminal.
- Plots saved inside `results/` as PNG images.

