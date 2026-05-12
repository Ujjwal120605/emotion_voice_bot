# 🎙️ Emotion-Aware Voice Bot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge\&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge\&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge\&logo=scikitlearn)
![Librosa](https://img.shields.io/badge/Librosa-Audio%20Processing-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Working-success?style=for-the-badge)

### AI-powered Speech Emotion Recognition System

**RV College of Engineering — EC355TDD Project**

Developed by:

* **Ujjwal Bajpai** (1RV23EC175)
* **Pranshul Bhargava** (1RV23EC101)

</div>

---

# 📌 Overview

Emotion-Aware Voice Bot is an intelligent speech emotion recognition system that analyzes human voice recordings and predicts emotional states in real time.

The project combines:

* 🎧 Advanced audio feature extraction
* 🤖 Machine Learning ensemble models
* 📊 Interactive Streamlit dashboard
* 💬 Emotion-aware empathetic responses

The system can classify speech into:

| Emotion   | Emoji |
| --------- | ----- |
| Neutral   | 😐    |
| Happy     | 😄    |
| Sad       | 😢    |
| Angry     | 😠    |
| Fearful   | 😨    |
| Disgusted | 🤢    |
| Surprised | 😲    |

---

# ✨ Features

## 🎤 Speech Emotion Recognition

* Upload `.wav`, `.mp3`, `.ogg`, `.flac`, or `.m4a` files
* Real-time emotion prediction
* Confidence score generation

## 📈 Interactive Dashboard

* Modern dark-themed Streamlit UI
* Waveform visualization
* Spectrogram analysis
* MFCC visualization
* Probability distribution charts

## 🤖 Ensemble Machine Learning Model

Uses a voting ensemble of:

* Random Forest
* Gradient Boosting
* Support Vector Machine (SVM)

## 🎧 Advanced Audio Features

Extracted using **Librosa**:

* MFCC
* Chroma Features
* Mel Spectrogram
* Zero Crossing Rate
* RMS Energy
* Spectral Centroid
* Spectral Bandwidth
* Spectral Rolloff
* Tonnetz
* Pitch / Fundamental Frequency

## 💬 Empathetic AI Responses

The bot responds differently depending on detected emotion.

Example:

| Emotion  | Response                                                            |
| -------- | ------------------------------------------------------------------- |
| Sad 😢   | “I'm really sorry you're feeling down. I'm here for you.”           |
| Happy 😄 | “That's wonderful! Your positivity is contagious!”                  |
| Angry 😠 | “I understand you're frustrated. Let's work through this together.” |

---

# 🏗️ System Architecture

```text
Voice Input
     ↓
Audio Preprocessing
     ↓
Feature Extraction (130 Features)
     ↓
StandardScaler
     ↓
Ensemble ML Model
     ↓
Emotion Prediction
     ↓
Empathetic Response + Visualizations
```

---

# 🧠 Machine Learning Pipeline

## 📌 Feature Vector

The model extracts a **130-dimensional feature vector** from each audio sample.

### Feature Breakdown

| Feature            | Dimensions |
| ------------------ | ---------- |
| MFCC               | 80         |
| Chroma             | 24         |
| Mel Spectrogram    | 2          |
| Zero Crossing Rate | 2          |
| RMS Energy         | 2          |
| Spectral Centroid  | 2          |
| Spectral Bandwidth | 2          |
| Spectral Rolloff   | 2          |
| Tonnetz            | 12         |
| Pitch / F0         | 2          |
| **Total**          | **130**    |

---

# 🧪 Model Training

The project uses synthetic emotional audio generation when real datasets are unavailable.

Synthetic audio simulates:

* pitch variation
* energy changes
* vibrato
* spectral brightness
* noise characteristics

Each emotion has unique acoustic fingerprints.

Example:

| Emotion   | Characteristics                         |
| --------- | --------------------------------------- |
| Happy     | High pitch, bright spectrum, fast tempo |
| Sad       | Low pitch, slow tempo                   |
| Angry     | Loud, noisy, aggressive spectrum        |
| Fearful   | Trembling voice modulation              |
| Surprised | Sudden energy burst                     |

---

# 📊 Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Core programming language |
| Streamlit    | Interactive dashboard     |
| scikit-learn | Machine learning models   |
| Librosa      | Audio processing          |
| NumPy        | Numerical computations    |
| Matplotlib   | Visualizations            |
| SoundFile    | Audio handling            |
| Pickle       | Model serialization       |

---

# 📂 Project Structure

```text
Emotion-Aware-Voice-Bot/
│
├── app.py
├── train_model.py
├── model/
│   ├── pipeline.pkl
│   ├── label_encoder.pkl
│   ├── metadata.json
│   └── confusion_matrix.npy
│
├── requirements.txt
├── README.md
└── assets/
```

---

# ⚙️ Installation Guide

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/emotion-aware-voice-bot.git
cd emotion-aware-voice-bot
```

---

## 2️⃣ Create Virtual Environment

### macOS / Linux

```bash
python3.11 -m venv tfenv
source tfenv/bin/activate
```

### Windows

```bash
python -m venv tfenv
tfenv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install tensorflow librosa streamlit pandas numpy scikit-learn soundfile matplotlib
```

---

# 🚀 Running the Project

## Train the Model

```bash
python train_model.py
```

Expected output:

```text
[EVAL] Test Accuracy: 98.5%
[SAVE] Model saved to 'model/'
```

---

## Start Streamlit Dashboard

```bash
streamlit run app.py
```

The app will open in your browser automatically.

---

# 📸 Dashboard Features

## 🎤 Audio Upload

Upload speech recordings for emotion analysis.

## 📈 Visualizations

* Waveform plot
* Spectrogram
* MFCC heatmap
* Probability bars

## 📜 Session History

Tracks previously analyzed emotions.

## 🎭 Demo Mode

Generate synthetic emotional voices for testing.

---

# 📊 Sample Output

```text
Detected Emotion: HAPPY 😄
Confidence Score: 96.2%

Bot Response:
"That's wonderful! Your positivity is contagious!"
```

---

# 🔬 Research References

1. Y. Guo et al., “A Multi-Feature Fusion Speech Emotion Recognition Method,” IEEE Access 2023
2. J. Wagner et al., “Dawn of the Transformer Era in Speech Emotion Recognition,” 2023
3. W. Chen et al., “DST: Deformable Speech Transformer for Emotion Recognition,” 2023
4. J. Ye et al., “Temporal Modeling Matters for Speech Emotion Recognition,” 2023
5. K. L. Ong et al., “SCQT-MaxViT: Speech Emotion Recognition,” IEEE Access 2023

---

# 🎯 Future Improvements

* 🎙️ Real-time microphone emotion detection
* 🧠 Deep learning CNN/LSTM models
* 🌍 Multilingual emotion recognition
* ☁️ Cloud deployment
* 📱 Mobile app integration
* 🗣️ Voice assistant support

---

# 👨‍💻 Authors

## Ujjwal Bajpai

RV College of Engineering
Department of Electronics & Communication Engineering
USN: 1RV23EC175

## Pranshul Bhargava

RV College of Engineering
Department of Electronics & Communication Engineering
USN: 1RV23EC101

---

# 📜 License

This project is developed for academic and research purposes.

---

# ⭐ Acknowledgements

Special thanks to:

* RV College of Engineering
* Open-source Python community
* Librosa & scikit-learn contributors
* Streamlit developers

---

<div align="center">

## 🎙️ “Understanding emotions through voice.”

Made with ❤️ using Python, ML, and Streamlit.

</div>
