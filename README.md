# 🎧 Speech Emotion Recognition System (End-to-End)

An end-to-end **Speech Emotion Recognition (SER)** system that detects human emotions from speech audio.  
The project focuses not only on model accuracy, but also on **real-world deployment, scalability, and explainability**.

Built as a **full-stack applied ML project** using deep learning, FastAPI, and React.

---

## 🚀 Features

- 🎙️ Emotion detection from **uploaded audio files** or **live microphone recording**
- 🧠 Deep Learning model using **CNN + LSTM** for temporal emotion modeling
- 📊 **Utterance-level prediction** using majority voting across audio segments
- 🧩 Handles **long audio files** (minutes) safely using batched inference
- 📈 Emotion **distribution bars** and **timeline visualization**
- 🌐 Fully deployed **backend (API)** and **frontend (web app)**

---

## 🧠 Model Architecture


### Why CNN + LSTM?
- **CNN** captures local acoustic patterns (pitch, energy, timbre)
- **LSTM** captures temporal emotional flow
- **Voting** stabilizes predictions across time

---

## 📚 Datasets Used

The model is trained using multiple publicly available datasets to improve robustness:

- **RAVDESS** – clean, acted emotional speech
- **CREMA-D** – diverse speakers and emotions
- **IEMOCAP** – conversational, semi-natural emotional speech

Using multiple datasets helps the model generalize better to real-world audio.

---

## 🧪 Emotion Inference Strategy

Instead of predicting emotion from a single short clip:

1. Audio is split into overlapping windows
2. Windows are grouped into sequences
3. Each sequence predicts one emotion
4. **Final emotion is chosen by majority voting**
5. Confidence = dominance of final emotion across time

This mirrors how emotions persist and evolve in real speech.

---

## 🛠️ Tech Stack

### 🔹 Machine Learning
- TensorFlow / Keras
- Librosa
- NumPy
- OpenCV

### 🔹 Backend
- FastAPI
- Uvicorn
- Python
- Deployed on **Render**

### 🔹 Frontend
- React
- JavaScript
- Web Audio API (microphone recording)
- Deployed on **Vercel / Netlify**

---

## 🌐 Deployment

- **Backend API**: FastAPI app deployed on Render  
- **Frontend**: React app deployed on a static hosting platform  
- Supports HTTPS (required for microphone access)

> Note: On free-tier cloud hosting, the first request may take a few seconds due to cold start.

---

## 📊 Output Example

- **Final Emotion**: e.g. *Sad*
- **Confidence**: Vote-based dominance
- **Emotion Distribution**: Bar chart (vote-based)
- **Emotion Timeline**: Emotion progression over time

This makes the model’s decision **interpretable and explainable**.

---

## 🎯 Key Learnings

This project helped me understand:

- Difference between **model accuracy** and **system reliability**
- Handling long audio and memory constraints
- Importance of batching during inference
- Why probabilities ≠ final decisions
- Full ML pipeline: training → API → frontend → deployment

---

## 🚧 Limitations & Future Work

- Improve accuracy using **Transformer-based audio models**
- Add multilingual emotion recognition
- Speaker adaptation
- Emotion intensity estimation
- Better UI/UX visualizations

---

## 👨‍🎓 Author

**Bhavesh Girase**  
Second-year undergraduate student  
Interested in Applied Machine Learning, Deep Learning, and ML Systems

---

## 📌 Disclaimer

This project is intended for **educational and research purposes**.  
Emotion recognition is a complex problem and predictions may not always reflect true emotional states.

---

⭐ If you found this project interesting, feel free to explore, fork, or share feedback!
Link: https://ser-frontend-zdoh.vercel.app/
