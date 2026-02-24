# Electrical Demand Prediction Dashboard 

A real-time electricity demand forecasting dashboard built with Python using a hybrid deep learning architecture (CNN + LSTM + GRU).
The system predicts electricity demand for selected cities with a 24-hour lead time.

This project leverages deep learning to model both short-term fluctuations and long-term temporal dependencies in electricity consumption data.
The application is deployed through an interactive Streamlit dashboard and exposed using ngrok.

![load_pred](https://github.com/user-attachments/assets/bf3ba47e-abcc-4159-b77f-3e5bd4b15087)

### Model Architecture

- Convolutional Neural Networks (CNN) to detect short-term local patterns.
- Long Short-Term Memory (LSTM) to learn sequential dependencies.
- Gated Recurrent Unit (GRU) to capture long-term dependencies.

### Architecture Flow

- Input Time-Series Data
- CNN (feature extraction)
- LSTM (sequence learning)
- GRU (long-term dependency modeling)
- Dense Layer
- 24-hour Demand Forecast

### Future Improvements

- Multi-step forecasting beyond 24 hours
- Model Optimization
- Cloud Deployment(AWS)
