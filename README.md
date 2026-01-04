# Action Recognition AI (CNN + LSTM)

This project is a Deep Learning application that recognizes actions and describes images. It uses a **CNN** for feature extraction and an **LSTM** for text generation.

### Features
*   **CNN Architecture:** Xception (Pre-trained on ImageNet)
*   **RNN Architecture:** 512-unit LSTM
*   **Web Framework:** Flask (Python)
*   **Dataset:** Flickr8k

### How to Run
1. Install requirements: `pip install flask tensorflow pillow numpy`
2. Download the model file (`action_model_pro.h5`) from the link below and place it in the main folder.
3. Run the app: `python app.py`
4. Open `http://127.0.0.1:5000` in your browser.

### Model Download Link
https://drive.google.com/file/d/1RKDtiSUf_IT1B2VLKdHHRYaUQ8ALXGev/view?usp=sharing

### Sample Results
The model successfully identifies primary subjects and actions (e.g., "dog running," "person playing soccer").
