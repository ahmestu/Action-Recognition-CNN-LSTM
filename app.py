import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from flask import Flask, render_template, request

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

tokenizer_path = os.path.join(BASE_DIR, 'tokenizer_pro.pkl')
model_info_path = os.path.join(BASE_DIR, 'model_info.txt')
model_path = os.path.join(BASE_DIR, 'action_model_pro.h5')

with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)
with open(model_info_path, 'r') as f:
    max_length = int(f.read())

print("Loading Pro Model...")
model = load_model(model_path)
base_model = Xception(weights='imagenet', include_top=False, pooling='avg')
cnn_model = Model(inputs=base_model.inputs, outputs=base_model.outputs)
print("Pro AI Loaded!")

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer: return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        
        # Predict probabilities for all words.
        preds = model.predict([photo, sequence], verbose=0)[0]
        
        # Getting the top 10 most likely words.
        top_indices = np.argsort(preds)[-10:][::-1] 
        
        word = None
        current_words = in_text.split()
        
        for idx in top_indices:
            candidate = idx_to_word(idx, tokenizer)
            if candidate is None: continue
            
            # --- RULES ---
            # 1. Don't repeat the exact previous word.
            if candidate == current_words[-1]: continue
            
            # 2. Don't allow the same word more than twice in a sentence.
            if current_words.count(candidate) > 1: continue
            
            # 3. If it's a very common word like 'a', 'is', 'the', be even stricter.
            if candidate in ['a', 'the', 'is', 'an'] and candidate in current_words:
                continue

            word = candidate
            break
            
        if word is None or word == 'endseq':
            break
            
        in_text += ' ' + word
        
    result = in_text.replace('startseq', '').replace('endseq', '').strip()
    
    if len(result) < 5:
        return "a person is performing an action"
        
    return result

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    image_path = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            image_path = f"static/uploads/{filename}"
            
            # AI Logic.
            img = load_img(filepath, target_size=(299, 299))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            photo = cnn_model.predict(img, verbose=0)
            
            prediction = generate_desc(model, tokenizer, photo, max_length)
            
    return render_template('index.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)