import os
import pickle
import numpy as np
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'static\uploads'
app.config['MODEL_FILE'] = 'model.h5'
app.config['TOKENIZER_FILE'] = 'tokenizer_sentences.pkl'
app.config['LABEL_ENCODING_FILE'] = 'dictionary.pickle'
max_length = 300
pad_type = 'post'
trunc_type = 'post'

nlp_model = load_model(app.config['MODEL_FILE'])

with open(app.config['TOKENIZER_FILE'], 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

with open(app.config['LABEL_ENCODING_FILE'], 'rb') as handle:
    loaded_encoding_to_label = pickle.load(handle)

def preprocess_text(text):
    sequence = loaded_tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding=pad_type, truncating=trunc_type)
    return np.array(padded_sequence)

def predict_category(text):
    preprocessed_text = preprocess_text(text)
    prediction = nlp_model.predict(preprocessed_text)
    predicted_category_index = np.argmax(prediction[0])
predicted_category_label = loaded_encoding_to_label[predicted_category_index]    
    return predicted_category_label

@app.route("/")
def index():
    return "Hello World!"

@app.route("/prediction", methods=["POST"])
def prediction_route():
    if request.method == "POST":
        # Check if a file is uploaded
        if 'file' in request.files:
            file = request.files['file']
            # Read text from the file
            text = file.read().decode('utf-8')
        else:
            # If no file is uploaded, get text from the form data
            text = request.form.get("text")

        if text:
            predicted_category = predict_category(text)
            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success predicting"
                },
                "data": {
                    "predicted_category": predicted_category
                }
            }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Invalid input. Please provide text data or upload a file."
                },
                "data": None,
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None,
        }), 405

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))



resume_example =  resume_example = """
          TEACHER           Summary     Teacher devoted to helping children think creatively, solve problems independently and respect themselves and others.  Teacher focused on implementing highly effective instructional practices to improve student learning and academic performance.        Highlights          Friendly.  Complex problem solver,  Active listener  Reliable and punctual   Excellent communication skills  Cheerful and energetic      Qualified tutor  Charting and recordkeeping  Positive reinforcement methods  Skilled in working with special needs children   Approachable  Complex problem solver            Experience      teacher    August 2005   to   January 2016     Company Name   ï¼   City  ,   State            teacher    January 1997   to   January 2004     Company Name   ï¼   City  ,   State            teacher    January 1986   to   January 1989     Company Name   ï¼   City  ,   State      Skills Proficiency in Microsoft Office Capable of integrating these programs with the coursework taught Sound decision maker Giving recommendations and opinions to school management upon their requests Discipline students Setting basic class rules by encouraging student feedback Patience Encouraging students to express their discomforts and catering to them in timely manner Giving personalized attention to students Time management Breaking up grading material in small groups in order to evaluate them timely Structuring tasks based on priorities Proficiency in mathematics and science More than 20 years of teaching experience in Mathematics and Science Purposeful lesson planning Making flexible lesson plans based on promoting students' critical and analytic capabilities Self-motivated Fast learner Learned various mathematics software's such as 'graph master' in a relatively short period.          Education      Masters in Education   :   Education  ,   1992    Government College of Education   ï¼   City  ,   State  ,   Pakistan            Bachelor of Education   :   Education  ,   1990    Government College of Education   ï¼   City  ,   State  ,   Pakistan            Bachelor of Science   :   Biology, Chemistry  ,   1986    Karachi University   ï¼   City  ,   State  ,   Pakistan            BSc   :   Biology, Chemistry  ,   1986    Karachi University   ï¼   City  ,   State  ,   Pakistan            Skills     basic, lesson planning, lesson plans, Mathematics, Microsoft Office, express, Fast learner, Self-motivated, Sound, teaching, Time management
 """
resume_example = removeStopwords(resume_example) #di bersihkan data
loaded_model = load_model('/content/drive/MyDrive/Project/checkpoint/model1.h5')

# Memuat tokenizer dari file .pkl
with open('/content/drive/MyDrive/nlp/tokenizer_sentences1.pkl', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

# Memuat dictionary dari file .pkl
with open('/content/drive/MyDrive/Project/checkpoint/dictionary1.pkl', 'rb') as handle:
    loaded_encoding_to_label = pickle.load(handle)
example_sequence = loaded_tokenizer.texts_to_sequences([resume_example])
example_padded = pad_sequences(example_sequence, maxlen=max_length, padding=pad_type, truncating=trunc_type)
example_padded = np.array(example_padded)

# Melakukan prediksi menggunakan model yang telah dimuat
loaded_prediction = loaded_model.predict(example_padded)
print(loaded_prediction[0])
print(np.argmax(loaded_prediction[0]))
# Mendapatkan indeks kategori dengan probabilitas tertinggi
predicted_category_index = np.argmax(loaded_prediction[0])

# Mendapatkan label kategori dari dictionary
predicted_category_label = loaded_encoding_to_label[predicted_category_index]
# Get the indices of the top 5 predicted classes
top_indices = np.argsort(loaded_prediction[0])[::-1][:5]

# Display the top 5 predicted categories and their probabilities
#print(prediction[0])
print("Top 5 Predicted Categories:")
for index in top_indices:
    predicted_label = loaded_encoding_to_label.get(index, "Unknown")
    probability = loaded_prediction[0][index]
    print(f"{predicted_label}: {probability:.4f}")

# Menampilkan hasil prediksi
print("Predicted Category:", predicted_category_label)