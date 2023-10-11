import openai
from flask import Flask, render_template, request, jsonify
from gensim.models import Word2Vec
import os
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext  # type: ignore
os.environ["OPENAI_API_KEY"] = "YOUR API KEY"


# pip install flask
# Ganti dengan token API Anda
api_key = "YOUR API KEY"

# Memuat model Word Embeddings
# word2vec_model = Word2Vec.load("word2vec.model")

# Fungsi untuk berinteraksi dengan model GPT-3.5


def chat_with_gpt3(prompt, api_key, max_tokens=1000):

    prompt = f"'role': 'system','content': 'isi dengan persona yang diharapkan'"
    openai.api_key = api_key

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        # stop=None,
        temperature=1,
        # n=1
    )

    return response.choices[0].text.strip()  # type: ignore


app = Flask(__name__)

# Halaman utama dengan antarmuka pengguna


@app.route('/')
def index():
    return render_template('index.html')

# Endpoint untuk menerima permintaan dari antarmuka pengguna


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response_ = index.query(user_input)
    response = chat_with_gpt3(user_input, api_key)

    return jsonify({'response': response_.response})


if __name__ == '__main__':
    app.run(debug=True)
