import huggingface_hub
from llama_cpp import Llama
from flask import Flask, request, jsonify

app = Flask(__name__)

llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
    filename="*q5_0.gguf",
    verbose=False
)

@app.route('/chat', methods=['GET'])
def chat():
    user_message = request.args.get('message')
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    a = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "hello"},
            {"role": "user", "content": user_message}
        ]
    )

    return jsonify({"response": a})
