from flask import Flask, render_template, request, jsonify, Response
from ollama import Client
from dotenv import load_dotenv
from datetime import datetime
import os

load_dotenv()

app = Flask(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi4-mini")

client = Client(host=OLLAMA_HOST)

messages = [
    {
        "role": "system",
        "content": "Eres un asistente útil y respondes en español."
    }
]


def save_chat(user_msg, ai_msg):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"chats/chat_{now}.txt"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"Usuario: {user_msg}\n")
        f.write(f"IA: {ai_msg}\n\n")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Mensaje vacío"}), 400

    messages.append({"role": "user", "content": user_message})

    def generate():
        assistant_message = ""

        try:
            for part in client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                stream=True
            ):
                content = part["message"]["content"]
                assistant_message += content
                yield content

            messages.append({
                "role": "assistant",
                "content": assistant_message
            })

            save_chat(user_message, assistant_message)

        except Exception as e:
            yield f"\nError: {str(e)}"

    return Response(generate(), mimetype="text/plain")


@app.route("/clear", methods=["POST"])
def clear_chat():
    global messages

    messages = [
        {
            "role": "system",
            "content": "Eres un asistente útil y respondes en español."
        }
    ]

    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)