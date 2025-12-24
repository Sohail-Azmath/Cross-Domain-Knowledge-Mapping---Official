from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import jwt
import datetime
import os

app = Flask(__name__)
CORS(app)

SECRET_KEY = "abghy57ghhbghyju787hgyhluck"

users = {}  

# Serve frontend HTML
@app.get("/")
def home():
    return send_from_directory('.', 'login_page.html')

@app.get("/register-page")
def reg_page():
    return send_from_directory('.', 'Registration.html')

@app.get("/forgot-page")
def forgot_page():
    return send_from_directory('.', 'forgot.html')

# Your API routes remain the same
@app.post("/register")
def register():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    usertype = data.get("usertype")

    if username in users:
        return jsonify({"msg": "User already exists"}), 400

    users[username] = {"password": password, "type": usertype}
    return jsonify({"msg": "Registered successfully"}), 200

@app.post("/login")
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if username not in users or users[username]["password"] != password:
        return jsonify({"msg": "Invalid username or password"}), 400

    token = jwt.encode(
        {"user": username, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)},
        SECRET_KEY,
        algorithm="HS256"
    )
    response = jsonify({"token": token})
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

   # return jsonify({"token": token}), 200

@app.post("/reset-password")
def reset_password():
    data = request.json
    username = data.get("username")
    newpass = data.get("newpass")

    if username not in users:
        return jsonify({"msg": "User not found"}), 404

    users[username]["password"] = newpass
    return jsonify({"msg": "Password reset successful"}), 200

@app.get("/verify")
def verify():
    token = request.headers.get("Authorization")
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return jsonify({"user": decoded["user"]})
    except:
        return jsonify({"msg": "Invalid or expired token"}), 401
# -----------------------------
# NEW: Chat API
# -----------------------------
@app.post("/chat")
def chat():
    data = request.json
    user_msg = data.get("message", "")

    if not user_msg:
        return jsonify({"reply": "Please send a valid message."})

    # Example AI response
    # Replace this with your actual AI / semantic search / model logic
    bot_reply = f"Echo: {user_msg}"

    return jsonify({"reply": bot_reply})


if __name__ == "__main__":
    app.run(debug=True)
