---

# CalorAI - Backend Chatbot (FastAPI)

**CalorAI** is a backend chatbot built with **FastAPI** in Python, designed for nutritional prediction and interaction. The system is modular and follows clean architecture principles for easy maintenance and scalability.

---

## 📚 Project Overview

CalorAI provides nutritional predictions and human-like chatbot interactions. The backend performs the following steps:

1. **Classification**: A custom model based on **RoBERTa** determines whether the user input refers to a food item or not.
2. **Nutrient Regression**: If the input is classified as food, a regression model predicts nutritional information including:
   - Calories (kcal)
   - Fats (g)
   - Carbohydrates (g)
   - Proteins (g)
3. **Prompt Engineering**: The results are passed to **Google Gemini** for generating more natural, humanized chatbot responses.

---

## 🗂️ Project Structure

```plaintext
backend/
├── app/
│   ├── core/          # Configurations and utilities
│   ├── db/            # Database connections and models
│   ├── routers/       # API routers
│   ├── services/      # Business logic and services
│   ├── __init__.py
│   └── main.py        # FastAPI application instance
├── .env               # Environment variables
├── __init__.py
├── docker-compose.yml
├── Dockerfile
├── README.md
├── run.py             # Main entrypoint to run the backend
└── Pipfile
```

---

## 🚀 Running the Backend

### 1. Install pipenv

```bash
pip install pipenv
```

### 2. Install project dependencies

From the `backend/` directory:

```bash
pipenv install --dev
```

### 3. Activate the virtual environment

```bash
pipenv shell
```

### 4. Start the backend

```bash
python run.py
```

---

## 🔐 Environment Variables

The project requires API keys to interact with external services. Create a `.env` file at the root of `backend/` and include:

```dotenv
HUGGINGFACE_API_KEY=your_huggingface_token
GEMINI_API_KEY=your_google_gemini_token
```

### Notes:
- **AI models** are hosted on **HuggingFace Hub** due to size limitations on GitHub. You will need a HuggingFace API key to download them.
- **Google Gemini** is used for prompt engineering and conversational generation, requiring a valid Gemini API key.

---

## ➕ Adding Dependencies

- **Production dependency**:

```bash
pipenv install dependency-name
```

- **Development dependency**:

```bash
pipenv install --dev dependency-name
```

The `Pipfile` will be updated automatically.

---

## 🐳 Running with Docker

You can also spin up the backend using Docker:

```bash
docker-compose up --build
```

---

## 🌐 API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 📝 Notes

- Sensitive data (API keys, tokens, etc.) should remain in the `.env` file and **must not be committed to Git**.
- Modular structure with clear separation between **core**, **db**, **routers**, and **services**, following clean coding practices.

