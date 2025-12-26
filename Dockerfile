FROM python:3.10-slim

WORKDIR /app

# Copy dependencies first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLP models
RUN python -m nltk.downloader punkt stopwords
RUN python -m spacy download en_core_web_sm

# Copy project files
COPY . .

# Expose ports
EXPOSE 5000
EXPOSE 8501

# Run backend + frontend
CMD sh -c "python ui/server.py & exec streamlit run ui/app.py --server.address=0.0.0.0 --server.port=8501"

