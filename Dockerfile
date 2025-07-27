FROM python:3.10-slim

RUN apt-get update && apt-get install -y gcc portaudio19-dev && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Copy requirements.txt and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the CureConnect folder into the image
COPY CureConnect ./CureConnect

# Set working directory to CureConnect
WORKDIR /CureConnect

# Expose Streamlit default port (optional but good)
EXPOSE 8501

# Run Streamlit app with relative path inside CureConnect
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
