FROM python:3.9-slim

# Working Directory
WORKDIR /streamlit

# Copy source code to working directory
COPY . streamlit.py /streamlit/

# Install packages from requirements.txt

RUN pip install --no-cache-dir --upgrade pip &&\
    pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

CMD python streamlit.py

