FROM jupyter/datascience-notebook

WORKDIR /home/jovyan/work

# Copy requirements and install dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install the package in development mode
COPY . /home/jovyan/work/
RUN pip install -e /home/jovyan/work/

# Set environment variables
ENV PYTHONPATH=/home/jovyan/work

# Note about .env file
# The .env file is mounted via docker-compose.yml, not copied here
# This prevents sensitive credentials from being built into the image
