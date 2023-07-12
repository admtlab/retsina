FROM python:3.10

# Copy all the files to the container
COPY . .

# Set a directory for the app
WORKDIR /code-workspace

# Install Dependencies from requirements file
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install pymrmr==0.1.11

EXPOSE 5000

# Run the Python Program
ENTRYPOINT ["python", "run_experiments.py"]
