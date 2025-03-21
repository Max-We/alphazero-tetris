# Start from NVIDIA JAX container
FROM nvcr.io/nvidia/jax:24.10-py3

# Install system dependencies for pygraphviz and git
RUN apt-get update && apt-get install -y \
    graphviz \
    graphviz-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Configure Poetry to not create virtual environments
RUN poetry config virtualenvs.create false

# Set working directory
WORKDIR /app

# Copy poetry files
COPY pyproject.toml poetry.lock* ./

# Install dependencies (if lock file exists)
RUN poetry lock
RUN poetry install --no-root --no-interaction

# Install opencv-python-headless
RUN poetry remove opencv-python || true
RUN poetry add opencv-python-headless

ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV JAX_PLATFORM_NAME="cpu"

# Copy the rest of the application
COPY . .

# Set the entrypoint to python
ENTRYPOINT ["python"]
