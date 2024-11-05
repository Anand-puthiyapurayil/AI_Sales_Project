import os
from pathlib import Path

# Project name
project_name = "salespredictor"

# Define the list of files and directories for the project structure
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_processing.py",
    f"src/{project_name}/components/model_training.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/components/model_inference.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/mlflow_utils.py",  # MLflow utility script
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/train_pipeline.py",
    f"src/{project_name}/api/__init__.py",
    f"src/{project_name}/api/main.py",           # FastAPI main file
    f"src/{project_name}/dashboard/__init__.py",
    f"src/{project_name}/dashboard/app.py",      # Streamlit app
    f"config/config.yaml",                        # Placeholder for general config
    "requirements.txt",                          # Dependencies, including MLflow and Streamlit
    "setup.py",                                  # Setup script for packaging
    "research/trials.ipynb",                     # Jupyter notebook for experiments
    "templates/index.html",                      # Streamlit templates if needed
    ".streamlit/config.toml"                     # Streamlit-specific config
]

# Function to create directories and files
def create_structure(base_path, structure):
    for filepath in structure:
        filepath = Path(filepath)
        filedir, filename = os.path.split(filepath)

        # Create directory if it does not exist
        if filedir:
            os.makedirs(filedir, exist_ok=True)

        # Create the file if it doesn't exist or is empty
        if not filepath.exists() or filepath.stat().st_size == 0:
            with open(filepath, "w") as f:
                pass

# Run the setup
if __name__ == "__main__":
    create_structure(project_name, list_of_files)
    print("Project structure has been created.")
