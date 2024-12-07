import os
import git
import re
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU (if available) for TensorFlow

class AICodeAssistant:
    def __init__(self, base_path, model=None):
        self.base_path = base_path
        self.model = model if model else self.load_model()

    def load_model(self):
        """
        Load a pre-trained model for command classification (e.g., TensorFlow model).
        If the model is not available, a new model can be trained.
        """
        try:
            model = tf.keras.models.load_model('command_classifier_model.h5')
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            model = None
        return model

    def clone_repo(self, git_url_base, platform, repo_name):
        """
        Clone a repository from the specified platform.
        If the repository already exists, pull the latest changes and ensure it checks out the correct default branch.
        """
        try:
            repo_path = os.path.join(self.base_path, repo_name)

            if os.path.exists(repo_path):
                print(f"Repository {repo_name} already exists, pulling the latest changes.")
                repo = git.Repo(repo_path)
                origin = repo.remotes.origin
                origin.fetch()  # Fetch the latest changes
                
                # Get the default branch of the repository (e.g., "main" or "master")
                remote_refs = repo.remotes.origin.refs
                default_branch = next(ref for ref in remote_refs if ref.name.endswith('HEAD')).name.split('/')[-1]
                
                repo.git.checkout(default_branch)  # Checkout the default branch
                origin.pull()  # Pull the latest changes
                print(f"Repository {repo_name} is up-to-date.")
            else:
                print(f"Cloning repository {repo_name} from {platform}...")
                repo_url = f"{git_url_base}/{platform}/{repo_name}.git"
                git.Repo.clone_from(repo_url, repo_path)
                print(f"Repository {repo_name} cloned successfully.")
        except Exception as e:
            print(f"Error cloning {repo_name}: {str(e)}")
        return repo_path


    def read_code_files(self, repo_path):
        """
        Recursively read all code files in the repository.
        """
        code_lines = []
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith(('.py', '.js', '.java', '.cpp', '.txt')):  # You can add more file extensions as needed
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        code_lines.extend(lines)
        return code_lines

    def process_code_for_commands(self, code_lines):
        """
        Process the code and look for commands.
        Lines starting with "//" are considered information, not commands.
        """
        commands = []
        information = []  # Store information lines separately
        command_pattern = re.compile(r'//\s*(.*)')  # Match lines starting with "//"
        
        for line in code_lines:
            match = command_pattern.match(line)
            if match:
                information.append(match.group(1).strip())  # Treat as information
            else:
                # If not a comment, treat it as a command (you can customize this further)
                command = line.strip()
                if command:  # Only process non-empty lines
                    commands.append(command)

        return commands, information

    def classify_command(self, command):
        """
        Use the TensorFlow model to classify the command.
        """
        if self.model:
            input_data = np.array([command])
            prediction = self.model.predict(input_data)
            predicted_class = np.argmax(prediction, axis=1)
            return predicted_class
        else:
            return None

    def handle_command(self, command):
        """
        Handle the command, e.g., by printing or processing.
        """
        command_class = self.classify_command(command)
        if command_class is not None:
            print(f"Command classified as class {command_class[0]}: {command}")
        else:
            print(f"Unknown command: {command}")

    def analyze_repositories(self, repos):
        """
        Analyze each repository and process code for commands.
        """
        for platform, platform_repos in repos.items():
            for repo_group, repo_names in platform_repos.items():
                for repo_name in repo_names:
                    repo_path = self.clone_repo('https://github.com' if platform == 'github' else 'https://gitlab.com', platform, f'{repo_group}/{repo_name}')
                    code_lines = self.read_code_files(repo_path)
                    commands = self.process_code_for_commands(code_lines)
                    for command in commands:
                        self.handle_command(command)

# Example code for training a simple model
def train_command_classifier_model():
    commands = ['run server', 'build project', 'test unit', 'deploy server', 'check status']
    labels = ['action', 'action', 'action', 'action', 'status']

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    X = np.array(commands)
    y = labels_encoded

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(1,), dtype=tf.string),
        tf.keras.layers.TextVectorization(output_mode='int'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(len(set(labels)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10)
    model.save('command_classifier_model.h5')

if __name__ == "__main__":
    # Uncomment to train model
    # train_command_classifier_model()

    base_path = './git'
    repos = {
        'github': {
            'openai': ['openai-cookbook'],
            'mcengine': ['mcengine', 'currency']
        }
    }

    assistant = AICodeAssistant(base_path)
    assistant.analyze_repositories(repos)
