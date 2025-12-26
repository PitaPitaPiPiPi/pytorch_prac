import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.log_path = os.path.join(log_dir, filename)

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        print(formatted_message)
        with open(self.log_path, "a") as f:
            f.write(formatted_message + "\n")