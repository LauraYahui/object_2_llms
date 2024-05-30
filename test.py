import unittest
import subprocess
import os

class TestTrainScript(unittest.TestCase):
    
    def run_train_script(self, task, model_name_or_path, output_dir):
        result = subprocess.run(
            [
                "python", "train.py", 
                "--task", task, 
                "--model_name_or_path", model_name_or_path, 
                "--output_dir", output_dir,
                "--num_train_epochs", "1",  
                "--batch_size", "2"         
            ], 
            capture_output=True, 
            text=True
        )
        return result

    def test_retrieval_task(self):
        task = "retrieval"
        model_name_or_path = "bert-base-uncased"
        output_dir = "./test_retrieval_results"
        result = self.run_train_script(task, model_name_or_path, output_dir)
        self.assertIn("Training set size:", result.stdout)
        self.assertIn("Validation set size:", result.stdout)
        self.assertIn("Evaluation Accuracy:", result.stdout)
        if os.path.exists(output_dir):
            os.rmdir(output_dir)

    def test_summarization_task(self):
        task = "summarization"
        model_name_or_path = "facebook/bart-large"
        output_dir = "./test_summarization_results"
        result = self.run_train_script(task, model_name_or_path, output_dir)
        self.assertIn("Training set size:", result.stdout)
        self.assertIn("Validation set size:", result.stdout)
        self.assertIn("ROUGE Scores:", result.stdout)
        if os.path.exists(output_dir):
            os.rmdir(output_dir)

if __name__ == "__main__":
    unittest.main()
