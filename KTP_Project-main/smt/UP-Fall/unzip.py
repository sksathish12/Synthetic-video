import os
import zipfile
import re

class DatasetUnzipper:
    def __init__(self, base_folder):
        self.base_folder = base_folder

    def _get_subfolders(self, folder):
        # Get the folder names and sort them numerically
        subfolders = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        subfolders.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        return subfolders

    def get_subject_folders(self):
        return self._get_subfolders(self.base_folder)

    def get_activity_folders(self, subject_folder):
        return self._get_subfolders(os.path.join(self.base_folder, subject_folder))

    def get_trial_folders(self, subject_folder, activity_folder):
        return self._get_subfolders(os.path.join(self.base_folder, subject_folder, activity_folder))

    def unzip_and_delete(self):
        for subject_folder in self.get_subject_folders():
            for activity_folder in self.get_activity_folders(subject_folder):
                for trial_folder in self.get_trial_folders(subject_folder, activity_folder):
                    trial_path = os.path.join(self.base_folder, subject_folder, activity_folder, trial_folder)
                    for item in os.listdir(trial_path):
                        if item.endswith(".zip"):
                            zip_file_path = os.path.join(trial_path, item)
                            extract_folder = os.path.join(trial_path, item[:-4])  # using the zip filename without '.zip' as the folder name
                            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                                zip_ref.extractall(extract_folder)
                            os.remove(zip_file_path)
                            print(f"Unzipped to {extract_folder} and deleted: {zip_file_path}")

if __name__ == "__main__":
    base_folder = '../dataset/UP-Fall'
    unzipper = DatasetUnzipper(base_folder)
    unzipper.unzip_and_delete()