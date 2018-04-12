import os
file_name = "document1.txt"
script_dir = os.path.dirname(__file__)
rel_path = "documents/"+file_name
abs_file_path = os.path.join(script_dir, rel_path)
