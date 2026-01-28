import pathlib

home_dir = str(pathlib.Path.cwd()).replace("\\", "/") + "/"
project_dir = str(pathlib.Path.cwd().parent).replace("\\", "/") + "/"

input_dir = home_dir + "input/"
model_dir = home_dir + "model/"
output_dir = home_dir + "output/"
