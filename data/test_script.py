s = "Hello! I am a test script."

data_dir = "/scratch2/vaibhav/"

with open(data_dir + "test_script.txt", "w") as f:
    f.write(s)