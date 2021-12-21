import zipfile
with zipfile.ZipFile("rubbish_detection.zip", 'r') as zip_ref:
    zip_ref.extractall("./")