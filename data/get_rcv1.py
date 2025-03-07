# RCV1 Downloader (There's no guarantee that this program will work.)

import os
import subprocess
import requests
from tqdm import tqdm
import numpy as np

label_dict = {}

def get_num_of_doc(path):
    cmd = "cat " + path + " | grep .W | wc -l "
    output = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout
    return int(output.decode("utf8").split(" ")[0])

def download_file(url, filename, filesize):
    with open(filename + ".gz", "wb") as file:
        pbar = tqdm(total=filesize, unit="B", unit_scale=True)
        pbar.set_description("Downloading " + filename[16:] + ".gz")
        for chunk in requests.get(url + filename + ".gz", stream=True).iter_content(chunk_size=1024):
            file.write(chunk)
            pbar.update(len(chunk))
        pbar.close()

    cmd = ["gzip", "-d", filename + ".gz"]
    subprocess.run(cmd, stdout=subprocess.PIPE)

def get_labels(doc_id, label_dict):
    labels = label_dict.get(doc_id, [])
    return " ".join(labels)

url = "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/"

files = [
    ("lyrl2004_tokens_train.dat", 5108963),
    ("lyrl2004_tokens_test_pt0.dat", 44734992),
    ("lyrl2004_tokens_test_pt1.dat", 45595102),
    ("lyrl2004_tokens_test_pt2.dat", 44507510),
    ("lyrl2004_tokens_test_pt3.dat", 42052117),
]

print("This program downloads files from '" + url[:-1] + "'.")

# Download and process the RCV1-v2 dataset
for filename, filesize in files:
    download_file(url, filename, filesize)

    num_of_doc = get_num_of_doc(filename)

    with open(filename) as f:
        flag = False
        buf = []
        doc_id = []
        datafile = tqdm(f, total=num_of_doc, unit="Docs")
        datafile.set_description("Processing " + filename[16:])
        for i in f:
            if (".I" in i) and (not flag):
                doc_id.append(i.replace(".I ", "")[:-1])
                flag = True
            elif ".I" in i:
                doc_id.append(i.replace(".I ", "")[:-1])
                # Process the document here
                labels = get_labels(doc_id[-2], label_dict)  # Replace with actual label processing
                text = " ".join(buf).replace("\n", "").replace(".W", "")

                output = doc_id[-2] + "\t" + labels + "\t" + text[1:-1] + "\n"

                with open(filename + ".out", "a") as f_output:
                    f_output.write(output)
                    buf = []
                datafile.update()
            else:
                buf.append(i)
        else:
            # Process the last document
            labels = get_labels(doc_id[-1], label_dict)  # Replace with actual label processing
            text = " ".join(buf).replace("\n", "").replace(".W", "") + "\n"

            output = doc_id[-1] + "\t" + labels + "\t" + text

            with open(filename + ".out", "a") as f_output:
                f_output.write(output)
                buf = []

            datafile.update()

    datafile.close()
    os.remove(filename)

files = [i[0] + ".out" for i in files]

cmd = ["mv", files[0], "train_org.txt"]
subprocess.run(cmd, stdout=subprocess.PIPE)

cmd = ["cat " + " ".join(files[1:]) + " > test.txt"]
subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
[os.remove(i) for i in files[1:]]

# Download and process the RCV1-v2 labels
label_url = "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm"
label_file = "rcv1-v2.labels"
download_file(label_url, label_file, 0)  # Adjust the filesize if known

with open(label_file) as f:
    for line in f:
        parts = line.strip().split()
        doc_id = parts[0]
        labels = parts[1:]
        label_dict[doc_id] = labels

os.remove(label_file)