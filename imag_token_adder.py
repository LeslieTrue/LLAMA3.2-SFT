# read in a json and add <image>\n to the beginning of each cov.
import json
import os
import sys

path = "/group/mayi/tianzhe/project/RL-MLLM/data_collection/EQN_Data_image_200k/general_points.json"

data = json.load(open(path, 'r'))
for i in range(len(data)):
    data[i]['conversations'][0]['value'] = "<image>\n" + data[i]['conversations'][0]['value']
    

# write to a new file
output_path = "/group/mayi/tianzhe/project/RL-MLLM/data_collection/EQN_Data_image_200k/general_points_image.json"
with open(output_path, 'w') as f:
    json.dump(data, f, indent=4)