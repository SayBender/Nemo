import json
import numpy as np
import pandas as pd

def convert_coco_json_to_csv(filename,out_file=None):
    #filename: path to .json file
    s = json.load(open(filename, 'r'))
    if out_file is None:
        out_file = filename[:-5] + '.csv'
    print("saving to: ", out_file)
    out = open(out_file, 'w')
    out.write('id,label,x1,y1,x2,y2\n')

    all_ids = []
    for im in s['images']:
        all_ids.append(im['id'])

    all_ids_ann = []
    for ann in s['annotations']:
        image_id = ann['image_id']
        all_ids_ann.append(image_id)
        x1 = ann['bbox'][0]
        x2 = ann['bbox'][0] + ann['bbox'][2]
        y1 = ann['bbox'][1]
        y2 = ann['bbox'][1] + ann['bbox'][3]
        label = ann['category_id']
        out.write('{},{},{},{},{},{}\n'.format(image_id,label,x1, y1, x2, y2 ))

    all_ids = set(all_ids)
    all_ids_ann = set(all_ids_ann)
    no_annotations = list(all_ids - all_ids_ann)
    # Output images without any annotations
    for image_id in no_annotations:
        out.write('{},{},{},{},{},{}\n'.format(image_id, -1, -1, -1, -1, -1))
    out.close()

    # Sort file by image id
    s1 = pd.read_csv(out_file)
    s1.sort_values('id', inplace=True)
    s1.to_csv(out_file, index=False)
    