import os
import pandas as pd
import xml.etree.ElementTree as ET
import math
from itertools import chain
import numpy as np
from skimage import io
import json
import cv2
from pathlib import Path
from PIL import Image

def pascalvoc_xml2tf_csv():
    path = './annotation_data'
    xml_list = []
    for file in os.scandir(path):
        if file.is_file() and file.name.endswith(('.xml')):
            xml_file = os.path.join(path, file.name)
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                value = (root.find('filename').text,
                        int(root.find('size')[0].text),
                        int(root.find('size')[1].text),
                        member[0].text,
                        int(member[4][0].text),
                        int(member[4][1].text),
                        int(member[4][2].text),
                        int(member[4][3].text) )
                xml_list.append(value)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df.to_csv('./output/pascalvoc_xml2tf_csv.csv', index=None)

#vgg to coco
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def polygon_vgg_json2coco_json():
    class_keyword = "class"
    outfile = './output/polygon_vgg_json2coco_json.json'
    vgg_path = './annotation_data/'+ os.listdir('annotation_data')[0]
    print(vgg_path)
    dataset_dir = './image_data'
    print(class_keyword)
    with open(vgg_path) as f:
        vgg = json.load(f)

    images_ids_dict = {}
    images_info = []
    for i,v in enumerate(vgg.values()):
        image_path = os.path.join(dataset_dir, v['filename'])
        #print(image_path)
        if os.path.exists(image_path):
            images_ids_dict[v["filename"]] = i
            image = io.imread(image_path)
            height, width = image.shape[:2]  
            images_info.append({"file_name": v["filename"], "id": i, "width": width, "height": height})
        else:
            print("fzil")
    #classes = {class_keyword} | {r["region_attributes"][class_keyword] for v in vgg.values() for r in v["regions"]
    #                         if class_keyword in r["region_attributes"]}
    classes = {r["region_attributes"][class_keyword] for v in vgg.values() for r in v["regions"]
                             if class_keyword in r["region_attributes"]}
    print("classes::", classes)
    category_ids_dict = {c: i for i, c in enumerate(classes, 1)}
    categories = [{"supercategory": class_keyword, "id": v, "name": k} for k, v in category_ids_dict.items()]
    print(categories)
    annotations = []
    suffix_zeros = math.ceil(math.log10(len(vgg)))
    print(suffix_zeros,"hello")
    print(len(images_info))
    for i, v in enumerate(vgg.values()):
        image_path = os.path.join(dataset_dir, v['filename'])
        if os.path.exists(image_path):
        #for j, r in enumerate(v["regions"].values()):
            for j, r in enumerate(v["regions"]):

                if class_keyword in r["region_attributes"]:
                    print("class_keyword:", class_keyword)
                    print('r["region_attributes"] ', r["region_attributes"])
                    x, y = r["shape_attributes"]["all_points_x"], r["shape_attributes"]["all_points_y"]
                    print("image_id:", images_ids_dict[v["filename"]])
                    print("cate id>>>", category_ids_dict[r["region_attributes"][class_keyword]])
                    annotations.append({
                        "segmentation": [list(chain.from_iterable(zip(x, y)))],
                        "area": PolyArea(x, y),
                        "bbox": [min(x), min(y), max(x)-min(x), max(y)-min(y)],
                        "image_id": images_ids_dict[v["filename"]],
                        "category_id": category_ids_dict[r["region_attributes"][class_keyword]],
                        "id": int(f"{i:0>{suffix_zeros}}{j:0>{suffix_zeros}}"),
                        "iscrowd": 0
                        })

        coco = {
            "images": images_info,
            "categories": categories,
            "annotations": annotations
            }
    if outfile is None:
        outfile = vgg_path.replace(".json", "_coco2.json")
    with open(outfile, "w") as f:
        json.dump(coco, f)

def polygon_vgg_json2tf_csv():
    test_dir = Path("./annotation_data")
    csv_columns =['filename','width','height','class','xmin','ymin','xmax','ymax'] 
    main_lst = []
    image_path="./image_data"
    
    for input_file in test_dir.rglob('*.json'):
        print(input_file)
        input_dir = os.path.split(input_file)[0]
        with open(input_file,'r') as f:
            #print(input_file)
            sample_json = json.load(f)
            for j,item in enumerate(sample_json.values()):
                lst = []
                file_name = item['filename']
                try:
                    label = item['regions'][0]['region_attributes']['Class']
                except:
                    try:
                        label = item['regions'][1]['region_attributes']['Class']
                    except:
                        try:
                            label = item['regions'][0]['region_attributes']['class']
                        except:
                            try:
                                label = item['regions'][1]['region_attributes']['class']
                            except:
                                try:
                                    label = item['regions'][0]['region_attributes']['MAIN_CF']
                                except:
                                    label = item['regions'][1]['region_attributes']['MAIN_CF']
                print("file:", file_name)
                print(os.path.join(image_path, file_name))
                img  = cv2.imread(os.path.join(image_path, file_name))         
                height,width,_= img.shape
                try:
                    xmin = min(item['regions'][1]['shape_attributes']['all_points_x'])
                    ymin = min(item['regions'][1]['shape_attributes']['all_points_y'])
                    xmax = max(item['regions'][1]['shape_attributes']['all_points_x'])
                    ymax = max(item['regions'][1]['shape_attributes']['all_points_y'])
                except:
                    xmin = min(item['regions'][0]['shape_attributes']['all_points_x'])
                    ymin = min(item['regions'][0]['shape_attributes']['all_points_y'])
                    xmax = max(item['regions'][0]['shape_attributes']['all_points_x'])
                    ymax = max(item['regions'][0]['shape_attributes']['all_points_y'])
                lst = [file_name, width, height, label, xmin, ymin, xmax, ymax]
                main_lst.append(lst)
        

    data_df = pd.DataFrame(data=main_lst ,columns = csv_columns)

    data_df.to_csv("./output/polygon_vgg_json2tf_csv.csv", index=None)
    
def rectangle_vgg_json2tf_csv():
    test_dir = Path("./annotation_data")
    csv_columns =['filename','width','height','class','xmin','ymin','xmax','ymax'] 
    main_lst = []
    image_path = Path("./image_data")

    for input_file in test_dir.rglob('*.json'):
        print(input_file)
        input_dir = os.path.split(input_file)[0]
        with open(input_file,'r') as f:
            sample_json = json.load(f)
            for j,item in enumerate(sample_json.values()):
                lst = []
                file_name = item['filename']
                try:
                    label = item['regions'][0]['region_attributes']['Class']
                except:
                    try:
                        label = item['regions'][1]['region_attributes']['Class']
                    except:
                        try:
                            label = item['regions'][0]['region_attributes']['class']
                        except:
                            try:
                                label = item['regions'][1]['region_attributes']['class']
                            except:
                                continue
                            
                img  = cv2.imread(os.path.join(image_path,file_name))
                height,width,_ = img.shape
                try:
                    xmin = item['regions'][1]['shape_attributes']['x']
                    ymin = item['regions'][1]['shape_attributes']['y']
                    xmax = item['regions'][1]['shape_attributes']['width']+xmin
                    ymax = item['regions'][1]['shape_attributes']['height']+ymin
                except:
                    xmin = item['regions'][0]['shape_attributes']['x']
                    ymin = item['regions'][0]['shape_attributes']['y']
                    xmax = item['regions'][0]['shape_attributes']['width']+xmin
                    ymax = item['regions'][0]['shape_attributes']['height']+ymin
                lst = [file_name, width, height, label, xmin, ymin, xmax, ymax]
                main_lst.append(lst)

    data_df = pd.DataFrame(data=main_lst ,columns = csv_columns)
    data_df.to_csv("./output/rectangle_vgg_json2tf_csv.csv", index=None)
    
def pascalvoc_xml2polygon_vgg_json():
    xmlpath = './annotation_data/'
    img_path = './image_data/'
    out_path = './output/'
    files=os.listdir(xmlpath)
    data = {}
    for file in files:
        if file.endswith('xml'):
            tree=ET.parse(xmlpath+file)
            root=tree.getroot()
            objects=root.findall("object")
            width=root.find("size")[0].text
            height=width=root.find("size")[1].text
            imgname=root.findall("filename")[0].text
            regions_parent=[]
            imgsize = os.path.getsize('./image_data/'+imgname)
            imgkey = imgname + str(imgsize)
            jobj={imgkey:{"filename":imgname,"size":imgsize,"regions":[],"file_attributes":{}}}

            print(len(objects))
            for obj in objects:
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.findtext('xmin'))
                ymin = int(bndbox.findtext('ymin'))
                xmax = int(bndbox.findtext('xmax'))
                ymax = int(bndbox.findtext('ymax'))
                w = xmax - xmin
                h = ymax - ymin
                regions={"shape_attributes":{"name":"polygon","all_points_x":[xmin,xmin,xmax,xmax],"all_points_y":[ymin,ymax,ymax,ymin]},"region_attributes":{"class":"student"}}
                regions_parent.append(regions)
            jobj[imgkey]['regions']=regions_parent
        data[imgkey] = jobj[imgkey]
    files=json.dumps(data)
    with open(out_path+'pascalvoc_xml2polygon_vgg_json.json','w') as f1:
        f1.write(files)
    f1.close()             

def merge_json():
    full_json = {}
    path = './annotation_data/'
    output_path = './output/'
    filenames = os.listdir('annotation_data')
        
    for name in filenames:
        with open((path + name), 'r') as data_file:
            data = json.load(data_file)
        full_json.update(data)
    with open((output_path+'merge_json.json'), 'w') as fp:
        json.dump(full_json, fp)
        
def pascalvoc_xml2coco_json():
    xmlpath = './annotation_data/'
    img_path = './image_data/'
    out_path = './output/'
    files=os.listdir(xmlpath)
    data = {}
    for file in files:
        if file.endswith('xml'):
            tree=ET.parse(xmlpath+file)
            root=tree.getroot()
            objects=root.findall("object")
            width=root.find("size")[0].text
            height=width=root.find("size")[1].text
            imgname=root.findall("filename")[0].text
            regions_parent=[]
            imgsize = os.path.getsize('./image_data/'+imgname)
            imgkey = imgname + str(imgsize)
            jobj={imgkey:{"filename":imgname,"size":imgsize,"regions":[],"file_attributes":{}}}

            print(len(objects))
            for obj in objects:
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.findtext('xmin'))
                ymin = int(bndbox.findtext('ymin'))
                xmax = int(bndbox.findtext('xmax'))
                ymax = int(bndbox.findtext('ymax'))
                w = xmax - xmin
                h = ymax - ymin
                regions={"shape_attributes":{"name":"polygon","all_points_x":[xmin,xmin,xmax,xmax],"all_points_y":[ymin,ymax,ymax,ymin]},"region_attributes":{"class":"student"}}
                regions_parent.append(regions)
            jobj[imgkey]['regions']=regions_parent
        data[imgkey] = jobj[imgkey]
    files=json.dumps(data)
    
    with open(out_path+'pascalvoc_xml2coco_json.json','w') as f1:
        f1.write(files)
    f1.close()  
    
    class_keyword = "class"
    outfile = './output/pascalvoc_xml2coco_json.json'
    vgg_path = './output/'+ 'pascalvoc_xml2coco_json.json'
    print(vgg_path)
    dataset_dir = './image_data'
    print(class_keyword)
    with open(vgg_path) as f:
        vgg = json.load(f)

    images_ids_dict = {}
    images_info = []
    for i,v in enumerate(vgg.values()):
        image_path = os.path.join(dataset_dir, v['filename'])
        #print(image_path)
        if os.path.exists(image_path):
            images_ids_dict[v["filename"]] = i
            image = io.imread(image_path)
            height, width = image.shape[:2]  
            images_info.append({"file_name": v["filename"], "id": i, "width": width, "height": height})
        else:
            print("fzil")
    #classes = {class_keyword} | {r["region_attributes"][class_keyword] for v in vgg.values() for r in v["regions"]
    #                         if class_keyword in r["region_attributes"]}
    classes = {r["region_attributes"][class_keyword] for v in vgg.values() for r in v["regions"]
                             if class_keyword in r["region_attributes"]}
    print("classes::", classes)
    category_ids_dict = {c: i for i, c in enumerate(classes, 1)}
    categories = [{"supercategory": class_keyword, "id": v, "name": k} for k, v in category_ids_dict.items()]
    print(categories)
    annotations = []
    suffix_zeros = math.ceil(math.log10(len(vgg)))
    print(suffix_zeros,"hello")
    print(len(images_info))
    for i, v in enumerate(vgg.values()):
        image_path = os.path.join(dataset_dir, v['filename'])
        if os.path.exists(image_path):
        #for j, r in enumerate(v["regions"].values()):
            for j, r in enumerate(v["regions"]):

                if class_keyword in r["region_attributes"]:
                    print("class_keyword:", class_keyword)
                    print('r["region_attributes"] ', r["region_attributes"])
                    x, y = r["shape_attributes"]["all_points_x"], r["shape_attributes"]["all_points_y"]
                    print("image_id:", images_ids_dict[v["filename"]])
                    print("cate id>>>", category_ids_dict[r["region_attributes"][class_keyword]])
                    annotations.append({
                        "segmentation": [list(chain.from_iterable(zip(x, y)))],
                        "area": PolyArea(x, y),
                        "bbox": [min(x), min(y), max(x)-min(x), max(y)-min(y)],
                        "image_id": images_ids_dict[v["filename"]],
                        "category_id": category_ids_dict[r["region_attributes"][class_keyword]],
                        "id": int(f"{i:0>{suffix_zeros}}{j:0>{suffix_zeros}}"),
                        "iscrowd": 0
                        })

        coco = {
            "images": images_info,
            "categories": categories,
            "annotations": annotations
            }
    if outfile is None:
        outfile = vgg_path.replace(".json", "_coco2.json")
    with open(outfile, "w") as f:
        json.dump(coco, f)
    
def clean(dir):
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))