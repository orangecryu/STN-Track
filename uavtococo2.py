import json
import pandas as pd
import os



classList = ["car", "truck", "bus"]
# By default, coco dataset starts index of categories from 1
PRE_DEFINE_CATEGORIES = {key: idx + 1 for idx, key in enumerate(classList)}
# print(PRE_DEFINE_CATEGORIES)


txtdir = 'D:/xxk/3/GT'
out_json_file = "out.json"

json_dict = {"images": [], "type": "instances", "annotations": [],
             "categories": []}
# json格式如下，我们要写这总的4个部分，先把最简单的categories写了吧
# {"images": [], "type": "instances", "annotations": [], "categories": []}

# 先把categories部分写完
for cate, cid in PRE_DEFINE_CATEGORIES.items():
    cat = {'supercategory': cate, 'id': cid, 'name': cate}
    json_dict['categories'].append(cat)

def get_annot_data(txt_file):
    '''Read annotation into a Pandas dataframe'''
    annot_data =  pd.read_csv(txt_file, delimiter=',', names=['<frame_index>','<target_id>','<bbox_left>','<bbox_top>','<bbox_width>','<bbox_height>','<out-of-view>','<occlusion>','<object_category>'])
    return annot_data



# 记录键值对为后面id进行使用
dict_imagename2id = {}

# 先实现images部分
# begin
imageid= 0
imagesdir = 'D:/xxk/3/4'
for image_name in os.listdir(imagesdir):
    file_name = image_name
    width = 540
    height = 1024
    id = imageid
    imageid += 1
    dict_imagename2id[file_name[:-4]] = id
    image = {'file_name': file_name, 'height': height, 'width': width, 'id': id}
    json_dict['images'].append(image)
# end


# images部分写好了

# 接下来写annotations部分
bndid_change = 1
for txt_file in os.listdir(txtdir):
    if 'gt_whole' in txt_file:
        # print(txt_file)
        txtwholepath = txtdir + '/'+ txt_file
        # print(txtwholepath)
        with open(txtwholepath) as f:
            annot_data = get_annot_data(txtwholepath)
            # print(annot_data)
            # print(annot_data.loc[0])
            # for index in annot_data:
            #     print(index)
            # pandas按行输出
            for index, row in annot_data.iterrows():
                # print(index)  # 输出每行的索引值
                # 要加int不然会出现int64错误
                o_width = int(row['<bbox_width>'])
                o_height = int(row['<bbox_height>'])
                xmin = int(row['<bbox_left>'])
                ymin = int(row['<bbox_top>'])
                category_id = int(row['<object_category>'])
                bnd_id = int(bndid_change)
                # image_id需要建立一个map或者叫dict
                # 转化成000001
                imageindx = row['<frame_index>']
                k = str(imageindx)
                str1 = txt_file[0:5]
                str2 = k.zfill(6)
                dictfilename = str1 + '_' + str2
                image_id = int(dict_imagename2id[dictfilename])
                print(image_id)
                ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':image_id, 'bbox': [xmin, ymin, o_width, o_height],'category_id': category_id, 'id': bnd_id, 'ignore': 0,'segmentation': []}
                bndid_change += 1
                json_dict['annotations'].append(ann)

            # break

# about json
json_fp = open(out_json_file, 'w')
json_str = json.dumps(json_dict)
json_fp.write(json_str)
json_fp.close()

