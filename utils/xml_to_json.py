import json
import math
from io import BytesIO
import pickle

import pandas as pd
import requests
import xml_to_dict
import json
from PIL import Image
from tqdm import tqdm

total_data = 0
total_correct = 0
total_no_renderpos = 0
total_no_text = 0
total_wrong_image = 0
total_wrong_xml = 0

def process_bbox(XML_BBOX, IM_SIZE, SHEET_SIZE, angle, center):
    RATIO = IM_SIZE[0] / SHEET_SIZE[0]
    x1, y1, x2, y2 = map(float, XML_BBOX)
    x1, y1, x2, y2 = (x1 * RATIO, y1 * RATIO, x2 * RATIO, y2 * RATIO)
    center = (center[0] * RATIO, center[1] * RATIO)

    if angle != 0:
        angle = 360 - angle
        angle = math.radians(angle)
        # Calculate the center point of the bbox
        center_x, center_y = center
        # Calculate the distance from the center to each corner of the bbox
        distance_x = (x1 - center_x)
        distance_y = (y1 - center_y)
        # Apply rotation to the distances
        new_distance_x = distance_x * math.cos(angle) - distance_y * math.sin(angle)
        new_distance_y = distance_x * math.sin(angle) + distance_y * math.cos(angle)
        # Calculate the new corners after rotation
        x1 = center_x + new_distance_x
        y1 = center_y + new_distance_y
        x2 = center_x - new_distance_x
        y2 = center_y - new_distance_y

    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

    return x1, y1, x2, y2

def get_render_bbox(text):
    global total_no_renderpos
    if "RenderPos" not in text or text['RenderPos'] == None:
        total_no_renderpos += 1
        print("No RenderPos", end=" ")
        return None
    render_pos = json.loads(text['RenderPos'])
    render_bbox = []

    left, top, right, bottom = map(float, text['Position'].values())

    for render in render_pos['c']:
        try:
            x, a, w, y = map(float, [render['x'], render['a'], render['w'], render['y']])
        except:
            print("Wrong render pos", end=" ")
            return None
        left_ = left + x
        right_ = left_ + w
        bottom_ = top + y
        top_ = bottom_ - a

        render_bbox.append((left_, top_, right_, bottom_))

    return render_bbox

def get_bbox(render_bbox):
    min_x = min([x[0] for x in render_bbox])
    min_y = min([x[1] for x in render_bbox])
    max_x = max([x[2] for x in render_bbox])
    max_y = max([x[3] for x in render_bbox])

    return min_x, min_y, max_x, max_y

# 오류 처리 함수 
def dictIntoList(TEXT_TagData) :
  if type(TEXT_TagData) is dict :
    temp = []
    temp.append(TEXT_TagData)
    TEXT_TagData = temp
  
  return TEXT_TagData

def process_xml_dict(xml_dict, thumbnail):
    global total_no_text
    processed_json = {}
    processed_json['form'] = []

    SHEET_SIZE = tuple(map(int, xml_dict['SHEET']['SHEETSIZE'].values()))
    IM_SIZE = thumbnail.size

    # 오류: TEXT, RendorPos, TEXT안 text 데이터가 list가 아닌 dict인 경우
    if 'TEXT' not in xml_dict['SHEET']:
        total_no_text += 1
        print("No TEXT", end=" ")
        return None

    TEXT_TagData = xml_dict['SHEET']['TEXT']
    TEXT_TagData = dictIntoList(TEXT_TagData)       # TEXT안 text 데이터 형태가 list가 아닌 dict인 오류 처리 

    # Process XML to json
    for idx, text in enumerate(TEXT_TagData):
        left, top, right, bottom = map(float, text['Position'].values())
        center = ((left + right) / 2, (top + bottom) / 2)

        render_bbox = get_render_bbox(text)
        if render_bbox is None or len(render_bbox) == 0: return None

        XML_BBOX = get_bbox(render_bbox)

        t = text['Text']
        x1, y1, x2, y2 = process_bbox(XML_BBOX, IM_SIZE, SHEET_SIZE, int(float(text['@Rotate'])), center)

        processed_json['form'].append({
            "text": t,
            "box": [x1, y1, x2, y2],
            "font_id": text['Font']['@FamilyIdx'],
            "font_size": text['Font']['@Size'],
            "style": {
                "bold": text['Font']['Style']['@Bold'] == 'true',
                "italic": text['Font']['Style']['@Italic'] == 'true',
                "strikeout": text['Font']['Style']['@Strikeout'] == 'true',
                "underline": text['Font']['Style']['@Underline'] == 'true'
            },
            "linespace": text['Font']['@LineSpace'],
            "opacity": text['@Opacity'],
            "rotate": text['@Rotate'],
            "id": idx,
            "sheet_size" : SHEET_SIZE
        })

        processed_json['form'][-1]['words'] = []

        render_pos = json.loads(text['RenderPos'])

        for j, bbox in enumerate(render_bbox):
            x1_, y1_, x2_, y2_ = process_bbox(bbox, IM_SIZE, SHEET_SIZE, int(float(text['@Rotate'])), center)
            color = render_pos['c'][j]['f']
            if color.startswith('rgba'):
                color = color[5:-1]
            else: color = color[4:-1]
            color = list(map(int, color.split(",")))
            processed_json['form'][-1]['words'].append({
                "text": render_pos['c'][j]['t'],
                "box": [x1_, y1_, x2_, y2_],
                "font_size": float(render_pos['c'][j]['s']),
                "letter_spacing": float(render_pos['c'][j]['ds']),
                #"font_id": int(render_pos['c'][j]['yd']),
                "color": color
            })

    return processed_json

def process_xml(sheet_url, thumbnail_url):
    global total_wrong_image
    global total_wrong_xml
    try:
        sample_thumbnail = Image.open(BytesIO(requests.get(thumbnail_url).content))
    except:
        print(f"Error at loading image, {thumbnail_url}", end = " ")
        total_wrong_image += 1
        return None
    try:
        sample_xml = requests.get(sheet_url).content.decode("utf-8")
    except:
        print(f"Error at loading xml, {sheet_url}", end = " ")
        total_wrong_xml += 1
        return None
    sample_json = xml_to_dict.XMLtoDict().parse(sample_xml)

    processed_json = process_xml_dict(sample_json, sample_thumbnail)

    return processed_json

def hex_to_rgb(hex_string):
    # remove '#' and convert to RGB
    return [int(hex_string[idx:idx+2], 16) for idx in (1, 3, 5)]


def make_sample_json(xml_sample_loc):
    # Read sample CSV and download thumbnail, XML

    global total_data
    global total_correct
    df = pd.read_csv(xml_sample_loc)
    df = df.reset_index()  # make sure indexes pair with number of rows
    for idx, row in df.iterrows():
        if row['category'] == '프레젠테이션' and row['language'] == 'en':
            total_data += 1
            #row = df.iloc[955]

            processed_json = process_xml(row['sheet_url'], row['thumbnail_url'])
            if processed_json is None : 
                print("idx:", idx)
                continue
            processed_json['template_idx'] = int(row['tempate_idx'])
            processed_json['sheet_url'] = row['sheet_url']
            processed_json['thumbnail_url'] = row['thumbnail_url']
            processed_json['page_no'] = int(row['page_no'])
            processed_json['title'] = row['title']
            processed_json['primary_colors'] = [hex_to_rgb(color) for color in row['primary_colors'][1:-1].split(",")]
            processed_json['primary_color_weights'] = [float(weight) for weight in row['primary_color_weights'][1:-1].split(",")]
            processed_json['language'] = row['language']
            processed_json['keyword'] = list(set(row['keyword'].split("|")))
            print("Correct file idx: ", idx)
            total_correct += 1
            filename = f"../new_data/json_data/processed_{idx}.pickle"
            with open(filename, "wb") as file_:
                pickle.dump(processed_json, file_)
            sample_thumbnail = Image.open(BytesIO(requests.get(processed_json['thumbnail_url']).content))
            sample_thumbnail = sample_thumbnail.save(f"../new_data/images/image_{idx}.png")
        

    #   except:
    #     with open("error_index.txt", "a") as file:
    #       file.write(f"error in {str(idx+59525)}" + "\n")
    print()
    print("total data: ", total_data)
    print("total correct: ", total_correct)
    print("total no renderpos: ", total_no_renderpos)
    print("total no text: ", total_no_text)
    print("total wrong image: ", total_wrong_image)
    print("total wrong xml: ", total_wrong_xml)


if __name__ == "__main__":
    make_sample_json('../data/sample_20230703-1.csv') 
 
