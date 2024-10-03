import pandas as pd
from glob import glob
from collections import Counter
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import imageio
import numpy as np
import warnings
from scipy.ndimage import gaussian_filter
import re
import os
import json
warnings.filterwarnings("ignore")

def generate_heatmap(base_image, width, height, x_ratio, y_ratio, gaze_data, radius=5, _x='x_position', _y='y_position'):
    data=gaze_data.to_dict(orient='records')
    # Create a blank image with the specified dimensions
    heatmap_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(heatmap_image)
    
    # Generate the intensity map from fixation data
    intensity_map = np.zeros((height, width))
    frames = []
    frames_captioned = []
    for gaze in data:
        x, y = gaze[_x]*x_ratio, gaze[_y]*y_ratio
        x=int(x)
        y=int(y)
        if y>=height-radius or y<0+radius  or x<0+radius  or x>=width-radius:
            continue
        intensity_map[y][x] += gaze['Time (in secs)']
        
    # Apply Gaussian blur to smooth out the heatmap
    intensity_map = gaussian_filter(intensity_map, sigma=radius)
    
    # Normalize intensity map
    intensity_map /= np.max(intensity_map)
    
    # Apply colors according to intensity map
    for y in range(height):
        for x in range(width):
            intensity = min(255, max(0, int(intensity_map[y][x] * 255)))
            draw.point((x, y), fill=(255 - intensity, 0, 0, intensity))
    
    return heatmap_image

def extract_first_element(lst):
    if isinstance(lst, list) and len(lst) > 0:
        return lst[0]
    else:
        return None
        
class GazeHeatMapGenerator:
    def __init__(self, 
                 mimic_eye_path='physionet.org/files/mimic-eye',
                 mimic_cxr_path='physionet.org/files/mimic-cxr/2.0.0',
                 mimic_cxr_vqa_path='physionet.org/files/mimic-ext-mimic-cxr-vqa/1.0.0',
                ):
        self.mimic_eye_path=mimic_eye_path
        self.mimic_cxr_path=mimic_cxr_path
        mimic_eye_images_path=os.path.join(mimic_eye_path,'patient_*/CXR-JPG/s*')
        meta_file=os.path.join(mimic_eye_path,'spreadsheets/cxr_meta.csv')
        cxr_split_file=os.path.join(mimic_eye_path,'spreadsheets/CXR-JPG/cxr_split.csv')
        cxr_reports_file=os.path.join(mimic_cxr_path,'mimic-cxr-sections/mimic_cxr_sectioned.csv')
        self.meta_df = pd.read_csv(meta_file)
        remove_list=[k for k,v in Counter(meta_df['subject_id'].tolist()).items() if v>1]
        self.set_index('subject_id', inplace=True)
        self.remove_list=[str(i) for i in remove_list]
        self.meta_df.drop(index=remove_list, inplace=True)
        self.patient_subjects = glob('files/mimic-eye/patient_*/CXR-JPG/s*')
        self.cxr_reports = pd.read_csv(cxr_reports_file)
        self.cxr_reports.fillna('', inplace=True)
        self.cxr_reports.set_index('study', inplace=True)
        self.cxr_split = pd.read_csv(cxr_split_file, index_col=1)
        
        with open(os.path.join(mimic_cxr_vqa_path,'train.json'), 'r') as file:
            train = json.load(file)
        with open(os.path.join(mimic_cxr_vqa_path,'valid.json'), 'r') as file:
            valid = json.load(file)
        with open(os.path.join(mimic_cxr_vqa_path,'test.json'), 'r') as file:
            test = json.load(file)
        data=train+valid+test
        vqadf=pd.DataFrame(data)
        vqadf=vqadf[['image_path','answer','question']]
        vqadf=vqadf[vqadf['answer'].apply(len) > 0]
        self.vqadf=vqadf

    def process_patient(self, bp):
        patient_id=bp.split('/patient_')[-1].split('/')[0]      
        if patient_id in self.remove_list:
            return None
        study_id=bp.split('/')[-1]
        dicom_id=self.meta_df.loc[int(patient_id)]['dicom_id']
        split=self.cxr_split.loc[dicom_id]['split']
        EG=self.meta_df.loc[int(patient_id)]['in_eye_gaze']
        REFLACX=self.meta_df.loc[int(patient_id)]['in_reflacx']
        
        try:
            findings=self.cxr_reports.loc[study_id]['findings']
            findings = re.sub("\s+", " ", findings)
        except:
            findings = ""
        try:
            impression=self.cxr_reports.loc[study_id]['impression']
            impression = re.sub("\s+", " ", impression)
        except:
            impression=""
            
        image_path=os.path.join(self.mimic_eye_path,f'/patient_{patient_id}/CXR-JPG/{study_id}/{dicom_id}.jpg')
        image = Image.open(image_path)
        width, height=image.size
        if width>height:
            newheight=int(float(height)/float(width)*512.0)
            image512 = image.resize((512, newheight))
        else:
            newwidth=int(float(width)/float(height)*512.0)
            image512 = image.resize((newwidth, 512))
        resized_image_path=image_path.replace('.jpg','_512.png')
        image512.save(resized_image_path)
        width512, height512=image512.size
        image.close()
        
        if EG:
            gaze_data=pd.read_csv(os.path.join(self.mimic_eye_path,f'/patient_{patient_id}/EyeGaze/fixations.csv'))
            initial_value=gaze_data['Time (in secs)'].iloc[0]
            gaze_data['timestamp_start_fixation']=gaze_data['Time (in secs)'].shift().fillna(0.0)
            gaze_data['timestamp_end_fixation']=gaze_data['Time (in secs)']
            gaze_data['Time (in secs)']=gaze_data['Time (in secs)'].diff().fillna(initial_value)
            gaze_data=gaze_data[(gaze_data['X_ORIGINAL']>0)&(gaze_data['Y_ORIGINAL']>0)&(gaze_data['X_ORIGINAL']<width)&(gaze_data['Y_ORIGINAL']<height)]
            gaze_data=gaze_data[['Time (in secs)','X_ORIGINAL', 'Y_ORIGINAL','transcript']]
            full_heatmap_image=generate_heatmap(image512.convert('RGBA'), width512, height512, width512/width, height512/height, gaze_data, _x='X_ORIGINAL', _y='Y_ORIGINAL')

            m=pd.read_csv(os.path.join(self.mimic_eye_path,f'/patient_{patient_id}/EyeGaze/master_sheet.csv'))
            m=pd.read_csv(bp.replace('bounding_boxes', 'master_sheet')).loc[0]
            temp_dict=m[['gender', 'anchor_age', 'cxr_exam_indication']].fillna('').to_dict()
            ddx="Here is the list of possible diseases for the given chest X-ray:\n"
        
            for k,v in m[['dx1', 'dx2', 'dx3', 'dx4', 'dx5', 'dx6', 'dx7', 'dx8', 'dx9', ]].dropna().to_dict().items():
                ddx+=f'{k[-1]}. {v}.\n'
    
        else:
            for i in os.listdir(os.path.join(self.mimic_eye_path,f'patient_{patient_id}/REFLACX/main_data/')):
                try:
                    gaze_data=pd.read_csv(os.path.join(self.mimic_eye_path,f'patient_{patient_id}/REFLACX/main_data/{i}/fixations.csv'))
                except:
                    gaze_data=None
                if gaze_data is not None:
                    break
            gaze_data=gaze_data[(gaze_data['x_position']>0)&(gaze_data['y_position']>0)&(gaze_data['x_position']<width)&(gaze_data['y_position']<height)]
            gaze_data['Time (in secs)']=gaze_data['timestamp_end_fixation']-gaze_data['timestamp_start_fixation']
            gaze_data=gaze_data[['Time (in secs)','x_position', 'y_position','transcript']]
            full_heatmap_image=generate_heatmap(image512.convert('RGBA'), width512, height512, width512/width, height512/height, gaze_data, _x='x_position', _y='y_position')
            ddx=""
            
        full_result_image = Image.alpha_composite(image512.convert('RGBA'), full_heatmap_image)
        full_heatmap_image_path=image_path.replace('.jpg','_heatmap.png')
        full_result_image.save(full_heatmap_image_path)
        image512.close()
        full_result_image.close()
    
        temp_dict={}
        temp_dict['image_id']=resized_image_path.replace(self.mimic_eye_path, '')
        temp_dict['heatmap_image_id']=full_heatmap_image_path.replace(self.mimic_eye_path, '')
        temp_dict['findings']=findings
        temp_dict['impression']=impression
        temp_dict['differential_diagnosis']=ddx
        temp_dict['split']=split
        temp_dict['source']='EG' if EG else 'REFLACX'
        return temp_dict

    def process_all(self):
        data_dict=[]
        train_data_dict=[]
        val_test_data_dict=[]
        
        for bp in tqdm(self.patient_subjects, total=len(self.patient_subjects)):
            temp_dict=self.process_patient(bp)
            if temp_dict is None:
                continue
            data_dict.append(temp_dict)
            if temp_dict['source']=='REFLACX':
                train_data_dict.append(temp_dict)
            else:
                val_test_data_dict.append(temp_dict)
        
        train_df=pd.DataFrame(train_data_dict)
        train_df.set_index('image_id', inplace=True)
        vqadf=self.vqadf.copy()
        vqadf.set_index('image_path', inplace=True)
        merged_df=pd.merge(train_df, vqa_df, left_index=True, right_index=True)[['answer','question']]
        merged_df['answer'] = merged_df['answer'].apply(extract_first_element)
        train_data_dict=merged_df.reset_index().to_dict(orient='records')
        with open('mimic-eye-heat-train.json', 'w') as fi:
            fi.write(json.dumps(train_data_dict))
        
        test_df=pd.DataFrame(val_test_data_dict)
        test_df.set_index('image_id', inplace=True)
        vqadf = vqadf[vqadf['answer'].apply(lambda x: x == ['yes'] or x == ['no'])]
        vqadf.drop_duplicates(subset='image_path', inplace=True)
        vqadf.set_index('image_path', inplace=True)
        merged_df=pd.merge(test_df, vqa_df, left_index=True, right_index=True)[['answer','question']]
        merged_df['answer'] = merged_df['answer'].apply(extract_first_element)
        val_test_data_dict=merged_df.reset_index().to_dict(orient='records')

        with open('mimic-eye-heat-test.json', 'w') as fi:
            fi.write(json.dumps(val_test_data_dict))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-eye-path", type=str, default='physionet.org/files/mimic-eye')
    parser.add_argument("--mimic-cxr-path", type=str, default='physionet.org/files/mimic-cxr/2.0.0')
    args = parser.parse_args()

    gvg=GazeVideoGenerator(mimic_eye_path=args.mimic_eye_path,
                           mimic_cxr_path=args.mimic_cxr_path
                          )
    gvg.process_all()
