import json
import pandas as pd
from tqdm import tqdm
import os
from random import shuffle

choices_dict={"remove": "Y", "insert": "Y", "replace": "Y", "original": "N"}

PROMPT = {
'DEFAULT_DDX':("What are the possible differential diagnoses for this patient?"),
'DEFAULT_ERR':("Please choose the most suitable one between Y and N as the answer to this question. "
               "Does this findings report about the given chest x-ray, contain any mistakes or errors?\n\n"
              ),
'DEFAULT_GEN':("Write a findings report on the given chest x-ray, including information about any abnormalities that you see."),
'DEFAULT_SUM':("Write an impression summarization of the given chest x-ray and findings report, "
               "including information about any abnormalities that you see.\nFindings: "),
'DEFAULT_VQA':("Answer with the option's letter from the given choices directly. "
              ),
'HEATMAP_DESC':("The chest x-ray image has red dots on top which indicate the eye gaze of the radiologist. "
                "The duration of eye gaze is represented as the darkness of the dots. "
                "Use this eye gaze information to answer the question.\n"),
}

def process_dict_ddx(di, idx, image_id='image_id'):
    if image_id=='image_id':    
        return {'image': di['image_id'].strip(), 'sys': SYSTEM['DEFAULT'], 'text': PROMPT['DEFAULT_DDX'], 
                'question_id': idx, 'differential_diagnosis': di['differential_diagnosis'].strip()}
    else:
        return {'image': di['heatmap_image_id'].strip(), 'sys': SYSTEM['DEFAULT'], 'text': PROMPT['HEATMAP_DESC']+PROMPT['DEFAULT_DDX'], 
                'question_id': idx, 'differential_diagnosis': di['differential_diagnosis'].strip()}

def process_dict_err(di, idx, image_id='image_id'):
    if image_id=='image_id':    
        return {'image': di['image_id'].strip(), 'sys': SYSTEM['DEFAULT'], 
                'text': PROMPT['DEFAULT_ERR']+f"{di['findings_mod']}\nY. mistakes or errors in findings.\nN. no mistakes or no errors in findings.\n", 
                'question_id': idx, 'label': di['label'].strip()}
    else:
        return {'image': di['heatmap_image_id'].strip(), 'sys': SYSTEM['DEFAULT'], 
            'text': 'text': PROMPT['HEATMAP_DESC']+PROMPT['DEFAULT_ERR']+f"{di['findings_mod']}\nY. mistakes or errors in findings.\nN. no mistakes or no errors in findings.\n", 
            'question_id': idx, 'label': di['label'].strip()}

def process_dict_gen(di, idx, image_id='image_id'):
    if image_id=='image_id':    
        return {'image': di['image_id'].strip(), 'sys': SYSTEM['DEFAULT'], 'text': PROMPT['DEFAULT_GEN'], 
                'question_id': idx, 'findings': di['findings_org'].strip()}
    else:
        return {'image': di['heatmap_image_id'].strip(), 'sys': SYSTEM['DEFAULT'], 'text': PROMPT['HEATMAP_DESC']+PROMPT['DEFAULT_GEN'], 
                'question_id': idx, 'findings': di['findings_org'].strip()}

def process_dict_sum(di, idx, image_id='image_id'):
    if image_id=='image_id':    
        return {'image': di['image_id'].strip(), 'sys': SYSTEM['DEFAULT'], 
                'text': PROMPT['DEFAULT_SUM']+f"{di['findings_org']}\nY. mistakes or errors in findings.\nN. no mistakes or no errors in findings.\nImpression: ", 
                'question_id': idx, 'impression': di['impression'].strip()}

    else:
        return {'image': di['heatmap_image_id'].strip(), 'sys': SYSTEM['DEFAULT'], 
                'text': PROMPT['HEATMAP_DESC']+PROMPT['DEFAULT_SUM']+f"{di['findings_org']}\nY. mistakes or errors in findings.\nN. no mistakes or no errors in findings.\nImpression: ", 
                'question_id': idx, 'impression': di['impression'].strip()}

def process_dict_vqa(di, idx, image_id='image_id'):
    if image_id=='image_id':    
        return {'image': di['image_id'].strip(), 'sys': SYSTEM['DEFAULT'], 
                'text': PROMPT['DEFAULT_VQA']+f"{di['question']}\nY. yes.\nN. no.\n", 
                'question_id': idx, 'answer': di['answer'].strip()}
    else:
        return {'image': di['heatmap_image_id'].strip(), 'sys': SYSTEM['DEFAULT'], 
                'text': PROMPT['HEATMAP_DESC']+PROMPT['DEFAULT_VQA']+f"{di['question']}\nY. yes.\nN. no.\n", 
                'question_id': idx, 'answer': di['answer'].strip()}

def process(d, mode='default'):
    if mode=='default':
        image_id='image_id'
    else:
        image_id='heatmap_image_id'

    with open(f'DDX/MICCAI_{mode}.jsonl', 'w') as fi:
        fi.writelines([json.dumps(process_dict_ddx(di, idx, image_id))+"\n" for idx, di in enumerate(d)])
    # with open(f'ERR/MICCAI_{mode}.jsonl', 'w') as fi:
    #     fi.writelines([json.dumps(process_dict_err(di, idx, image_id))+"\n" for idx, di in enumerate(d)])
    with open(f'GEN/MICCAI_{mode}.jsonl', 'w') as fi:
        fi.writelines([json.dumps(process_dict_gen(di, idx, image_id))+"\n" for idx, di in enumerate(d)])
    with open(f'SUM/MICCAI_{mode}.jsonl', 'w') as fi:
        fi.writelines([json.dumps(process_dict_sum(di, idx, image_id))+"\n" for idx, di in enumerate(d)])
    with open(f'VQA/MICCAI_{mode}.jsonl', 'w') as fi:
        fi.writelines([json.dumps(process_dict_vqa(di, idx, image_id))+"\n" for idx, di in enumerate(d)])

if __name__ == '__main__':
    if not os.path.exists('DDX'):
        os.makedirs('DDX')
    if not os.path.exists('GEN'):
        os.makedirs('GEN')
    if not os.path.exists('SUM'):
        os.makedirs('SUM')
    if not os.path.exists('VQA'):
        os.makedirs('VQA')
    if not os.path.exists('ERR'):
        os.makedirs('ERR')
        
    with open('mimic-eye-heat-test.json') as f:
        d=json.loads(f.read())
    process(d, mode='default')
    process(d, mode='heatmap')
