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

def process_dict_vqa(di, idx, dataset='reflacx'):
    return {'id': f'{dataset}_vqa_'+str(idx), 'image': di['heatmap_image_id'].strip(), 
            'conversations': [
                {'from': 'human', 'value': '<image>\n'+PROMPT['HEATMAP_DESC']+PROMPT['DEFAULT_VQA']+f"{di['question']}\nY. yes.\nN. no.\n"}, 
                {'from': 'gpt', 'value': di['answer'].strip()}
            ]}                

def process_dict_err(di, idx, dataset='reflacx'):
    return {'id': f'{dataset}_err_'+str(idx), 'image': di['heatmap_image_id'].strip(),
            'conversations': [
                {'from': 'human', 'value': '<image>\n'+PROMPT['HEATMAP_DESC']+PROMPT['DEFAULT_ERR']+f"{di['findings_mod']}\nY. mistakes or errors in findings.\nN. no mistakes or no errors in findings.\n"},
                {'from': 'gpt', 'value': choices_dict[di['label'].strip()]}
            ]}

def process_dict_ddx(di, idx, dataset='reflacx'):
    return {'id': f'{dataset}_ddx_'+str(idx), 'image': di['heatmap_image_id'].strip(), 
            'conversations': [
                {'from': 'human', 'value': '<image>\n'+PROMPT['HEATMAP_DESC']+PROMPT['DEFAULT_DDX']},
                {'from': 'gpt', 'value': di['differential_diagnosis'].strip()}
            ]}

def process_dict_gen(di, idx, dataset='reflacx'):
    return {'id': f'{dataset}_gen_'+str(idx), 'image': di['heatmap_image_id'].strip(), 
            'conversations': [
                {'from': 'human', 'value': '<image>\n'+PROMPT['HEATMAP_DESC']+PROMPT['DEFAULT_GEN']},
                {'from': 'gpt', 'value': di['findings_org'].strip()}
            ]}
    
def process_dict_sum(di, idx, dataset='reflacx'):
    return {'id': f'{dataset}_sum_'+str(idx), 'image': di['heatmap_image_id'].strip(),
            'conversations': [
                {'from': 'human', 'value': '<image>\n'+PROMPT['HEATMAP_DESC']+PROMPT['DEFAULT_SUM']+f"{di['findings_org']}\nImpression: "},
                {'from': 'gpt', 'value': di['impression'].strip()}
            ]}




def process_train():
    with open('mimic-eye-heat-train.json') as f:
        d=json.loads(f.read())
    vqa_data_dict=[process_dict_vqa(di, idx) for idx, di in enumerate(d)]
    ddx_data_dict=[process_dict_ddx(di, idx) for idx, di in enumerate(d)]
    #err_data_dict=[process_dict_err(di, idx) for idx, di in enumerate(d)]
    gen_data_dict=[process_dict_gen(di, idx) for idx, di in enumerate(d)]
    sum_data_dict=[process_dict_sum(di, idx) for idx, di in enumerate(d)]
    #miccai_data_dict=err_data_dict+vqa_data_dict+ddx_data_dict+gen_data_dict+sum_data_dict
    miccai_data_dict=vqa_data_dict+ddx_data_dict+gen_data_dict+sum_data_dict
    shuffle(miccai_data_dict)
    with open('instruction_miccai_heatmap.json', 'w') as fi:
        fi.write(json.dumps(miccai_data_dict))

if __name__ == '__main__':
    process_train()
