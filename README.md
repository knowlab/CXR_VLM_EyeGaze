# CXR_VLM_EyeGaze

[![License:Physionet](https://img.shields.io/badge/License-Physionet-red.svg)]([https://physionet.org/](https://physionet.org/content/mimic-eye-multimodal-datasets/1.0.0/))


## MIMIC-Eye-Heat

The MIMIC-Eye-Heat dataset captures gaze patterns as heat map. 
The heat map is generated from radiologists' eye-gaze during CXR interpretation.
The eye gaze data is from [MIMIC-Eye](https://physionet.org/content/mimic-eye-multimodal-datasets/1.0.0/).
Currently, error detection data cannot be released due to the original MIMIC-CXR data, 
so the training and evaluation datasets are released without this task.

### Access Requirements

The MIMIC-Eye-Heat dataset is constructed from the MIMIC-Eye (v1.0.0), MIMIC-Ext-MIMIC-CXR-VQA (v1.0.0), MIMIC-CXR-JPG (v2.1.0), and MIMIC-CXR (v2.0.0).  
All these source datasets require a credentialed Physionet license. To access the source datasets, you must fulfill all of the following requirements:  

1. Be a [credentialed user](https://physionet.org/settings/credentialing/)
    - If you do not have a PhysioNet account, register for one [here](https://physionet.org/register/).
    - Follow these [instructions](https://physionet.org/credential-application/) for credentialing on PhysioNet.
    - Complete the "CITI Data or Specimens Only Research" [training course](https://physionet.org/about/citi-course/).
2. Sign the data use agreement (DUA) for each project
    - https://physionet.org/sign-dua/mimic-cxr/2.0.0/
    - https://physionet.org/sign-dua/mimic-cxr-jpg/2.1.0/
    - https://physionet.org/sign-dua/mimic-eye-multimodal-datasets/1.0.0/

## Environment Setup

Python should be 3.10 or higher. Set up the environment and install the required packages using the commands below:  

```
# Set up the environment
conda create -n mimiceyegaze python=3.10 -y

# Activate the environment
conda activate mimiceyegaze

# Install required packages
pip install pandas==1.5.3 tqdm==4.64.1 scipy==1.10.0 pillow==9.4.0
```


## Usage

### Downloading Images and Reports

1. Download with your USERNAME for PhysioNet and unzip the the downloaded zip files
```bash
wget -r -N -c -np --user USERNAME --ask-password https://physionet.org/content/mimic-eye-multimodal-datasets/get-zip/1.0.0/
wget -r -N -c -np --user USERNAME --ask-password https://physionet.org/files/mimic-cxr/2.0.0/mimic-cxr-reports.zip
wget -r -N -c -np --user USERNAME --ask-password https://physionet.org/content/mimic-ext-mimic-cxr-vqa/get-zip/1.0.0/
```

2. Create the sections file of MIMIC-CXR (mimic_cxr_sectioned.csv.gz) with [create_sections_file.py](https://github.com/MIT-LCP/mimic-cxr/tree/master/txt)

3. Extract the files
```bash
gzip -d mimic_cxr_sectioned.csv.gz
```
4. Run the code to generate videos
```
python video_dataset_processing.py
```

5. Run the code to generate prompt
```
python prompt_processing.py
python instruction_tuning_processing.py
```

## License

The code in this repository is provided under the terms of the MIT License. The final output of the dataset created using this code, the MIMIC-Eye-Video, is subject to the terms and conditions of the original dataset from Physionet: [MIMIC-CXR License](https://physionet.org/content/mimic-cxr/view-license/2.0.0/).
