import pandas as pd
from datetime import datetime
import re
import numpy as np
from typing import Tuple
import os
from llm import es_keyword_extractor, nli_keyword_extractor
from postprocessing import convert_lesion, convert_pirads

def extract_cols(
        path: str
    ) -> list:
    ''' 
        desc
        args
        return
    '''
    columns = []
    with open(path, 'r') as file:
        lines = file.readlines()
        columns_line = None
        for line in lines:
            if line.startswith('# columns = '):
                columns_line = line
                break
    # extract the column names using regex
    if columns_line:
        match = re.search(r'# columns = (.+)', columns_line)
        if match:
            columns_str = match.group(1)
            columns = [column.strip() for column in columns_str.split(',')]
    if not columns:
        print('Could not extract columns!')
        raise Exception('Could not extract columns!')
    return columns
    
def extract_aregion_and_coordinates(
        df_s_ll: pd.DataFrame
    ) -> pd.DataFrame:
    ''' 
        desc
            Extract prostate anatomical zone, target coordinates from fcsv file
            and apply static mapping to map A_Region abbreviations to full
            anatomical descriptions A_Description
        args
        return
    '''
    df_f_s = pd.DataFrame()
    #TODO: Compare keys in dict again to all label occurrences in fcsvs, make sure that list is complete
    ar_mapping = {
        'Right': 'RIGHT',
        'RPZplMid': 'RIGHT PERIPHERAL ZONE POSTERIOR LATERAL MID',
        'LPZplMid': 'LEFT PERIPHERAL ZONE POSTERIOR LATERAL MID',
        'LPZplBase': 'LEFT PERIPHERAL ZONE POSTERIOR LATERAL BASE',
        'Left': 'LEFT',
        'LTZaMid': 'LEFT TRANSITION ZONE ANTERIOR MID',
        'LPZpmMid': 'LEFT PERIPHERAL ZONE POSTERIOR MEDIAL MID',
        'LPZpmApex': 'LEFT PERIPHERAL ZONE POSTERIOR MEDIAL APEX',
        'LPZplApex': 'LEFT PERIPHERAL ZONE POSTERIOR LATERAL APEX',
        'RPZpmApex': 'RIGHT PERIPHERAL ZONE POSTERIOR MEDIAL APEX',
        'RTZaMid': 'RIGHT TRANSITION ZONE ANTERIOR MID',
        'RPZplApex': 'RIGHT PERIPHERAL ZONE POSTERIOR LATERAL APEX',
        'RTZpMid': 'RIGHT TRANSITION ZONE POSTERIOR MID',
        'RTZaApex': 'RIGHT TRANSITION ZONE ANTERIOR APEX',
        'RPZpmMid': 'RIGHT PERIPHERAL ZONE POSTERIOR MEDIAL MID',
        'RTZaBase': 'RIGHT TRANSITION ZONE ANTERIOR BASE',
        'RPZaMid': 'RIGHT PERIPHERAL ZONE ANTERIOR MID',
        'LPZaApex': 'LEFT PERIPHERAL ZONE ANTERIOR APEX',
        'AFSmid': 'ANTERIOR FIBROMUSCULAR STROMA MID',
        'LPZaMid': 'LEFT PERIPHERAL ZONE ANTERIOR MID',
        'LTZpBase': 'LEFT TRANSITION ZONE POSTERIOR BASE',
        'RPZplBase': 'RIGHT PERIPHERAL ZONE POSTERIOR LATERAL BASE',
        'MidlinePZApex': 'MIDLINE PERIPHERAL ZONE APEX',
        'LPZbase': 'LEFT PERIPHERAL ZONE BASE',
        'MidlinePZBase': 'MIDLINE PERIPHERAL ZONE BASE',
        'RPZpApex': 'RIGHT PERIPHERAL ZONE POSTERIOR APEX',
        'Bladder wall': 'BLADDER WALL',
        'RTZBase': 'RIGHT TRANSITION ZONE BASE',
        'LTZpMid': 'LEFT TRANSITION ZONE POSTERIOR MID',
        'RPzpmMid': 'RIGHT PERIPHERAL ZONE POSTERIOR MEDIAL MID',
        'RPzplApex': 'RIGHT PERIPHERAL ZONE POSTERIOR LATERAL APEX',
        'RPZpmBase': 'RIGHT PERIPHERAL ZONE POSTERIOR MEDIAL BASE',
        'RPzPmBase': 'RIGHT PERIPHERAL ZONE POSTERIOR MEDIAL BASE',
        'Peri urethral': 'PERI URETHRAL',
        'LTZaBase': 'LEFT TRANSITION ZONE ANTERIOR BASE',
        'LPZpmBase': 'LEFT PERIPHERAL ZONE POSTERIOR MEDIAL BASE',
        'LPZlApex': 'LEFT PERIPHERAL ZONE LATERAL APEX',
        'LTZaMidApex': 'LEFT TRANSITION ZONE ANTERIOR MID APEX',
        'LTZaApex': 'LEFT TRANSITION ZONE ANTERIOR APEX',
        '5:AX T2 FRFSE-TumorROI_PZ_2-label': 'PERIPHERAL ZONE', #?
        '5:AX T2 FRFSE-TumorROI_CGTZ_1-label': 'CENTRAL GLAND TRANSITION ZONE', #?
        'RTZpApex': 'RIGHT TRANSITION ZONE POSTERIOR APEX',
        'LeftLN': 'LEFT LATERAL NODULE', #?
        'LCGmid': 'LEFT CENTRAL GLAND MID',
        'RPZmid': 'RIGHT PERIPHERAL ZONE MID',
        '8:AX T2 FRFSE-TumorROI_PZ_1-label': 'PERIPHERAL ZONE', #?
        '8:AX T2 FRFSE-TumorROI_PZ_2-label': 'PERIPHERAL ZONE', #?
        'MidlinePZ': 'MIDLINE PERIPHERAL ZONE',
        'Midline T2': 'MIDLINE TRANSITION ZONE',
        'Nodule': 'NODULE',
        'LPZpBase': 'LEFT PERIPHERAL ZONE POSTERIOR BASE',
        '5:AX T2 FRFSE-TumorROI_PZ_1-label': 'PERIPHERAL ZONE', #?
    }
    for index, row in df_s_ll.iterrows():
        try:
            df_target = pd.read_csv(
                row['Raw_File_Path_x'], 
                comment='#', 
                names=extract_cols(row['Raw_File_Path_x']), 
                on_bad_lines='warn'
            )
            df_target = df_target[['x','y','z','label']]
            df_target = df_target.rename(columns={
                'x': 'x_cs', 
                'y': 'y_cs', 
                'z': 'z_cs', 
                'label': 'A_Region'
            })
            df_target['Case_Number'] = row['Case_Number']
            df_target['MRN'] = row['MRN']
            df_target['Date'] = row['Date']
            df_target['Path_To_Target'] = row['Raw_File_Path_x']
            df_f_s = pd.concat([df_f_s, df_target], axis=0, ignore_index=True)
        except:
            print(f"No target or error in columns of fcsv, file_path: {row['Raw_File_Path_x']}")

    #### MANUAL INTERVENTION ##############
    # df_f_s = df_f_s[df_f_s['Path_To_Target'].str.contains('mpReviewPreprocessed')] 
    df_f_s = df_f_s[df_f_s['A_Region'] != 'Right'] #not uniquely matchable with diagnosis sectino of path report -> not always!
    df_f_s = df_f_s[df_f_s['A_Region'] != 'Left']  #not uniquely matchable with diagnosis sectino of path report -> not always!
    # df_f_s = df_f_s[df_f_s['A_Region'] != 'MarkupsFiducial_1-1'] #human error
    # df_f_s = df_f_s[df_f_s['A_Region'] != 'F-1'] #human error
    # df_f_s = df_f_s[df_f_s['A_Region'] != 'F-2'] #human error
    # df_f_s = df_f_s[df_f_s['A_Region'] != 'F-3'] #human error
    # df_f_s = df_f_s[df_f_s['A_Region'] != 'inputMarkupNode-3'] #human error
    # df_f_s = df_f_s[df_f_s['A_Region'] != 'inputMarkupNode-4'] #human error
    # df_f_s = df_f_s[df_f_s['A_Region'] != 'inputMarkupNode-5'] #human error
    # df_f_s = df_f_s[df_f_s['A_Region'] != 'inputMarkupNode-6'] #human error
    # df_f_s = df_f_s[df_f_s['A_Region'] != 'inputMarkupNode-7'] #human error
    # df_f_s = df_f_s[~df_f_s['A_Region'].isna()]
    #######################################

    df_f_s = df_f_s.drop_duplicates(subset=[
                        'Case_Number', 
                        'MRN', 
                        'Date', 
                        'x_cs', 
                        'y_cs', 
                        'z_cs', 
                        'A_Region']
    )
    key_exists_mask = [elem in ar_mapping.keys() for elem in df_f_s['A_Region']]
    df_f_s = df_f_s[key_exists_mask]
    df_f_s['A_Description'] = df_f_s['A_Region'].apply(lambda row: ar_mapping[row]) # abbreviations to desriptions
    return df_f_s

def t2w_hbv_adc_extraction(
        df_ppip: pd.DataFrame,
        mods: list,
        image_dir_path: str,
        d_metric: str,
        models: list,
        label_augmentation: list,
        thresholds: list
    ) -> pd.DataFrame:
    '''
        desc
            for every matched preprocedural study from 
            df_ppip, extract the three necessary mri
            modalities, t2w, dwi (hbv) and adc
        args
            - df_ppip: patient-level matched dataframe
        returns 
            - df_images: dataframe with abs paths to all three mri modalities for every study
    '''
    df_images = pd.DataFrame()
    matched_mods = {}
    df_ppip = df_ppip.astype(str)
    df_ppip['PatientID'] = df_ppip['PatientID'].str.zfill(8)
    df_ppip['ANON_AccNumber'] = df_ppip['ANON_AccNumber'].str.zfill(10)
    for patient_id in os.listdir(image_dir_path):
        patient_path = os.path.join(image_dir_path, patient_id)
        if not os.path.isdir(patient_path):
            continue
        if not any(df_ppip["PatientID"].str.contains(patient_id)):
            continue 
        for acc_number in os.listdir(patient_path):
            acc_number_path = os.path.join(patient_path, acc_number)
            if not os.path.isdir(acc_number_path):
                continue 
            if not any(df_ppip["ANON_AccNumber"].str.contains(acc_number)):
                continue
            mr_path = os.path.join(acc_number_path, 'MR')
            if not os.path.isdir(mr_path):
                continue
            series_descrs = os.listdir(mr_path)
            matched_mods = es_keyword_extractor(
                                series_descrs,
                                mods,
                                models[0],
                                thresholds[0],
                                d_metric,
                                label_augmentation[0]
                            )
            for modality,score_list in matched_mods.items():
                if score_list[0] is not None:
                    matched_series_descr = score_list[0]
                    score_matched_series = score_list[1]
                    best_match_path = os.path.join(mr_path, matched_series_descr)
                    mrn = df_ppip.loc[df_ppip['ANON_AccNumber'] == acc_number, 'MRN'].values[0]
                    acc_number_ip = df_ppip.loc[df_ppip['ANON_AccNumber'] == acc_number,'Accession_Number_ip'].values[0] 
                    acc_number_pp_identified = df_ppip.loc[df_ppip['ANON_AccNumber'] == acc_number,'Accession_Number_pp'].values[0]
                    acc_date_ip = df_ppip.loc[df_ppip['ANON_AccNumber'] == acc_number,'Accession_Date_ip'].values[0]
                    acc_date_pp = df_ppip.loc[df_ppip['ANON_AccNumber'] == acc_number,'Accession_Date_pp'].values[0]
                    new_row = pd.DataFrame(
                        [[
                            mrn,
                            patient_id, 
                            acc_number,
                            acc_number_ip,
                            acc_number_pp_identified,
                            acc_date_ip,
                            acc_date_pp,
                            mr_path,
                            modality, 
                            matched_series_descr, 
                            best_match_path
                        ]], 
                        columns=[
                            'MRN',
                            'PatientID', 
                            'ANON_AccNumber', 
                            'Accession_Number_ip',
                            'Accession_Number_pp',
                            'Accession_Date_ip',
                            'Accession_Date_pp',
                            'Studypath_pp',
                            'Modality', 
                            'Seriesdescription', 
                            'Seriespath_pp'
                        ]
                    )
                    df_images = pd.concat([df_images, new_row], ignore_index=True)
                    print(f"Match found (PatientID: {patient_id}, StudyID: {acc_number}) with modality: {modality} --> Series: {matched_series_descr}, conf: {score_matched_series}")
                else:
                    print(f"No match found for (PatientID: {patient_id}, StudyID: {acc_number}) with modality: {modality} --> Series: {matched_series_descr}, conf: {score_matched_series}")     
    return df_images

def extract_gleason_score(
        d_section: str
    ) -> str:
    '''
        desc
        args
        returns 
    '''
    match1 = re.search(r'(\D|^)(\d+\s*\+\s*\d+\s*=\s*\d+)(?=\D|$)', d_section)
    match2 = re.search(r'(?i)benign', d_section)
    if match1:
        gs = match1.group()
    elif match2:
        gs = 'benign'
    else:
        gs = 'no_match'
    return gs

def extract_grade_group(
        d_section: str
    ) -> str:
    '''
        desc
        args
        returns 
    '''
    match1 = re.search(r'(?i)Grade\s*Group\s*\d', d_section)
    match2 = re.search(r'(?i)benign', d_section)
    if match1:
        ggg = match1.group()
    elif match2:
        ggg = 'benign'
    else:
        ggg = 'no_match'
    return ggg

def ip_diagnostics_extraction(
        df_annotations: pd.DataFrame
    ) -> pd.DataFrame:
    '''
        desc
        args
        returns 
    '''
    df_annotations['lesion_GS'] = df_annotations['D_Section'].apply(lambda d_section: extract_gleason_score(d_section))
    df_annotations['lesion_ISUP'] = df_annotations['D_Section'].apply(lambda d_section: extract_grade_group(d_section))
    return df_annotations

def extract_pirads(
        single_diag: str,
        models: list,
        candidates_pirads: list,
        thresholds: list
    ) -> str:

    ''' 
        desc
        args
        return
    '''
    # turn single_diag into list to ensure correct processing in extractor (expects list)
    single_diag = [single_diag]
    matched_pirads = nli_keyword_extractor(
                        single_diag,
                        candidates_pirads, 
                        models[1],
                        thresholds[1]
    )
    pirads = list(matched_pirads.keys())[0]
    return pirads

def create_lsize_candidates(
        single_diag: str,
    ) -> list:
    ''' 
    desc
        construct hypotheses (labels) dynamically based on basic nlp techniques 
        to reduce size of hypothesis space! Approach: 
            1) Extract all numbers in single_diagnosis string 
            2) Check if "mm" or "cm" sequence is in string
            3) Check all numbers with mm/cm as hypotheses against premise 
            -> Assumption: Adding the "mm/cm" gives enough semantic to extract 1.9cm instead of PI-RADS 3
    args
    return
    '''
    hypotheses_mm = []
    hypotheses_cm = []
    numbers = re.findall(r'\d+(?:\.\d+)?', single_diag)
    hypotheses_mm = [num + "mm" for num in numbers]
    hypotheses_cm = [num + "cm" for num in numbers]
    hypotheses_all = hypotheses_cm + hypotheses_mm
    return hypotheses_all

def extract_lesion_size(
        single_diag: str,
        models: list,
        thresholds: list
    ) -> str:
    ''' 
        desc
        args
        return
    '''
    lesion_size = None
    # dynamically create lesion sizes to reduce potential candidates and incr efficiency
    candidate_lsize_all = create_lsize_candidates(single_diag)
    matched_lesion_sizes = nli_keyword_extractor(
                        [single_diag], # convert str to list for nli extractor
                        candidate_lsize_all, 
                        models[1],
                        thresholds[1]
    )
    lesion_size = list(matched_lesion_sizes.keys())[0]
    return lesion_size

def pp_diagnostics_extraction(
        df_annotations: pd.DataFrame,
        models: list,
        candidates_pirads: list,
        thresholds: list
    ) -> pd.DataFrame:
    '''
        desc
        args
        returns 
    '''
    df_annotations['PI_RADS'] = df_annotations['Single_Diagnosis'].apply(
        lambda single_diag: extract_pirads(
                                single_diag, 
                                models, 
                                candidates_pirads, 
                                thresholds
                            )
    )
    df_annotations['Lesion_Size'] = df_annotations['Single_Diagnosis'].apply(
        lambda single_diag: extract_lesion_size(
                                single_diag, 
                                models, 
                                thresholds
                            )
    )
    # postprocessing
    df_annotations['PI_RADS'] = df_annotations['PI_RADS'].apply(lambda pirads: convert_pirads(pirads))
    df_annotations['Lesion_Size'] = df_annotations['Lesion_Size'].apply(lambda l_size: convert_lesion(l_size))
    return df_annotations
    

    def rad_entity_linkage(
            row: None,
            df_rr_impr: None,
            classifier: None,
        ) -> str:

        ''' Desc 
                Extract and match PI-RADS, Lesion Size and anatomical region from radiology impression sections with histopathology and biopsy target coordinates!
            Args:
                - row:
                - df_rr_impr: Dataframe of all single impression sections from rad reports
                - classifier: nli model
            Return:
                None
        '''
        best_match = None
        best_score = -1
        threshold = 0.5 
        best_sequence = None

        hypothesis = row['A_Description']

        #Select fitting row from df_rr_f2 based on MRN, Procedure_Date, Anatomical Region
        if df_rr_impr['Accession_Date_ip'].dtype.name != 'datetime64[ns]':
            df_rr_impr['Accession_Date_ip'] = df_rr_impr['Accession_Date_ip'].apply(lambda val: datetime.strptime(val, "%Y-%m-%d"))
        df_rr_candidates = df_rr_impr[
            (df_rr_impr['MRN'] == row['MRN']) &
            (df_rr_impr['Accession_Date_ip'] == row['Procedure_Date'])
            ]
        premises = list(df_rr_candidates['Single_Diagnosis'])
        #Inverse normal problem and find best matching "premise" for fixed label
        for premise in premises:
            result = classifier(premise, hypothesis, multi_label=False)
            score = result['scores'][0] #best score and sequence are always first
            sequence = result['sequence']
            if (score > best_score) and (score > threshold):
                best_score = score
                best_sequence = sequence
                print(f'Results:{result}\n')
        single_diag = best_sequence
        print(f'Final premise: {single_diag}\nhypothesis: {hypothesis}\n\n')
        return single_diag
