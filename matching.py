import pandas as pd
from datetime import datetime
from extraction import extract_aregion_and_coordinates
from preprocessing import filter_out_ips
from llm import es_keyword_extractor, nli_keyword_extractor

def identify_bq_request(
        df_bq: pd.DataFrame,
        df_cv: pd.DataFrame
        
    ) -> pd.DataFrame:
    ''' 
        desc
            merge mapping dataframe form rpdr with dataframe loaded from big query request 
            to map de-identified PatientID, ANON_AccNumber, etc. back to original values
            for matching with other data sources
        args:
            - df_bq: biq query dataframe
            - df_cv: conversion / mapping dataframe
        return:
            - merged dataframe
    '''
    df_bq = df_bq[['PatientID', 'AccessionNumber', 'StudyDate']].drop_duplicates(ignore_index=True) 
    df_bq['StudyDate'] = df_bq['StudyDate'].apply(lambda val: datetime.strptime(val, "%Y-%m-%d"))
    print(f'Number of unique image studies from bq: {df_bq.shape[0]}')
    df_cv = df_cv[['ORIG_StudyDate', 'ANON_StudyDate', 'ORIG_MRN', 'ANON_MRN', 'ORIG_AccNumber', 'ANON_AccNumber']].drop_duplicates(ignore_index=True)
    df_cv['ORIG_StudyDate'] = df_cv['ORIG_StudyDate'].apply(lambda val: datetime.strptime(val, "%Y-%m-%d"))
    df_cv['ANON_StudyDate'] = df_cv['ANON_StudyDate'].apply(lambda val: datetime.strptime(val, "%Y-%m-%d"))
    print(f'Max study number for conversion (from rpdr request): {df_cv.shape[0]}')
    df_cvbq = pd.merge(df_cv, df_bq, left_on=['ANON_StudyDate', 'ANON_MRN', 'ANON_AccNumber'], right_on=['StudyDate', 'PatientID', 'AccessionNumber'])
    print(f'Number of mapped image studies: {df_cvbq.shape[0]}')
    return df_cvbq

def patient_ip_pp(
        df_spr: pd.DataFrame,
        df_rcvbq: pd.DataFrame
    ) -> pd.DataFrame:
    ''' 
        desc
            create new dataframe grouped by patients with every row consisting of intraprocedural 
            study with the most recent preprocedural study, matched by comparing the study dates
        args:
            - df_spr: df containing all intraprocedural studies (patholgoy, target coordinates, intraprocedural rad report)
            - df_rcvbq: df containing all preprocedural studies (big query imaging studies, conversion, preprocedural rad report)
        return:
            - merged dataframe
    '''
    df_ppip = pd.DataFrame()
    df_concat= pd.concat(
        [df_spr[['Type', 'MRN', 'Accession_Number', 'Accession_Date']], df_rcvbq], 
        ignore_index=True
    )
    grouped = df_concat.groupby('MRN')
    for name, group in grouped:
        ip_rows = group[group['Type'] == 'ip']
        if not ip_rows.empty:
            for index, ip_row in ip_rows.iterrows():
                pp_rows = group[(group['Type'] == 'pp') & (group['Accession_Date'] <= ip_row['Accession_Date']) & (group['Accession_Number'] != ip_row['Accession_Number'])] #get "all" pp dates before ip date that are not the same (see pp definition above!!)
                if not pp_rows.empty:
                    latest_pp_row = pp_rows.loc[pp_rows['Accession_Date'].idxmax()] #get "nearest" pp date to ip date
                    new_row = pd.DataFrame(
                        [[
                            latest_pp_row['MRN'],
                            latest_pp_row['PatientID'].astype(int),
                            latest_pp_row['ANON_AccNumber'].astype(int),
                            ip_row['Accession_Number'],
                            latest_pp_row['Accession_Number'],
                            ip_row['Accession_Date'],
                            latest_pp_row['Accession_Date']
                        ]],
                        columns=[
                            'MRN',
                            'PatientID',
                            'ANON_AccNumber',
                            'Accession_Number_ip',
                            'Accession_Number_pp',
                            'Accession_Date_ip',
                            'Accession_Date_pp'
                        ],
                    )
                    df_ppip = pd.concat([df_ppip, new_row], ignore_index=True)
                else:
                    pass
        else:
            pass
    print(f'Number of studies with matched ip-pp: {df_ppip.shape[0]}')
    return df_ppip

def patient_level_matching(
        df_bq: pd.DataFrame,
        df_cv: pd.DataFrame,
        df_pp: pd.DataFrame,
        df_rr: pd.DataFrame,
        df_s_pl: pd.DataFrame
        
    ) -> None:
    ''' 
        desc
        args:
        return:
    '''
    # converting object dtypes to specific dtypes
    df_rr = df_rr.astype(str).astype({'MRN': 'int64', 'Accession_Date': 'datetime64[ns]'})
    df_pp = df_pp.astype(str).astype({'MRN': 'int64', 'Procedure_Date': 'datetime64[ns]'})
    df_s_pl = df_s_pl.astype(str).astype({'MRN': 'int64', 'Date': 'datetime64[ns]'})
    # preparing preprocedural study candidates (pp) by merging df_bq, df_cv, df_rr
    df_cvbq = identify_bq_request(df_bq, df_cv)
    # TODO: check the reduction of image studies by merging with rad reports
    df_cvbq['ORIG_AccNumber'] = df_cvbq['ORIG_AccNumber'].str.strip()
    df_rcvbq = pd.merge(df_cvbq, 
                        df_rr, 
                        left_on=['ORIG_MRN', 'ORIG_StudyDate'], 
                        right_on=['MRN', 'Accession_Date']
    )
    df_rcvbq = filter_out_ips(df_rcvbq) # discard all candidates with "(?i)biopsy" in title
    df_rcvbq.insert(loc=0, column='Type', value='pp')
    print(f'Number of pp candidates, df_rcvbq: {df_rcvbq.shape[0]}')
    # preparing intraprocedural study candidates (ip) by merging df_pp, df_s_pl, df_rr
    df_sp = pd.merge(df_pp, 
                     df_s_pl, 
                     left_on=['Procedure_Date', 'MRN'], 
                     right_on=['Date', 'MRN']
    ) 
    df_sp = df_sp.drop('Accession_Number', axis=1) # get rid of Accession_Date of path reports as not relevant and naming conflict with rad
    # TODO: Check reduction of study candidates when merging with df_rr
    df_spr = pd.merge(df_rr, 
                      df_sp, 
                      left_on=['Accession_Date', 'MRN'], 
                      right_on=['Procedure_Date', 'MRN']
    ) 
    df_spr.insert(loc=0, column='Type', value='ip')
    print(f'Number of ip candidates, df_spr: {df_spr.shape[0]}')
    # create df_patients with matched pp and ip per patient
    df_ppip = patient_ip_pp(df_spr, df_rcvbq)
    df_ppip = df_ppip.drop_duplicates()
    # in addition drop unconclusive first biopsies in case of repeat biopsies (2 ips 1 pp)
    df_ppip = df_ppip.sort_values(by=['Accession_Date_pp'], ascending=False) # ascending false to get most recent!!
    df_ppip = df_ppip.drop_duplicates(subset=['Accession_Number_pp'], keep='first') # keep only most recent ip (conclusive one!)
    print(f'Final number of matched ip-pp studies, df_ppip: {df_ppip.shape[0]}')
    return df_ppip

def lesion_level_matching(
        df_images: pd.DataFrame,
        df_pp: pd.DataFrame,
        df_rr: pd.DataFrame,
        df_s_ll: pd.DataFrame,
        d_metric: str,
        models: list,
        label_augmentation: list,
        thresholds: list
    ) -> pd.DataFrame:
    ''' 
        desc
        args:
        return:
    '''
    matched_d_sections = {}
    matched_single_diagnoses = {}
    df_annotations = pd.DataFrame()
    # keep only axt2 modality in Seriespath_pp (others are irrelevant for annotations)
    df_images = df_images.drop_duplicates(
        subset=['MRN',
                'PatientID', 
                'ANON_AccNumber', 
                'Accession_Number_ip',
                'Accession_Number_pp',
                'Accession_Date_ip',
                'Accession_Date_pp',
                'Studypath_pp'],
        keep='first'
    ) 
    df_images['Accession_Date_ip'] = df_images['Accession_Date_ip'].astype('datetime64[ns]')
    df_images['Accession_Date_pp'] = df_images['Accession_Date_pp'].astype('datetime64[ns]')
    df_s_ll['Date'] = df_s_ll['Date'].astype('datetime64[ns]')
    df_s_ll['MRN'] = df_s_ll['MRN'].astype(int)
    df_images['MRN'] = df_images['MRN'].astype(int)
    df_pp['MRN'] = df_pp['MRN'].astype(int)
    df_rr['MRN'] = df_rr['MRN'].astype(int)
    # get a_region, a_description from df_s_ll (+ target coordinates for efficiency)
    df_s_ll = extract_aregion_and_coordinates(df_s_ll)
    # ensure that only one target coordinate per A_Region (see logs for reasoning)
    df_s_ll = df_s_ll.drop_duplicates(subset=['MRN', 'Date', 'Case_Number', 'A_Region'])
    # clean df_rr
    df_rr = df_rr.drop_duplicates()
    df_rr['Single_Diagnosis'] = df_rr['Single_Diagnosis'].str.strip()
    df_rr = df_rr[df_rr['Single_Diagnosis'] != '']
    for index, row in df_images.iterrows():
        # get matching a_description and coordinates from df_s_ll
        a_descriptions = df_s_ll.loc[
            (df_s_ll['MRN'] == row['MRN']) &
            (df_s_ll['Date'] == row['Accession_Date_ip']), 
            'A_Description'].values.tolist()
        a_regions = df_s_ll.loc[
            (df_s_ll['MRN'] == row['MRN']) &
            (df_s_ll['Date'] == row['Accession_Date_ip']), 
            'A_Region'].values.tolist()
        case_nums = df_s_ll.loc[
            (df_s_ll['MRN'] == row['MRN']) &
            (df_s_ll['Date'] == row['Accession_Date_ip']), 
            'Case_Number'].values.tolist()
        x_cs = df_s_ll.loc[
            (df_s_ll['MRN'] == row['MRN']) &
            (df_s_ll['Date'] == row['Accession_Date_ip']), 
            'x_cs'].values.tolist()
        y_cs = df_s_ll.loc[
            (df_s_ll['MRN'] == row['MRN']) &
            (df_s_ll['Date'] == row['Accession_Date_ip']), 
            'y_cs'].values.tolist()
        z_cs = df_s_ll.loc[
            (df_s_ll['MRN'] == row['MRN']) &
            (df_s_ll['Date'] == row['Accession_Date_ip']), 
            'z_cs'].values.tolist()
        # if no a_description (e.g. target coordinate was discarded due to unconclusive label, e.g. "Intraop-Target-2"), skip
        if not a_descriptions:
            print('No a_description found - no biopsy coordinate available!')
            continue
        # get matching d_section for a_description if available from df_pp
        d_section_candidates = df_pp.loc[
            (df_pp['MRN'] == row['MRN']) &
            (df_pp['Procedure_Date'] == row['Accession_Date_ip']), 
            'D_Section'].values.tolist()
        matched_d_sections = es_keyword_extractor(
                                d_section_candidates,
                                a_descriptions,
                                models[0],
                                thresholds[0],
                                d_metric,
                                label_augmentation[1]
                             )
        # get matching single_diagnosis for a_description if available from df_rr
        single_diagnoses_candidates = df_rr.loc[
            (df_rr['MRN'] == row['MRN']) &
            (df_rr['Accession_Date'] == row['Accession_Date_pp']),
            'Single_Diagnosis'].values.tolist()
        matched_single_diagnoses = nli_keyword_extractor(
                                    single_diagnoses_candidates, #premises!
                                    a_descriptions, #hypotheses!
                                    models[1],
                                    thresholds[1]
                                )
        # store all as individual rows in new df
        for a_description, a_region, case_num, x_c, y_c, z_c, matched_d_section, matched_single_diagnosis in zip(a_descriptions, a_regions, case_nums, x_cs, y_cs, z_cs, matched_d_sections.items(),matched_single_diagnoses.items()):
            d_section = matched_d_section[1][0]
            single_diagnosis = matched_single_diagnosis[1][0]
            new_row = pd.DataFrame(
                [[
                                row['MRN'],
                                case_num,
                                row['PatientID'],
                                row['ANON_AccNumber'],
                                row['Accession_Number_ip'],
                                row['Accession_Number_pp'],
                                row['Accession_Date_ip'],
                                row['Accession_Date_pp'],
                                row['Studypath_pp'],
                                row['Seriespath_pp'],
                                x_c,
                                y_c,
                                z_c,
                                a_region,
                                a_description,
                                d_section,
                                single_diagnosis
                ]],
                            columns=[
                                'MRN',
                                'Case_Number',
                                'PatientID',
                                'ANON_AccNumber',
                                'Accession_Number_ip',
                                'Accession_Number_pp',
                                'Accession_Date_ip',
                                'Accession_Date_pp',
                                'Studypath_pp',
                                'Seriespath_pp',
                                'X_Coordinate',
                                'Y_Coordinate',
                                'Z_Coordinate',
                                'A_Region',
                                'A_Description',
                                'D_Section',
                                'Single_Diagnosis'
                            ]
            ) 
            df_annotations = pd.concat([df_annotations, new_row], ignore_index=True)
        print(f'Final number of matched ip-pp studies, df_annotations: {df_annotations.shape[0]}')
    return df_annotations