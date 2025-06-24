import pandas as pd
import re
from datetime import datetime
import os
from pydicom import dcmread
from typing import Tuple
from tqdm import tqdm

def extract_inst(
    raw_text_input: str
) -> str:
    ''' 
        desc: 
            extract name of hospital institution report was created in
        args:
            - raw prostate related text input
        return:
            - institution (either BWH or MGH) as string format
    '''
    first_50 = raw_text_input[:50]
    if 'MGH' in first_50:
        institution = 'MGH'
    elif 'MGH' not in first_50:
        institution = 'BWH'
    return institution

def extract_mrn(
    raw_text_input: str
    ) -> str:  
    ''' 
    desc: 
        extract medical record number from raw text input of every report per row
    args:
        - raw prostate related text input
    return:
        - mrn in string format
    '''
    second_half = raw_text_input.partition('H|')[2]
    mrn = second_half.partition('|')[0]
    return mrn

def extract_an(
    raw_text_input: str
    ) -> str:  
    ''' 
    desc: 
        extract medical accession number (report number) from raw text input of every report per row
    args:
        - raw prostate related text input
    return:
        - accession number in string format
    '''     
    first_50 = raw_text_input[:50]
    tokens = first_50.split("|")
    an = tokens[4]
    return an

def extract_ad(
    raw_text_input: str
    ) -> str:  
    ''' 
    desc: 
        extract accession date (date the report was generated) from raw text input of every report per row
    args:
        - raw prostate related text input
    return:
        - accession date in string format
    '''  
    match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', raw_text_input)
    if match:
        try:
            date = datetime.strptime(match.group(), '%m/%d/%Y').date()
            return date
        except(ValueError):
            date = datetime.strptime(match.group(), '%m/%d/%y').date()
            return date
    else:
        date = "No date!"
        return date

def list_files(
    filepath: str,
    filetype: str      
    ) -> list:  
    ''' 
    desc: 
        - crawl through BiopsyCaseArchive and extract all .fcsv files
        - extract relative paths to biopsy case archive on external SSD (bwh_prostate_ssd) and store .fcsv and .nrrd in two pandas dfs
    args:
        - abs path to directory and target file endings to search for
    return:
        - list of all matching abs paths in diretory
    ''' 
    paths = []
    for root, dirs, files in tqdm(os.walk(filepath), desc="list all fcsv files", position=0):
        for file in files:
            if file.lower().endswith(filetype.lower()):
                paths.append(os.path.join(root, file))
    return(paths)

def list_all(
    root_dir: str     
    ) -> list:  
    ''' 
    desc: 
        - crawl through BiopsyCaseArchive and extract all .fcsv files
        - extract relative paths to biopsy case archive on external SSD (bwh_prostate_ssd) and store .fcsv and .nrrd in two pandas dfs
    args:
        - abs path to directory and target file endings to search for
    return:
        - list of all matching abs paths in diretory
    '''
    file_paths = []
    for root, dirs, files in tqdm(os.walk(root_dir), desc="list all intraop dcm files", position=0):
      # Check if current directory is a "CaseXXX" folder
      if os.path.basename(root).startswith("Case"):
         intraop_directory = os.path.join(root, "DICOM", "Intraop")
         # Check if "Intraop" directory exists
         if os.path.isdir(intraop_directory):
               # Add the first file in the "Intraop" directory to our list of file paths
               intraop_files = os.listdir(intraop_directory)
               if intraop_files:
                  file_paths.append(os.path.join(intraop_directory, intraop_files[0]))
    # print(file_paths)
    return(file_paths)

def extract_case(
    row: pd.Series      
    ) -> str:  
    ''' 
    desc: 
        extract case number out of abs path of every row of pandas dataframe
    args:
        - row of pandas dataframe as pd.Series object
    return:
        - case number as string
    '''
    second_half = row['Raw_File_Path'].partition('/Case')[2]
    case = second_half[:3]
    return case

def extract_case_date(
    row: pd.Series      
    ) -> str:  
    ''' 
    desc: 
        extract case date out of abs path of every row of pandas dataframe
    args:
        - row of pandas dataframe as pd.Series object
    return:
        - case date as string
    '''
    try:
        second_half = row['Raw_File_Path'].partition('/Case')[2]
        first_15 = second_half[:15]
        upper = first_15.partition('-')[2]
        first_8 = upper[:8]
        date = datetime.strptime(first_8, '%Y%m%d').date()
        return date
    except(ValueError):
        return "No date!" #123 cases with no date, as not specified in case directory (DICOM metadata of no use, as it seems study date is not stored, other options?)

def extract_case_mrn(
    row: pd.Series      
    ) -> str:  
    ''' 
    desc: 
        extract mrn out of abs path of every row of pandas dataframe
    args:
        - row of pandas dataframe as pd.Series object
    return:
        - mrn as string
    ''' 
    try:
        file_path = row['Raw_File_Path']
        d_file = dcmread(file_path, force=True)
        try:
            mrn = d_file.PatientID
        except(AttributeError):
            mrn = "AttributeError" #2 Case874 & Case844
    except(IsADirectoryError):
        mrn = "IsADirectoryError" #2 Case154 & Case155
    return mrn

def extract_impression_section(
    row: pd.Series      
    ) -> str:  
    ''' 
    desc: 
        extract impression section from every radiology report with simple keyword detection
    args:
        - row of pandas dataframe as pd.Series object
    return:
        - impression section as string
    ''' 
    diag = row['Raw_Text_Input'].partition('IMPRESSION')[2]  #Seperator1
    diag = diag.partition('CLINICAL')[0] #Seperator2
    words = diag.replace('\n', 'newline')
    # tokens = word_tokenize(diag) #tokenize
    # words = [w.lower() for w in tokens] #convert to lowercase
    # words = [w for w in tokens if not w in stop_words] #removing stopwords
    return words

def extract_rtype(
        raw_text_input: str
        
    ) -> str:
    ''' 
        desc
            utility function for filtering out ip reports from pp candidates in
            df_rcvbq, extracts report description from standardized header.
        args:
        return:
    '''
    second_half = raw_text_input.partition('M|')[2]
    rtype = second_half.partition('|')[0]
    return rtype

def filter_out_ips(
        df_rcvbq: pd.DataFrame
        
    ) -> pd.DataFrame:
    ''' 
        desc
            filter out all ips from rad reports to avoid getting wrong single_diagnosis
            in repeat biopsies (e.g. after incoclusive first biopsies of GS6). Start with
            simple filtering method of header.
            UPDATE: Incllude also "amigo" in filtering, as these reports also refer to ips,
            but do not include biopsy in the title.
        args:
        return:
    '''
    # if "(?i)biopsy" in header discard from df (ip report!)
    df_rcvbq = df_rcvbq[~df_rcvbq['Report_Type'].str.contains('(?i)biopsy')]
    # if "(?i)amigo" in header discard from df (ip report!)
    df_rcvbq = df_rcvbq[~df_rcvbq['Report_Type'].str.contains('(?i)amigo')]
    return df_rcvbq

def process_impression_section(
    df_rr: pd.DataFrame      
    ) -> pd.DataFrame:  
    ''' 
    desc: 
        extract, filter, shorten, split impression section for every radiology report
    args:
        - radiology report as pandas dataframe
    return:
        - processed dataframe with a separate and individual lesion diagnosis section in every row
    '''
    new_row_1 = []
    new_row_2 = []
    single_diags = []
    df_rr_f = pd.DataFrame()
    df_rr_f1 = pd.DataFrame()
    regex_1 = r"(NOTE:newlinePI-RADS \(Prostate Imaging Reporting and Data System\)|newlinePI-RADS 1:)"
    regex_2 = r"(?<!PI-RADS\s)(?<=\b)\d\.\s" #using numbering as easy parititioning of different lesions and ignoring edge case with PI-RADS 4.
    # extract raw impression section
    df_rr.insert(loc=5, column='Impression_section', value=df_rr.apply(lambda row: extract_impression_section(row), axis=1))
    # de-noise and shorten impression section
    for i, row in df_rr.iterrows():
        text = row['Impression_section']
        lesion_candidate = re.split(regex_1, text)[0]
        new_row_1 = pd.DataFrame(
            [[
            row["Institution"], 
            row["MRN"], 
            row["Accession_Number"], 
            row["Accession_Date"], 
            row["Report_Type"],
            lesion_candidate, 
            row["Impression_section"]
            ]],
            columns=["Institution", 
                     "MRN", 
                     "Accession_Number", 
                     "Accession_Date", 
                     "Report_Type",
                     "lesion_candidate", 
                     "Impression_section"
            ]
        )
        df_rr_f = pd.concat([df_rr_f, new_row_1], ignore_index=True)
    # cleaning for semantic anatomical matching
    df_rr_f['lesion_candidate']=df_rr_f['lesion_candidate'].apply(lambda row: row.replace(r'newline', ''))
    df_rr_f['lesion_candidate']=df_rr_f['lesion_candidate'].apply(lambda row: row.replace(r':', ''))
    df_rr_f['lesion_candidate']=df_rr_f['lesion_candidate'].apply(lambda row: row.replace(r'IMPRESSION', ''))
    # separate multiple lesions in one impression section (simple regex looking for "Number. " pattern)
    for i, row in df_rr_f.iterrows():
        text = row['lesion_candidate']
        if re.search(regex_2, text):
            single_diags = re.split(regex_2, text)
            for single_diagnosis in single_diags:
                new_row_2 = pd.DataFrame(
                    [[
                    row["Institution"], 
                    row["MRN"], 
                    row["Accession_Number"], 
                    row["Accession_Date"], 
                    row["Report_Type"],
                    single_diagnosis
                    ]],
                    columns=["Institution", 
                             "MRN", 
                             "Accession_Number", 
                             "Accession_Date", 
                             "Report_Type",
                             "Single_Diagnosis"
                    ]
                )
                df_rr_f1 = pd.concat([df_rr_f1, new_row_2], ignore_index=True)
        else:
            new_row_2 = pd.DataFrame(
                [[
                row["Institution"], 
                row["MRN"], 
                row["Accession_Number"], 
                row["Accession_Date"], 
                row["Report_Type"],
                row["lesion_candidate"]
                ]],
                columns=["Institution", 
                         "MRN", 
                         "Accession_Number", 
                         "Accession_Date", 
                         "Report_Type",
                         "Single_Diagnosis"
                ]
            )
            df_rr_f1 = pd.concat([df_rr_f1, new_row_2], ignore_index=True)
    print(f'Number of preprocessed rad reports with lesion level diagnoses: {df_rr_f1.shape[0]}')
    return df_rr_f1

def extract_diagnosis_section(
    row: pd.Series      
    ) -> str:  
    ''' 
        desc: 
            extract diagnosis section from every pathology report with simple keyword detection
        args:
            - row of pandas dataframe as pd.Series object
        return:
            - diagnosis section as string
    '''
    diag = row['Raw_Text_Input'].partition('DIAGNOSIS')[2]  #Seperator1
    diag = diag.partition('CLINICAL')[0] #Seperator2
    words = diag.replace('\n', 'newline')
    # tokens = word_tokenize(diag) #tokenize
    # words = [w.lower() for w in tokens] #convert to lowercase
    # words = [w for w in tokens if not w in stop_words] #removing stopwords
    return words

def extract_procedure_date(
    row: pd.Series      
    ) -> str:  
    ''' 
        desc: 
            extract procedure date from every pathology report 
            with two-stage regex, first on raw_text_input, if not 
            successful use regex on tokenized diagnosis section
        args:
            - row of pandas dataframe as pd.Series object
        return:
            - procedure date as a string
    '''
    # first stage regex date extraction from raw_text_input
    match_1 = re.search(r'(?i)(Procedure Date:|Date of Operation:|Date Taken:|Date of Procedure:)\s+(\d{1,2}/\d{1,2}/\d{2,4})', row['Raw_Text_Input'])
    match_2 = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', row['Diagnosis_section'])
    if match_1:
        try:
            procedure_date = datetime.strptime(match_1.group(2), '%m/%d/%Y').date() #using search with regex capturing group
        except(ValueError):
            procedure_date = datetime.strptime(match_1.group(2), '%m/%d/%y').date()
    elif match_2:
        try:
            procedure_date = datetime.strptime(match_2.group(), '%m/%d/%Y').date() #using search with regex capturing group
        except(ValueError):
            procedure_date = datetime.strptime(match_2.group(), '%m/%d/%y').date()
    else:
        print('No date could be exracted!')
        procedure_date = 'No date!'
    return procedure_date

def process_diagnosis_section(
    df_pp: pd.DataFrame      
    ) -> pd.DataFrame:  
    ''' 
    desc: 
        extract, filter, shorten, split impression section for every radiology report
    args:
        - pathology report as pandas dataframe
    return:
        - processed dataframe with a separate and individual lesion diagnosis section in every row
    '''
    new_row = []
    df_pp_f = pd.DataFrame()
    df_pp = df_pp[df_pp['Raw_Text_Input'].str.contains('DIAGNOSIS')] # exclude all cases without clear diagnosis section
    df_pp.insert(loc=4, column='Diagnosis_section', value=df_pp.apply(extract_diagnosis_section, axis=1))   
    df_pp.insert(loc=4, column='Procedure_Date', value=df_pp.apply(extract_procedure_date, axis=1))
    df_pp = df_pp[df_pp['Procedure_Date'] != 'No date!'] # exclude reports where no date could be extracted
    # store every section of diagnosis paragraph in separate row (using "newlinenewline" as splitting keyword)
    for i, row in tqdm(df_pp.iterrows(), desc='Writing diagnosis section candidates', position=0):
        text = row['Diagnosis_section']
        elements = re.split('newlinenewline', text)
        for element in elements:
            new_row = pd.DataFrame([[
                row['MRN'], 
                row['Procedure_Date'], 
                row['Accession_Number'],
                element, 
                row['Diagnosis_section']]],
                columns=['MRN', 'Procedure_Date', 'Accession_Number', 'D_Section', 'D_Paragraph'])
            df_pp_f = pd.concat([df_pp_f, new_row], ignore_index=True)
    # cleaning for semantic anatomical matching & baseline histopathology extraction (Gleason Score, Grade Group)
    df_pp_f.drop(df_pp_f[df_pp_f['D_Section'] == ':'].index, inplace=True)
    df_pp_f.drop(df_pp_f[df_pp_f['D_Section'] == 'newline'].index, inplace=True)
    df_pp_f.drop(df_pp_f[df_pp_f['D_Section'] == ''].index, inplace=True)
    print(f'Number of preprocessed path reports with lesion level diagnoses: {df_pp_f.shape[0]}')
    return df_pp_f

def process_txt_report(
    df_report: pd.DataFrame,
    report_type: str,
    minimum_date: str = None
) -> pd.DataFrame:
    ''' 
        desc: 
            do general preprocessing of semi-structured clinical reports (radiology and pathology)
        args:
            - pandas dataframe with single raw report in separate rows
            - specificaiton of report type for additional preprocessing steps for path report
            - minimum date to discard very old cases with different formatting (etc.)
        return:
            - pandas dataframe with basic structured data extracted from only prostate related report
    '''
    df_report = df_report[df_report['Raw_Text_Input'].str.contains('(?i)prostat')]
    df_report.insert(loc=0, column='Institution', value=df_report['Raw_Text_Input'].apply(lambda raw_text_input: extract_inst(raw_text_input)))
    df_report.insert(loc=1, column='MRN', value=df_report['Raw_Text_Input'].apply(lambda raw_text_input: extract_mrn(raw_text_input)))
    df_report.insert(loc=2, column='Accession_Number', value=df_report['Raw_Text_Input'].apply(lambda raw_text_input: extract_an(raw_text_input)))
    # df_report['Accession_Number'] = df_report['Accession_Number'].str.strip()
    df_report = df_report.drop_duplicates(subset=['Accession_Number']) 
    df_report.insert(loc=3, column='Accession_Date', value=df_report['Raw_Text_Input'].apply(lambda raw_text_input: extract_ad(raw_text_input)))
    if minimum_date:
        df_report = df_report[df_report['Accession_Date'] >= datetime.strptime(minimum_date, '%m/%d/%Y').date()] # select all reports on and after 01/01/2005
    # additional separate preprocessing steps for rad and path reports
    if report_type == 'path':
        df_report = process_diagnosis_section(df_report)
        print(f'Prostate cases in {report_type} report: {df_report.shape[0]}')
    if report_type == 'rad':
        df_report.insert(loc=4, column='Report_Type', value=df_report['Raw_Text_Input'].apply(lambda raw_text_input: extract_rtype(raw_text_input)))
        df_report = process_impression_section(df_report)
        print(f'Prostate cases in {report_type} report: {df_report.shape[0]}')
    return df_report

def process_slicer_dir(
    slicer_outputs_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ''' 
        desc: 
            create pandas dataframe for relevant biopsy target coordinates
        args:
            - absolute file path to handcurate BiopsyCaseArchive with 3D slicer outputs
        return:
            - pandas dataframe with mrn, case number, date and raw file path
    '''
    #Target paths (df_fcsv_pl is for patient matching - using drop_duplicates() function for exporting date_cross_ref_s.csv, df_fcsv_ll is for lesion_level matching)
    df_fcsv_pl = pd.DataFrame({'Raw_File_Path': list_files(slicer_outputs_path, '.fcsv')})
    df_fcsv_ll = pd.DataFrame({'Raw_File_Path': list_files(slicer_outputs_path, '.fcsv')})
    df_dcm = pd.DataFrame({'Raw_File_Path': list_all('/Volumes/prostates/ProstateBiopsyCasesArchive/')})
    df_fcsv_pl.insert(loc=0, column='Case_Number', value=df_fcsv_pl.apply(lambda row: extract_case(row), axis=1))
    df_fcsv_pl.insert(loc=1, column='Date', value=df_fcsv_pl.apply(lambda row: extract_case_date(row), axis=1))
    df_fcsv_ll.insert(loc=0, column='Case_Number', value=df_fcsv_ll.apply(lambda row: extract_case(row), axis=1))
    df_fcsv_ll.insert(loc=1, column='Date', value=df_fcsv_ll.apply(lambda row: extract_case_date(row), axis=1))
    df_dcm.insert(loc=0, column='Case_Number', value=df_dcm.apply(lambda row: extract_case(row), axis=1))
    df_dcm.insert(loc=1, column='Date', value=df_dcm.apply(lambda row: extract_case_date(row), axis=1))
    df_dcm.insert(loc=2, column='MRN', value=df_dcm.apply(lambda row: extract_case_mrn(row), axis=1))
    # keep only preprocedural targets (otherwise intraop targets visualized for preop scan!)
    # TODO: Calculate numbers when also considering intraop images and targets
    df_fcsv_ll = df_fcsv_ll[df_fcsv_ll['Raw_File_Path'].str.contains('(?i)pre')]
    df_fcsv_pl = df_fcsv_pl[df_fcsv_pl['Raw_File_Path'].str.contains('(?i)pre')]
    # remove all columns with no date extracted
    df_fcsv_ll = df_fcsv_ll[df_fcsv_ll['Date'] != 'No date!']
    df_fcsv_pl = df_fcsv_pl[df_fcsv_pl['Date'] != 'No date!']
    df_dcm = df_dcm[df_dcm['Date'] != 'No date!']
    df_dcm = df_dcm.drop_duplicates(subset=['Case_Number'], ignore_index=True) #795 distinct case numbers with date in naming
    df_fcsv_pl = df_fcsv_pl.drop_duplicates(subset=['Case_Number'], ignore_index=True) #795 distinct case numbers with date in naming
    
    # TODO: try pipeline without manual correction section to see if it makese a difference (should not -> complete automatic should also be feasible)
    ############################ MANUAL CORRECTION SECTION ############################
    # Manually correcting and adding MRN and dates 
    # mrn: 9 edge cases (viewing PatientID in slicer) 
    # date: 123 "No date!" cases, maybe non-manual solution (see extraction above)
    # attributeError & IsADirectoryError
    df_dcm.loc[df_dcm['Case_Number']=='154', 'MRN'] = '***'
    df_dcm.loc[df_dcm['Case_Number']=='155', 'MRN'] = '***'
    df_dcm.loc[df_dcm['Case_Number']=='874', 'MRN'] = '***'
    df_dcm.loc[df_dcm['Case_Number']=='844', 'MRN'] = '***'
    # other
    df_dcm = df_dcm[df_dcm['Case_Number']!='734'] #Daily QA AMIGO -> delete
    df_dcm = df_dcm[df_dcm['Case_Number']!='441'] #TestSubject -> delete
    df_dcm = df_dcm[df_dcm['Case_Number']!='521'] #manual error (erroneous patientid)
    df_dcm = df_dcm[df_dcm['Case_Number']!='134'] #manual error (erroneous patientid)
    df_dcm = df_dcm[df_dcm['Case_Number']!='021'] #manual error (erroneous patientid)
    ############################ MANUAL CORRECTION SECTION ############################

    #Merge df_fcsv_pl / df_fcsv_ll / df_nrrd and df_dcm
    df_dcmfcsv_pl = pd.merge(df_fcsv_pl, df_dcm, on=['Case_Number', 'Date'])
    df_dcmfcsv_pl = df_dcmfcsv_pl.iloc[:,[0,1,3,2,4]]
    print(df_dcmfcsv_pl.shape[0])

    df_dcmfcsv_ll = pd.merge(df_fcsv_ll, df_dcm, on=['Case_Number', 'Date'])
    df_dcmfcsv_ll = df_dcmfcsv_ll.iloc[:,[0,1,3,2,4]]
    print(df_dcmfcsv_ll.shape[0])

    return df_dcmfcsv_pl, df_dcmfcsv_ll
    