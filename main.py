import yaml
import re
import pandas as pd
from preprocessing import process_txt_report, process_slicer_dir
from matching import patient_level_matching, lesion_level_matching
from extraction import t2w_hbv_adc_extraction, ip_diagnostics_extraction, pp_diagnostics_extraction
from postprocessing import dfs_cleaning, create_ohif_labels
from export import images_export, annotations_export

def generate_dataset(
    configs: dict 
) -> None:
    """
        TODO function to generate dataset
        Args:
            - configs: Dict, read-in parameters from yaml file
    """
    ##################
    # 1. IMPORT DATA #
    ##################

    # path
    rp = open(configs['pat_report_path'], 'r').read()
    pp = re.split(r'\[report_end\]', rp)[1:] # skip first row
    df_pp = pd.DataFrame(pp, columns=['Raw_Text_Input'])

    # rad
    rd = open(configs['rad_report_path'], 'r').read()
    rr = re.split(r'\[report_end\]', rd)[1:] # skip first row
    df_rr = pd.DataFrame(rr, columns=['Raw_Text_Input'])

    # images (big query + conversion table)
    df_bq = pd.read_csv(configs['img_bq_url'])
    df_cv = pd.read_csv(configs['img_bq_conv'])
    print(f'All path cases: {df_pp.shape[0]}')
    print(f'All rad cases: {df_rr.shape[0]}')
    print(f'unfiltered big query image studies: {df_bq.shape[0]}')
    print(f'image studies in de-identification conversion table: {df_cv.shape[0]}')

    #############################################
    # 2. PREPROCESSING & PATIENT-LEVEL MATCHING #
    #############################################

    df_pp = process_txt_report(df_pp, 
                               'path', 
                               configs['minimum_date_considered']
    )
    df_rr = process_txt_report(df_rr, 
                               'rad', 
                               configs['minimum_date_considered']
    )
    df_s_pl, df_s_ll = process_slicer_dir(configs['slicer_ouputs_path'])
    # TODO: create new alternative patient-level matching without rr (if possible) as comparison for data loss
    df_ppip = patient_level_matching(df_bq, 
                                     df_cv, 
                                     df_pp, 
                                     df_rr, 
                                     df_s_pl
    )
    
    ########################################################
    # 3. CLINICAL ENTITY EXTRACION & LESION-LEVEL MATCHING #
    ########################################################

    df_images = t2w_hbv_adc_extraction(df_ppip, 
                                       configs['to_be_extracted_mods'], 
                                       configs['image_dir_path'], 
                                       configs['d_metrics'], 
                                       configs['models'], 
                                       configs['label_augmentation'],
                                       configs['thresholds']
    )
    df_annotations = lesion_level_matching(df_images,
                                           df_pp, 
                                           df_rr, 
                                           df_s_ll, 
                                           configs['d_metrics'], 
                                           configs['models'], 
                                           configs['label_augmentation'],
                                           configs['thresholds']
    )
    df_annotations = pp_diagnostics_extraction(df_annotations, 
                                               configs['models'], 
                                               configs['candidates_pi_rads'],
                                               configs['thresholds']
    )
    df_annotations = ip_diagnostics_extraction(df_annotations)

    ###################################
    # 4. POSTPROCESSING & DATA EXPORT #
    ###################################
    
    # postprocessing
    df_images = pd.read_csv('/Volumes/bwh_prostate_ssd/bwh_dataset_creation/final_baseline_dfs/df_images_man_corr_v1.csv', sep=';')
    df_annotations = pd.read_csv('/Volumes/bwh_prostate_ssd/bwh_dataset_creation/final_baseline_dfs/df_annotations_man_corr_v2.csv', sep=';')
    df_images, df_annotations = dfs_cleaning(df_images, df_annotations)
    df_annotations = create_ohif_labels(df_annotations)

    # safety copy
    df_images.to_csv('/Volumes/bwh_prostate_ssd/bwh_dataset_creation/final_baseline_dfs/df_images_v7.csv', index=False)
    df_annotations.to_csv('/Volumes/bwh_prostate_ssd/bwh_dataset_creation/final_baseline_dfs/df_annotations_v7.csv', index=False)

    # debugging
    # df_annotations = pd.read_csv('/Volumes/bwh_prostate_ssd/bwh_dataset_creation/df_annotations_5_debugging.csv')
    # df_images = pd.read_csv('/Volumes/bwh_prostate_ssd/bwh_dataset_creation/df_images_5_debugging.csv')
    # df_annotations = pd.read_csv('/Volumes/bwh_prostate_ssd/bwh_dataset_creation/df_annotations_v2.csv')
    # df_images = pd.read_csv('/Volumes/bwh_prostate_ssd/bwh_dataset_creation/df_images_v1.csv')

    # export
    images_export(
            df_images, 
            configs['img_export_dtype'][0], # dcm export
            configs['imgs_output_dirs'][0] # ohif export location
    )
    # # images_export(
    # #         df_images, 
    # #         configs['img_export_dtype'][1] # mha export
    # # )
    # # images_export(
    # #         df_images, 
    # #         configs['img_export_dtype'][2] # nifti export
    # # )
    annotations_export(
            df_annotations,
            configs['annotation_export_dtype'][0], # dcm export
            configs['annotations_output_dirs'][0], # ohif point export location
            configs['dcm_sr_types'][0] # point
    )
    annotations_export(
        df_annotations,
        configs['annotation_export_dtype'][0], # dcm export
        configs['annotations_output_dirs'][1], # ohif bbox export location
        configs['dcm_sr_types'][1] # bbox
    )
    # annotations_export(
    #         df_annotations,
    #         configs['annotation_export_dtype'][1] # nifti export
    # )

if __name__ == '__main__':
    # load in YAML configuration
    configs = {}
    base_config_path = './configs.yaml'
    with open(base_config_path, 'r') as file:
        configs.update(yaml.safe_load(file))
    # call dataset generation function
    generate_dataset(configs)

    
    