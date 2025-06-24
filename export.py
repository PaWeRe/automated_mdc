import pandas as pd
import numpy as np
import os
import pydicom
import shutil
import re
from picai_prep.examples.dcm2mha.sample_archive import generate_dcm2mha_settings
from picai_prep import Dicom2MHAConverter
import pydicom
from pydicom.uid import generate_uid
from pydicom.filereader import dcmread
from pydicom.sr.codedict import codes
from highdicom.sr.content import (
    FindingSite,
    ImageRegion,
    ImageRegion3D,
    SourceImageForRegion
)
from highdicom.sr.enum import GraphicTypeValues3D
from highdicom.sr.enum import GraphicTypeValues
from highdicom.sr.sop import Comprehensive3DSR, ComprehensiveSR
from highdicom.sr.templates import (
    DeviceObserverIdentifyingAttributes,
    Measurement,
    MeasurementProperties,
    MeasurementReport,
    ObservationContext,
    ObserverContext,
    PersonObserverIdentifyingAttributes,
    PlanarROIMeasurementsAndQualitativeEvaluations,
    RelationshipTypeValues,
    TrackingIdentifier,
)
from highdicom.sr.value_types import (
    CodedConcept,
    CodeContentItem,
)
import logging
logger = logging.getLogger("highdicom.sr.sop")
logger.setLevel(logging.INFO)
import subprocess
from tqdm import tqdm
import shutil
from highdicom.sr.templates import QualitativeEvaluation
import highdicom as hd
from postprocessing import create_bbox_array

def save_single_bbox_dcm_sr_for_case(
      row: pd.Series, 
      reference_dcm_file: str
    ):
    ''' 
        desc
        args
        return
    '''
    # read in reference dcm file and modify PatientSex
    subprocess.call(["dcmodify", "-m", "PatientSex=O", reference_dcm_file]) # use dcmodify to modify the PatientSex attribute
    image_dataset = dcmread(reference_dcm_file)
    # describe the context of reported observations: the person that reported
    # the observations and the device that was used to make the observations
    observer_person_context = ObserverContext(
        observer_type=codes.DCM.Person,
        observer_identifying_attributes=PersonObserverIdentifyingAttributes(
            name='Anonymous^Reader'
        )
    )
    observer_device_context = ObserverContext(
        observer_type=codes.DCM.Device,
        observer_identifying_attributes=DeviceObserverIdentifyingAttributes(
            uid=generate_uid()
        )
    )
    observation_context = ObservationContext(
        observer_person_context=observer_person_context,
    )
    
    ###############################################################
    ########## Target coordinates & Bounding Boxes ################
    ###############################################################
    bbox_slice_list = create_bbox_array(row)
    if bbox_slice_list != 'no_lesion_size':
        imaging_measurements = []
        print("Parsing "+ row['pos_projected'])
        target_name = row['fid']
        pos_items = re.search(r"\ ?(-?[0-9]*\.[0-9]*)\ +(-?[0-9]*\.[0-9]*)\ +(-?[0-9]*\.[0-9]*)", row['pos_projected'])
        if not pos_items:
            print("Failed to parse "+ row['pos_projected'])
        source_image_ref = hd.sr.content.SourceImageForMeasurementGroup.from_source_image(image_dataset)
        # biopsy target point coordinate
        referenced_region_3d = ImageRegion3D(
            graphic_type=GraphicTypeValues3D.POLYGON,
            graphic_data=np.array(create_bbox_array(row)),
            frame_of_reference_uid=row['FrameOfReferenceUID']
        )
        # anatomical region
        if row['A_Description'] == 'TRANSITION ZONE':
            finding_location = CodedConcept(
                            value="399384005",
                            meaning="Transition zone of the prostate",
                            scheme_designator="SCT"
                        )
        elif row['A_Description'] == 'PERIPHERAL ZONE':
            finding_location = CodedConcept(
                            value="279706003",
                            meaning="Peripheral zone of the prostate",
                            scheme_designator="SCT"
                        )
        elif row['A_Description'] == 'STROMA':
            finding_location = CodedConcept(
                            value="717025007",
                            meaning="Anterior fibromuscular stroma of prostate",
                            scheme_designator="SCT"
                        )
        elif row['A_Description'] == 'VESICLE':
            finding_location = CodedConcept(
                            value="64739004",
                            meaning="Seminal vesicle",
                            scheme_designator="SCT"
                        )           
        # finding_sites = [
        #     FindingSite(anatomic_location=finding_location)
        # ]

        ###########################################
        ########## Gleason Grading ################
        ###########################################

        evaluation = []
        ggg_code = None
        #Clinical Significance
            # if int(row['lesion_ISUP'][-1]) >= 2:
            #   evaluation.append(CodeContentItem(CodedConcept(
            #                value="RID49502",
            #                meaning="clinically significant prostate cancer",
            #                scheme_designator="RADLEX"), codes.SCT.Yes, RelationshipTypeValues.CONTAINS))
        if isinstance(row['lesion_ISUP'], str) and (row['lesion_ISUP'].strip() != '') and (row['lesion_ISUP'].strip().lower() != 'nan') and (row['lesion_ISUP'].strip().lower() != 'benign') and (row['lesion_ISUP'].strip().lower() != 'metastatic pca'):
            if int(row['lesion_ISUP'][-1]) == 1:
                ggg_code = CodedConcept(
                            value="1279715000",
                            meaning="Grade group 1 (Gleason score 3 + 3 = 6)",
                            scheme_designator="SCT"
                        )
            elif int(row['lesion_ISUP'][-1]) == 2:
                ggg_code = CodedConcept(
                            value="1279714001",
                            meaning="Grade group 2 (Gleason score 3 + 4 = 7)",
                            scheme_designator="SCT"
                        )
            elif int(row['lesion_ISUP'][-1]) == 3:
                ggg_code = CodedConcept(
                            value="1279716004",
                            meaning="Grade group 3 (Gleason score 4 + 3 = 7)",
                            scheme_designator="SCT"
                        )
            elif int(row['lesion_ISUP'][-1]) == 4:
                ggg_code = CodedConcept(
                            value="1279717008",
                            meaning="Grade group 4 (Gleason score 4 + 4 = 8)",
                            scheme_designator="SCT"
                        )
            elif int(row['lesion_ISUP'][-1]) == 5:
                ggg_code = CodedConcept(
                            value="1279720000",
                            meaning="Grade group 5 (Gleason score 4 + 5 = 9)",
                            scheme_designator="SCT"
                        )          
            # print(ggg_code) 
            #RelationshipTypeValues.CONTAINS
            if ggg_code is not None:
                evaluation.append(QualitativeEvaluation(CodedConcept(
                            value="1515521000004104",
                            meaning="International Society of Pathology histologic grade group",
                            scheme_designator="SCT"), ggg_code))
        
        ###########################################
        ########## PI-RADS scoring ################
        ###########################################
        
        if isinstance(row['PI_RADS'], int):
            if row['PI_RADS'] == 1:
                pirads_code = CodedConcept(
                        value="RID50296",
                        meaning="PI-RADS 1 - Very low (lesion)",
                        scheme_designator="RADLEX"
                    )
            elif row['PI_RADS'] == 2:
                pirads_code = CodedConcept(
                        value="RID50297",
                        meaning="PI-RADS 2 - Low (lesion)",
                        scheme_designator="RADLEX"
                    )
            elif row['PI_RADS'] == 3:
                pirads_code = CodedConcept(
                        value="RID50298",
                        meaning="PI-RADS 3 - Intermediate (lesion)",
                        scheme_designator="RADLEX"
                    )
            elif row['PI_RADS'] == 4:
                pirads_code = CodedConcept(
                        value="RID50299",
                        meaning="PI-RADS 4 - High (lesion)",
                        scheme_designator="RADLEX"
                    )
            elif row['PI_RADS'] == 5:
                pirads_code = CodedConcept(
                        value="RID50300",
                        meaning="PI-RADS 5 - Very high (lesion)",
                        scheme_designator="RADLEX"
                    )
            if pirads_code is not None:
                evaluation.append(QualitativeEvaluation(CodedConcept(
                        value="RID50295",
                        meaning="PI-RADS Lesion Assessment Category",
                        scheme_designator="RADLEX"), pirads_code))
        imaging_measurements.append(
            PlanarROIMeasurementsAndQualitativeEvaluations(
                tracking_identifier=TrackingIdentifier(
                    uid=generate_uid(),
                    identifier=str(target_name)
                ),
                referenced_region=referenced_region_3d,
                finding_type=codes.SCT.Lesion,
                # measurements=measurements,
                qualitative_evaluations=evaluation,
                # finding_sites=finding_sites
            )
        )  
        print(len(imaging_measurements))
        # Create the report content
        procedure_code = CodedConcept(
                            value="719178004", 
                            scheme_designator="SCT", 
                            meaning="Multiparametric magnetic resonance imaging of prostate"
        )
        measurement_report = MeasurementReport(
            observation_context=observation_context,
            procedure_reported=procedure_code,
            imaging_measurements=imaging_measurements
        )
        # Create the Structured Report instance
        series_instance_uid = generate_uid()
        sr_dataset = Comprehensive3DSR(
            evidence = [image_dataset],
            content = measurement_report[0],
            series_number = 100,
            series_instance_uid = series_instance_uid,
            sop_instance_uid = generate_uid(),
            instance_number = 1,
            manufacturer='IDC',
            is_complete = True,
            is_final = True
        )
        return sr_dataset
    else:
        # referenced_region_3d = ImageRegion3D(
        # graphic_type=GraphicTypeValues3D.POINT,
        # # revert to just biopsy point coordinate if no lesion size provided to consruct bbox
        # graphic_data=np.array([[-float(pos_items.group(1)), -float(pos_items.group(2)), float(pos_items.group(3))]]), 
        # frame_of_reference_uid=row['FrameOfReferenceUID']
        # )
        print('No bbox created - no lesion size given!')
        sr_dataset = 'None'
        return sr_dataset

def save_bbox_dcm_sr_for_case(
      row: pd.Series, 
      reference_dcm_file: str
    ):
    ''' 
        desc
        args
        return
    '''
    # read in reference dcm file and modify PatientSex
    subprocess.call(["dcmodify", "-m", "PatientSex=O", reference_dcm_file]) # use dcmodify to modify the PatientSex attribute
    image_dataset = dcmread(reference_dcm_file)
    # describe the context of reported observations: the person that reported
    # the observations and the device that was used to make the observations
    observer_person_context = ObserverContext(
        observer_type=codes.DCM.Person,
        observer_identifying_attributes=PersonObserverIdentifyingAttributes(
            name='Anonymous^Reader'
        )
    )
    observer_device_context = ObserverContext(
        observer_type=codes.DCM.Device,
        observer_identifying_attributes=DeviceObserverIdentifyingAttributes(
            uid=generate_uid()
        )
    )
    observation_context = ObservationContext(
        observer_person_context=observer_person_context,
    )
    
    ###############################################################
    ########## Target coordinates & Bounding Boxes ################
    ###############################################################
    bbox_slice_list = create_bbox_array(row)
    if bbox_slice_list != 'no_lesion_size':
        for bbox_slice in bbox_slice_list:
            imaging_measurements = []
            print("Parsing "+ row['pos_projected'])
            target_name = row['fid']
            pos_items = re.search(r"\ ?(-?[0-9]*\.[0-9]*)\ +(-?[0-9]*\.[0-9]*)\ +(-?[0-9]*\.[0-9]*)", row['pos_projected'])
            if not pos_items:
                print("Failed to parse "+ row['pos_projected'])
            source_image_ref = hd.sr.content.SourceImageForMeasurementGroup.from_source_image(image_dataset)
            referenced_region_3d = ImageRegion3D(
            graphic_type=GraphicTypeValues3D.POLYGON,
            graphic_data=np.array(bbox_slice), # column created during postprocessing
            frame_of_reference_uid=row['FrameOfReferenceUID']
            )
            # anatomical region
            if row['A_Description'] == 'TRANSITION ZONE':
                finding_location = CodedConcept(
                                value="399384005",
                                meaning="Transition zone of the prostate",
                                scheme_designator="SCT"
                            )
            elif row['A_Description'] == 'PERIPHERAL ZONE':
                finding_location = CodedConcept(
                                value="279706003",
                                meaning="Peripheral zone of the prostate",
                                scheme_designator="SCT"
                            )
            elif row['A_Description'] == 'STROMA':
                finding_location = CodedConcept(
                                value="717025007",
                                meaning="Anterior fibromuscular stroma of prostate",
                                scheme_designator="SCT"
                            )
            elif row['A_Description'] == 'VESICLE':
                finding_location = CodedConcept(
                                value="64739004",
                                meaning="Seminal vesicle",
                                scheme_designator="SCT"
                            )           
            # finding_sites = [
            #     FindingSite(anatomic_location=finding_location)
            # ]

            ###########################################
            ########## Gleason Grading ################
            ###########################################

            evaluation = []
            ggg_code = None
            #Clinical Significance
                # if int(row['lesion_ISUP'][-1]) >= 2:
                #   evaluation.append(CodeContentItem(CodedConcept(
                #                value="RID49502",
                #                meaning="clinically significant prostate cancer",
                #                scheme_designator="RADLEX"), codes.SCT.Yes, RelationshipTypeValues.CONTAINS))
            if isinstance(row['lesion_ISUP'], str) and (row['lesion_ISUP'].strip() != '') and (row['lesion_ISUP'].strip().lower() != 'nan') and (row['lesion_ISUP'].strip().lower() != 'benign'):
                if int(row['lesion_ISUP'][-1]) == 1:
                    ggg_code = CodedConcept(
                                value="1279715000",
                                meaning="Grade group 1 (Gleason score 3 + 3 = 6)",
                                scheme_designator="SCT"
                            )
                elif int(row['lesion_ISUP'][-1]) == 2:
                    ggg_code = CodedConcept(
                                value="1279714001",
                                meaning="Grade group 2 (Gleason score 3 + 4 = 7)",
                                scheme_designator="SCT"
                            )
                elif int(row['lesion_ISUP'][-1]) == 3:
                    ggg_code = CodedConcept(
                                value="1279716004",
                                meaning="Grade group 3 (Gleason score 4 + 3 = 7)",
                                scheme_designator="SCT"
                            )
                elif int(row['lesion_ISUP'][-1]) == 4:
                    ggg_code = CodedConcept(
                                value="1279717008",
                                meaning="Grade group 4 (Gleason score 4 + 4 = 8)",
                                scheme_designator="SCT"
                            )
                elif int(row['lesion_ISUP'][-1]) == 5:
                    ggg_code = CodedConcept(
                                value="1279720000",
                                meaning="Grade group 5 (Gleason score 4 + 5 = 9)",
                                scheme_designator="SCT"
                            )          
                # print(ggg_code) 
                #RelationshipTypeValues.CONTAINS
                if ggg_code is not None:
                    evaluation.append(QualitativeEvaluation(CodedConcept(
                                value="1515521000004104",
                                meaning="International Society of Pathology histologic grade group",
                                scheme_designator="SCT"), ggg_code))
            
            ###########################################
            ########## PI-RADS scoring ################
            ###########################################
            
            if isinstance(row['PI_RADS'], int):
                if row['PI_RADS'] == 1:
                    pirads_code = CodedConcept(
                                value="RID50296",
                                meaning="PI-RADS 1 - Very low (lesion)",
                                scheme_designator="RADLEX"
                            )
                elif row['PI_RADS'] == 2:
                    pirads_code = CodedConcept(
                                value="RID50297",
                                meaning="PI-RADS 2 - Low (lesion)",
                                scheme_designator="RADLEX"
                            )
                elif row['PI_RADS'] == 3:
                    pirads_code = CodedConcept(
                                value="RID50298",
                                meaning="PI-RADS 3 - Intermediate (lesion)",
                                scheme_designator="RADLEX"
                            )
                elif row['PI_RADS'] == 4:
                    pirads_code = CodedConcept(
                                value="RID50299",
                                meaning="PI-RADS 4 - High (lesion)",
                                scheme_designator="RADLEX"
                            )
                elif row['PI_RADS'] == 5:
                    pirads_code = CodedConcept(
                                value="RID50300",
                                meaning="PI-RADS 5 - Very high (lesion)",
                                scheme_designator="RADLEX"
                            )
                if pirads_code is not None:
                    evaluation.append(QualitativeEvaluation(CodedConcept(
                                value="RID50295",
                                meaning="PI-RADS Lesion Assessment Category",
                                scheme_designator="RADLEX"), pirads_code))
            imaging_measurements.append(
                PlanarROIMeasurementsAndQualitativeEvaluations(
                    tracking_identifier=TrackingIdentifier(
                        uid=generate_uid(),
                        identifier=str(target_name)
                    ),
                    referenced_region=referenced_region_3d,
                    finding_type=codes.SCT.Lesion,
                    # measurements=measurements,
                    qualitative_evaluations=evaluation,
                    # finding_sites=finding_sites
                )
            )  
        # storing all imaging measurements (bbox slices) in measurement_report
        print(len(imaging_measurements))
    
        procedure_code = CodedConcept(
                            value="719178004", 
                            scheme_designator="SCT", 
                            meaning="Multiparametric magnetic resonance imaging of prostate"
        )
        measurement_report = MeasurementReport(
            observation_context=observation_context,
            procedure_reported=procedure_code,
            imaging_measurements=imaging_measurements
        )
        # Create the Structured Report instance
        series_instance_uid = generate_uid()
        sr_dataset = Comprehensive3DSR(
            evidence = [image_dataset],
            content = measurement_report[0],
            series_number = 100,
            series_instance_uid = series_instance_uid,
            sop_instance_uid = generate_uid(),
            instance_number = 1,
            manufacturer='IDC',
            is_complete = True,
            is_final = True
        )
        return sr_dataset
    else:
        # referenced_region_3d = ImageRegion3D(
        # graphic_type=GraphicTypeValues3D.POINT,
        # # revert to just biopsy point coordinate if no lesion size provided to consruct bbox
        # graphic_data=np.array([[-float(pos_items.group(1)), -float(pos_items.group(2)), float(pos_items.group(3))]]), 
        # frame_of_reference_uid=row['FrameOfReferenceUID']
        # )
        print('No bbox created - no lesion size given!')
        sr_dataset = 'None'
        return sr_dataset

def save_point_dcm_sr_for_case(
      row: pd.Series, 
      reference_dcm_file: str
    ):
    ''' 
        desc
        args
        return
    '''
    # read in reference dcm file and modify PatientSex
    subprocess.call(["dcmodify", "-m", "PatientSex=O", reference_dcm_file]) # use dcmodify to modify the PatientSex attribute
    image_dataset = dcmread(reference_dcm_file)
    # describe the context of reported observations: the person that reported
    # the observations and the device that was used to make the observations
    observer_person_context = ObserverContext(
        observer_type=codes.DCM.Person,
        observer_identifying_attributes=PersonObserverIdentifyingAttributes(
            name='Anonymous^Reader'
        )
    )
    observer_device_context = ObserverContext(
        observer_type=codes.DCM.Device,
        observer_identifying_attributes=DeviceObserverIdentifyingAttributes(
            uid=generate_uid()
        )
    )
    observation_context = ObservationContext(
        observer_person_context=observer_person_context,
    )
    
    ###############################################################
    ########## Target coordinates & Bounding Boxes ################
    ###############################################################

    imaging_measurements = []
    print("Parsing "+ row['pos_projected'])
    target_name = row['fid']
    pos_items = re.search(r"\ ?(-?[0-9]*\.[0-9]*)\ +(-?[0-9]*\.[0-9]*)\ +(-?[0-9]*\.[0-9]*)", row['pos_projected'])
    if not pos_items:
      print("Failed to parse "+ row['pos_projected'])
    source_image_ref = hd.sr.content.SourceImageForMeasurementGroup.from_source_image(image_dataset)
    # biopsy target point coordinate
    referenced_region_3d = ImageRegion3D(
        graphic_type=GraphicTypeValues3D.POINT,
        # graphic_data=np.array([[-float(pos_items.group(1)), -float(pos_items.group(2)), float(pos_items.group(3))]]),
        graphic_data=np.array([[-float(pos_items.group(1)), -float(pos_items.group(2)), float(pos_items.group(3))]]),
        frame_of_reference_uid=row['FrameOfReferenceUID']
    )
    # anatomical region
    if row['A_Description'] == 'TRANSITION ZONE':
      finding_location = CodedConcept(
                    value="399384005",
                    meaning="Transition zone of the prostate",
                    scheme_designator="SCT"
                )
    elif row['A_Description'] == 'PERIPHERAL ZONE':
      finding_location = CodedConcept(
                    value="279706003",
                    meaning="Peripheral zone of the prostate",
                    scheme_designator="SCT"
                )
    elif row['A_Description'] == 'STROMA':
      finding_location = CodedConcept(
                    value="717025007",
                    meaning="Anterior fibromuscular stroma of prostate",
                    scheme_designator="SCT"
                )
    elif row['A_Description'] == 'VESICLE':
      finding_location = CodedConcept(
                    value="64739004",
                    meaning="Seminal vesicle",
                    scheme_designator="SCT"
                )           
    # finding_sites = [
    #     FindingSite(anatomic_location=finding_location)
    # ]

    ###########################################
    ########## Gleason Grading ################
    ###########################################

    evaluation = []
    ggg_code = None
   #Clinical Significance
    # if int(row['lesion_ISUP'][-1]) >= 2:
    #   evaluation.append(CodeContentItem(CodedConcept(
    #                value="RID49502",
    #                meaning="clinically significant prostate cancer",
    #                scheme_designator="RADLEX"), codes.SCT.Yes, RelationshipTypeValues.CONTAINS))
    if isinstance(row['lesion_ISUP'], str) and (row['lesion_ISUP'].strip() != '') and (row['lesion_ISUP'].strip().lower() != 'nan') and (row['lesion_ISUP'].strip().lower() != 'benign') and (row['lesion_ISUP'].strip().lower() != 'metastatic pca'):
        if int(row['lesion_ISUP'][-1]) == 1:
            ggg_code = CodedConcept(
                        value="1279715000",
                        meaning="Grade group 1 (Gleason score 3 + 3 = 6)",
                        scheme_designator="SCT"
                    )
        elif int(row['lesion_ISUP'][-1]) == 2:
            ggg_code = CodedConcept(
                        value="1279714001",
                        meaning="Grade group 2 (Gleason score 3 + 4 = 7)",
                        scheme_designator="SCT"
                    )
        elif int(row['lesion_ISUP'][-1]) == 3:
            ggg_code = CodedConcept(
                        value="1279716004",
                        meaning="Grade group 3 (Gleason score 4 + 3 = 7)",
                        scheme_designator="SCT"
                    )
        elif int(row['lesion_ISUP'][-1]) == 4:
            ggg_code = CodedConcept(
                        value="1279717008",
                        meaning="Grade group 4 (Gleason score 4 + 4 = 8)",
                        scheme_designator="SCT"
                    )
        elif int(row['lesion_ISUP'][-1]) == 5:
            ggg_code = CodedConcept(
                        value="1279720000",
                        meaning="Grade group 5 (Gleason score 4 + 5 = 9)",
                        scheme_designator="SCT"
                    )          
        # print(ggg_code) 
        #RelationshipTypeValues.CONTAINS
        if ggg_code is not None:
            evaluation.append(QualitativeEvaluation(CodedConcept(
                        value="1515521000004104",
                        meaning="International Society of Pathology histologic grade group",
                        scheme_designator="SCT"), ggg_code))
      
    ###########################################
    ########## PI-RADS scoring ################
    ###########################################
      
    if isinstance(row['PI_RADS'], int):
        if row['PI_RADS'] == 1:
          pirads_code = CodedConcept(
                      value="RID50296",
                      meaning="PI-RADS 1 - Very low (lesion)",
                      scheme_designator="RADLEX"
                  )
        elif row['PI_RADS'] == 2:
          pirads_code = CodedConcept(
                      value="RID50297",
                      meaning="PI-RADS 2 - Low (lesion)",
                      scheme_designator="RADLEX"
                  )
        elif row['PI_RADS'] == 3:
          pirads_code = CodedConcept(
                      value="RID50298",
                      meaning="PI-RADS 3 - Intermediate (lesion)",
                      scheme_designator="RADLEX"
                  )
        elif row['PI_RADS'] == 4:
          pirads_code = CodedConcept(
                      value="RID50299",
                      meaning="PI-RADS 4 - High (lesion)",
                      scheme_designator="RADLEX"
                  )
        elif row['PI_RADS'] == 5:
          pirads_code = CodedConcept(
                      value="RID50300",
                      meaning="PI-RADS 5 - Very high (lesion)",
                      scheme_designator="RADLEX"
                  )
        if pirads_code is not None:
          evaluation.append(QualitativeEvaluation(CodedConcept(
                      value="RID50295",
                      meaning="PI-RADS Lesion Assessment Category",
                      scheme_designator="RADLEX"), pirads_code))
    imaging_measurements.append(
        PlanarROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=TrackingIdentifier(
                uid=generate_uid(),
                identifier=str(target_name)
            ),
            referenced_region=referenced_region_3d,
            finding_type=codes.SCT.Lesion,
            # measurements=measurements,
            qualitative_evaluations=evaluation,
            # finding_sites=finding_sites
        )
    )  
    print(len(imaging_measurements))
    # Create the report content
    procedure_code = CodedConcept(
                        value="719178004", 
                        scheme_designator="SCT", 
                        meaning="Multiparametric magnetic resonance imaging of prostate"
    )
    measurement_report = MeasurementReport(
        observation_context=observation_context,
        procedure_reported=procedure_code,
        imaging_measurements=imaging_measurements
    )
    # Create the Structured Report instance
    series_instance_uid = generate_uid()
    sr_dataset = Comprehensive3DSR(
        evidence = [image_dataset],
        content = measurement_report[0],
        series_number = 100,
        series_instance_uid = series_instance_uid,
        sop_instance_uid = generate_uid(),
        instance_number = 1,
        manufacturer='IDC',
        is_complete = True,
        is_final = True
    )
    return sr_dataset

def export_mha_conversion_from_dcm(
        df_images: pd.DataFrame,
        output_dir: str
    ) -> pd.DataFrame:
    ''' 
        desc
            based on PI-CAI challenge conversion steps: 
            https://github.com/DIAGNijmegen/picai_prep/tree/main/src/picai_prep
        args
        return
    '''
    # add row with "picai_mapping" for json generation for picai-prep mha conversion
    mapping = {'AX_T2': 't2w', 'AX_DWI_1400': 'hbv', 'Apparent_D_nt__mm2_s_': 'adc'}
    df_images['picai_mapping'] = df_images['Modality'].map(mapping)
    # export mapping from rows in table (to map Seriesdescription / DICOMdescription with t2w, adc, hbv)
    bwh_nndet_mapping_dict = {"t2w": {"SeriesDescription": []}, "adc": {"SeriesDescription": []}, "hbv": {"SeriesDescription": []}}
    for index, row in df_images.iterrows():
        if row['picai_mapping'] == 't2w':
            bwh_nndet_mapping_dict['t2w']['SeriesDescription'].append(row['dcm_seriesdescription'])
        elif row['picai_mapping'] == 'adc':
            bwh_nndet_mapping_dict['adc']['SeriesDescription'].append(row['dcm_seriesdescription'])
        elif row['picai_mapping'] == 'hbv':
            bwh_nndet_mapping_dict['hbv']['SeriesDescription'].append(row['dcm_seriesdescription'])
    generate_dcm2mha_settings(
        archive_dir="/Volumes/bwh_prostate_ssd/bwh_picai_nndet_inf/DICOM",
        output_path="./bwh_nndet_dcm2mha_settings.json",
        mappings=bwh_nndet_mapping_dict
    )

    #IMPORTANT: DON'T FORGET TO SET THE SAME OPTIONS AS IN ./bwh_test3_dcm2mha_settings.json
    ''' 
    "options": {
        "num_threads": 1, 
        "verify_dicom_filenames": false,
        "allow_duplicates": false,
        "verbose": 2
        }
    '''

    #Conversion DICOM -> MHA with picai_prep
    archive = Dicom2MHAConverter(
        input_dir="/Volumes/bwh_prostate_ssd/bwh_picai_nndet_inf/DICOM",
        output_dir="/Volumes/bwh_prostate_ssd/bwh_picai_nndet_inf/MHA",
        dcm2mha_settings="./bwh_nndet_dcm2mha_settings.json"
    )
    archive.convert()

def export_dcm_selection_from_df(
        df_images: pd.DataFrame,
        output_dir: str
    ) -> pd.DataFrame:
    ''' 
        desc
            Loop through DICOM folder for two reasons:
                1) Filter out all dwi dicom slices with b=1400 to get rid of duplication (MultiVolume)
                2) Read in DICOM Seriesdescription for creating picai_prep mapping without having to 
                use semantic segmentation or a rule based heuristic
        args
        return
    '''
    list_dcm_sd = []
    slice_count = 0
    for index, row in df_images.iterrows():
        series_descr = row['Seriesdescription']
        series_path = row['Seriespath_pp']
        # check if hbv folder to apply filtering of non 1400 slices
        if 'DWI' in series_descr and not 'ADC' in series_descr:
            # reset slice counter to zero at beginning of each new studyseries
            slice_count = 0
            if not os.path.exists(series_path):
                print('Seriespath does not exist!')
                continue
            for filename in sorted(os.listdir(series_path)):
                if not filename.endswith(".dcm"):
                    print("Skipping file:", filename)
                    continue
                # read in DICOM file
                filepath = os.path.join(series_path, filename)
                if not os.path.exists(filepath):
                    print('filepath does not exist!')
                ds = pydicom.dcmread(filepath)
                if not ds:
                    print('ds empty!')
                # check slice counter -> if already volume full continue (to prevent doubles in cases with bvalue 1400 in first volume and non in last)
                if ds.get('ImagesInAcquisition') and ds.get('ScanningSequence') and ds.get('InstanceNumber'):
                    num_img = int(ds.get('ImagesInAcquisition'))
                    num_seq = len(ds.get('ScanningSequence'))
                    inst_num = int(ds.get('InstanceNumber'))
                    print(f'Slice count: {slice_count}')
                    if slice_count >= (num_img // num_seq) * (num_seq - 1):
                        print(f'Full amount of slices: {slice_count} achieved! Filepath: {filepath}')
                        break #when full slices hop to next study!
                # check if "DiffusionBValue" tag is equal to 1400
                if ds.get("DiffusionBValue") == 1400:
                    print(f'Using diffusivitybvalue! Filepath: {filepath}')
                    # create output directory - get rid of "MR" subfolder to create required folder structure for picai_prep json gen
                    dirs = series_path.split('/')
                    last_four_dirs = [d for d in dirs[-4:] if d != 'MR']
                    output_series_path = '/'.join(last_four_dirs)
                    output_dir_series = os.path.join(output_dir, output_series_path)
                    if not os.path.exists(output_dir_series):
                        os.makedirs(output_dir_series)
                    # copy file to output directory
                    output_filepath = os.path.join(output_dir_series, filename)
                    shutil.copy2(filepath, output_filepath)
                    slice_count += 1
                # catching cases that don't have "DiffusionBValue" tag and using "SequenceName" instead
                elif ds.get('SequenceName'):
                    print(f'Using sequence name! Filepath: {filepath}')
                    seq_name = ds.get('SequenceName')
                    pattern = r'1400'
                    match = re.search(pattern, seq_name)
                    pattern_cont = r'#[\d]'
                    match_cont = re.search(pattern_cont, seq_name)
                    # for cases that a single 1400 volume stored
                    if match and match_cont is None:
                        # Create output directory - get rid of "MR" subfolder to create required folder structure for picai_prep json gen
                        print(f'match: {match}')
                        print(f'match_cont: {match_cont}')
                        print('Single 1400 volume!')
                        dirs = series_path.split('/')
                        last_four_dirs = [d for d in dirs[-4:] if d != 'MR']
                        output_series_path = '/'.join(last_four_dirs)
                        output_dir_series = os.path.join(output_dir, output_series_path)
                        if not os.path.exists(output_dir_series):
                            os.makedirs(output_dir_series)
                        # copy file to output directory
                        output_filepath = os.path.join(output_dir_series, filename)
                        shutil.copy2(filepath, output_filepath)
                        slice_count += 1
                    # for cases that have multiple 1400 volumes stored
                    elif match and match_cont:
                        print('Multivolume but not #4!')
                        if match_cont.group(0)=='#4':
                            print('Multivolume 1400 #4 match!')
                            # create output directory - get rid of "MR" subfolder to create required folder structure for picai_prep json gen
                            dirs = series_path.split('/')
                            last_four_dirs = [d for d in dirs[-4:] if d != 'MR']
                            output_series_path = '/'.join(last_four_dirs)
                            output_dir_series = os.path.join(output_dir, output_series_path)
                            if not os.path.exists(output_dir_series):
                                os.makedirs(output_dir_series)
                            # copy file to output directory
                            output_filepath = os.path.join(output_dir_series, filename)
                            shutil.copy2(filepath, output_filepath)
                            slice_count += 1
                    else:
                        print(f'Discard brute force! Filepath: {filepath}')
                # brute force approach: Get ImagesInAcquistion, divide by and get last portion of scans (highest likelihood of having b=1400)
                elif ds.get('SequenceName') is None and ds.get("DiffusionBValue") is None and ds.get('ImagesInAcquisition') and ds.get('ScanningSequence') and ds.get('InstanceNumber'):
                    print(f'Brute force! Filepath: {filepath}')
                    num_img = int(ds.get('ImagesInAcquisition'))
                    num_seq = len(ds.get('ScanningSequence'))
                    inst_num = int(ds.get('InstanceNumber'))
                    # print(f'inst_num: {inst_num}')
                    # print(f'Start last vol: {(num_img // num_seq) * (num_seq - 1)}')
                    # print(f'In last volume: {inst_num > (num_img // num_seq) * (num_seq - 1)}')
                    if inst_num > (num_img // num_seq) * (num_seq - 1): #filter out the slices that are part of the last volume
                        print(f'In last volume! Filepath: {filepath}')
                        # create output directory - get rid of "MR" subfolder to create required folder structure for picai_prep json gen
                        dirs = series_path.split('/')
                        last_four_dirs = [d for d in dirs[-4:] if d != 'MR']
                        output_series_path = '/'.join(last_four_dirs)
                        output_dir_series = os.path.join(output_dir, output_series_path)
                        if not os.path.exists(output_dir_series):
                            os.makedirs(output_dir_series)
                        # copy file to output directory
                        output_filepath = os.path.join(output_dir_series, filename)
                        shutil.copy2(filepath, output_filepath)
                        slice_count += 1
                    else:
                        print(f'Discard brute force! Filepath: {filepath}')
        else:
            if not os.path.exists(series_path):
                print('Seriespath does not exist!')
                continue
            # set incr_count to zero before using it to catch duplicates in next loop
            old_incr = 0 
            for filename in sorted(os.listdir(series_path)):
                if not filename.endswith(".dcm"):
                    print("Skipping file:", filename)
                    continue
                # read in DICOM file
                filepath = os.path.join(series_path, filename)
                # print(filepath)
                if not os.path.exists(filepath):
                    print('filepath does not exist!')
                ds = pydicom.dcmread(filepath)
                if not ds:
                    print('ds empty!')
                # catch doubled slices
                if ds.get('InstanceNumber'):
                    incr_count = ds.get('InstanceNumber')
                    if incr_count > old_incr:
                        # create output directory - get rid of "MR" subfolder to create required folder structure for picai_prep json gen
                        dirs = series_path.split('/')
                        last_four_dirs = [d for d in dirs[-4:] if d != 'MR']
                        output_series_path = '/'.join(last_four_dirs)
                        output_dir_series = os.path.join(output_dir, output_series_path)
                        if not os.path.exists(output_dir_series):
                            os.makedirs(output_dir_series)
                        # Copy file to output directory
                        output_filepath = os.path.join(output_dir_series, filename)
                        shutil.copy2(filepath, output_filepath)
                        #Update old_incr to currently highest slice number
                        old_incr = incr_count
                    else:
                        print(f'Continue - duplicate slide! Filepath: {filepath}')
                        continue
                # create output directory - get rid of "MR" subfolder to create required folder structure for picai_prep json gen
                dirs = series_path.split('/')
                last_four_dirs = [d for d in dirs[-4:] if d != 'MR']
                output_series_path = '/'.join(last_four_dirs)
                output_dir_series = os.path.join(output_dir, output_series_path)
                if not os.path.exists(output_dir_series):
                    os.makedirs(output_dir_series)
                # copy file to output directory
                output_filepath = os.path.join(output_dir_series, filename)
                shutil.copy2(filepath, output_filepath)
        # in both cases read out SeriesDescription of the last dicom slice (stored in ds) for every series
        list_dcm_sd.append(ds.get("SeriesDescription"))
    # append list as additional column to merged_df
    df_images['dcm_seriesdescription'] = list_dcm_sd
    return df_images

def images_export(
        df_images: pd.DataFrame,
        img_export_dtype: list,
        imgs_output_dir: str
    ) -> None:
    ''' 
        desc
        args
        return
    '''
    if img_export_dtype == 'dcm':
        df_images = export_dcm_selection_from_df(
                                    df_images, 
                                    imgs_output_dir
        )
    elif img_export_dtype == 'mha':
       pass #TODO: adapt script written in matching.ipynb
        # export_mha_conversion_from_dcm(
        #                             df_images,
        #                             )
    elif img_export_dtype == 'nifti':
        pass #TODO: similar to picai does to convert mha-> nifiti before using nnDetection

def annotations_export(
        df_annotations: pd.DataFrame,
        annotations_export_dtype: list,
        annotations_output_dir: str,
        dcm_sr_type: str
    ) -> None:
    ''' 
        desc
        args
        return
    '''
    counter = 1
    old_local_file = None
    old_study_id = None
    if annotations_export_dtype == 'dcm':
        for index, row in tqdm(df_annotations.iterrows(), desc='dcm_annotation_export'):
            local_file = row['closest_axt2_dcm_path']
            patient_id = str(row['PatientID']).zfill(8)
            study_id = str(row['ANON_AccNumber']).zfill(10)
            # to not overwrite SRs with the same closest slice (multiple targets for one study)
            # do not use closest slice, cause sometimes different closest slice for same study -> use study_id!
            if study_id == old_study_id:
                counter += 1
                old_study_id = study_id
            elif study_id != old_local_file:
                counter = 1
                old_study_id = study_id
            # dcm sreport for biopsy point coordinates
            if dcm_sr_type == 'point':
                sr_dataset = save_point_dcm_sr_for_case(
                                            row, 
                                            local_file
                )
                pydicom.write_file(
                    os.path.join(annotations_output_dir, f"{patient_id}_{study_id}_point_SR_{counter}.dcm"), 
                    sr_dataset
                )
            # dcm sreport for bounding boxes
            elif dcm_sr_type == 'bbox':
                sr_dataset = save_single_bbox_dcm_sr_for_case(
                                            row, 
                                            local_file
                )
                if sr_dataset != 'None':
                    pydicom.write_file(
                        os.path.join(annotations_output_dir, f"{patient_id}_{study_id}_bbox_SR_{counter}.dcm"), 
                        sr_dataset
                    )
    elif annotations_export_dtype == 'nifti':
       pass #TODO: adapt script written in matching.ipynb