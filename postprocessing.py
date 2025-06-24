import numpy as np
import pandas as pd
import re
import os
import glob
from pydicom.filereader import dcmread
from typing import Tuple
from sklearn.preprocessing import normalize

def listdir_nohidden(
        path: str
    ) -> str:
    '''
        desc
            Utility function to ignore .DS_Store when listing dir files.
            From: https://stackoverflow.com/questions/7099290/how-to-ignore-hidden-files-using-os-listdir
        args
        return
    '''
    return glob.glob(os.path.join(path, '*'))

def convert_lesion(
        lsize: str
    ) -> str:
    ''' 
        desc
        args
        return
    '''
    lesion_mm = 'no_match'
    if (lsize != '') and (lsize != 'no_match') and (lsize != 'no_hypotheses_or_no_premises'):
        if 'cm' in lsize:
            if re.search(r'(\d+\.?\d*)', lsize):
                lesion_cm = float(re.search(r'(\d+\.?\d*)', lsize)[0])
                lesion_mm = int(lesion_cm*10)
        elif 'mm' in lsize:
            lesion_mm = int(float(re.search(r'(\d+\.?\d*)', lsize)[0]))
        else:
            print('No mm/cm found!')
        lesion_mm = str(lesion_mm)
    else:
        print('Empty Lesion_Size row!')
    return lesion_mm

def convert_pirads(
        pirads: str
    ) -> str:
    ''' 
        desc
        args
        return
    '''
    pirads_score = 'no_match'
    if (pirads != '') and (pirads != 'no_match') and (pirads != 'no_hypotheses_or_no_premises'):
        if 'PI-RADS' in pirads:
            pirads_score = int(re.search(r'(\d+\.?\d*)', pirads)[0])
            pirads_score = str(pirads_score)
        else:
            print('No PI_RADS found!')
    else:
        print('Empty PI_RADS row!')
    return pirads_score

def dfs_cleaning(
        df_images: pd.DataFrame,
        df_annotations: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ''' 
        desc
            Minimal requirement for study to be kept is to have
            an identified biopsy target coordinate and matching
            histopathology.
        args
        return
    '''
    # TODO: further cleaning steps can be integrated (to keep manual intervention to minimum)
    df_annotations = df_annotations[
        (df_annotations['lesion_ISUP'] != 'no_match') &
        (df_annotations['lesion_GS'] != 'no_match')
    ]
    # also delete all studies from df_images that are not in df_annotations for consistency
    df_images = df_images[df_images['Studypath_pp'].isin(df_annotations['Studypath_pp'])]
    return df_images, df_annotations

def create_ohif_fid(
        row: pd.Series
    ) -> str:
    ''' 
        desc
        args
        return
    '''
    fid = None
    gs_str = str(row['lesion_GS']) if isinstance(row['lesion_GS'], float) else row['lesion_GS']
    isup_str = str(row['lesion_ISUP']) if isinstance(row['lesion_ISUP'], float) else row['lesion_ISUP']
    pirads_str = str(row['PI_RADS']) if isinstance(row['PI_RADS'], float) else row['PI_RADS']
    les_size_str = str(row['Lesion_Size']) if isinstance(row['Lesion_Size'], float) else row['Lesion_Size']
    # catch empty elements
    if gs_str is np.nan:
        gs_str = ''
    if isup_str is np.nan:
        isup_str = ''
    if pirads_str is np.nan:
        pirads_str = ''
    if les_size_str is np.nan:
        les_size_str = ''        
    fid = row['A_Region'] + ', GS: ' + gs_str + ', GGG: ' + isup_str.split()[-1] + ', PI_RADS: ' + pirads_str + ', max_lesion_diam[mm]: ' + les_size_str
    return fid

def create_ohif_pos(
        row: pd.Series
    ) -> str:
    ''' 
        desc
            create 3-tuple out of three biopsy target coordinates
            expected in dcm conversion script.
            IMPORTANT: Project biopsy coordinate point onto nearest
            slice for proper visualization in OHIF.
        args
        return
    '''
    # apply lambda function to concatenate the values from the three columns into a string with two spaces
    pos = '{:.6f}  {:.6f}  {:.6f}'.format(row['X_Coordinate'], row['Y_Coordinate'], row['Z_Coordinate'])
    return pos

def find_closest_element(
        dcm_obj_list: list, 
        target_value: float,
    ):
    ''' 
        desc
        args
        return
    '''
    closest_element = None
    min_difference = float('inf')
    for slice_element in dcm_obj_list:
        difference = abs(float(slice_element.SliceLocation) - float(target_value))
        if difference < min_difference:
            min_difference = difference
            closest_element = slice_element
    return closest_element

def get_point_on_closest_slice(
        x_coordinate: float,
        z_coordinate: float,
        normal_vector: list
    ) -> list:
    ''' 
        desc
            Get missing third coordinate of point on slice based on 
            condition that dot product of all points / vectors in plane
            with normal vector describing plane needs to be =0 
            xn1 + yn2 + zn3 = 0
        args
        return
    '''
    # get missing third coordinate of point on slice 
    y_coordinate = 1/normal_vector[1]*(-x_coordinate*normal_vector[0]-z_coordinate*normal_vector[2])
    point_on_slice = np.array([x_coordinate, y_coordinate, z_coordinate])
    return point_on_slice

def project_point_on_axt2w_slice(
        row: pd.Series
    ) -> str:
    '''
        desc
            1) use studypath to t2w image series as root (:= reference_dcm_obj)
            2) read in all .dcm slice files that are in folder and get their SliceLocation
            3) read in z-coordinate of biopsy point
            4) go trhough list of all slices and pick out slice that is closest to z-coordinate
            5) project entire biopsy target onto neareast t2w slice
            6) create new column with modified / projected biopsy coordinates
        args
        return 
    '''
    # get studypath to axt2 image series per study and patient
    reference_dcm_obj = listdir_nohidden(row['Seriespath_pp']) # get rid of .DS_Store
    # get list of all dcm slice objects
    dcm_obj_list = [dcmread(os.path.join(row['Seriespath_pp'],reference_dcm_file)) for reference_dcm_file in reference_dcm_obj]
    # get z-coordinate from table
    pos_items = re.search(r"\ ?(-?[0-9]*\.[0-9]*)\ +(-?[0-9]*\.[0-9]*)\ +(-?[0-9]*\.[0-9]*)", row['pos'])
    x_c = float(pos_items.group(1))
    y_c = float(pos_items.group(2))
    z_c = float(pos_items.group(3))
    # get slice that is closest to the z-coordinate of the biopsy coordinate
    closest_slice = find_closest_element(dcm_obj_list, z_c)
    ImageOrientationPatient, ImagePositionPatient = closest_slice.ImageOrientationPatient, closest_slice.ImagePositionPatient
    # project point onto nearest slice https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d
    point_3d = np.array([x_c, y_c, z_c])
    normal_vec = np.cross(np.array(ImageOrientationPatient[:3]), np.array(ImageOrientationPatient[3:7]))
    # normal_vec = normalize(normal_vec.reshape(1,-1))[0]
    # nearest slice is 2D plan defined by normal vector and  SliceLocation (x,y can be chosen freely)
    # point_on_slice = get_point_on_closest_slice(
    #                         20.0, # can be chosen freely (chosing x_c for convenience)
    #                         float(closest_slice.SliceLocation),
    #                         normal_vec
    # )
    point_on_slice = np.array(ImagePositionPatient)
    vector = np.subtract(point_3d, point_on_slice)
    distance = np.dot(vector, normal_vec)
    projected_point = point_3d - distance * normal_vec
    # validate projection by checking z-coordinate of projected point against nearest slice location
    # TODO: work out proper projection of point to reduce x,y shift due to angled slices
    # projected_point = np.array([x_c,y_c, float(closest_slice.SliceLocation)])
    # if projected_point[2] != float(closest_slice.SliceLocation):
    #     print(f'Z-c of projected point: {projected_point[2]} does not match closest slice location: {float(closest_slice.SliceLocation)}')
    #     projected_point = np.array([-1,-1,-1])
    projected_point_str = '{:.6f}  {:.6f}  {:.6f}'.format(projected_point[0], projected_point[1], projected_point[2])
    return projected_point_str

def create_bbox_array(
        row: pd.Series
    ) -> list:
    ''' 
        desc
        args
        return
    '''
    bbox_array = []
    # create fix rectangle per slice based on lesion diameter and projected_point!
    pos_items = re.search(r"\ ?(-?[0-9]*\.[0-9]*)\ +(-?[0-9]*\.[0-9]*)\ +(-?[0-9]*\.[0-9]*)", row['pos_projected'])
    x_c = float(pos_items.group(1))
    y_c = float(pos_items.group(2))
    z_c = float(pos_items.group(3))
    # filter out erroneous projection (-1, -1, -1)
    if pos_items == "(-1 -1 -1)":
        print('Point projection not correct, jumping annotation!')
        bbox_array = 'projection_error'
        return bbox_array
    if row['Lesion_Size'] == 'no_match':
        print('No lesion size found, jumping bbox!')
        bbox_array = 'no_lesion_size'
        return bbox_array
    xmin, xmax = x_c - int(row['Lesion_Size'])/2, x_c + int(row['Lesion_Size'])/2
    ymin, ymax = y_c - int(row['Lesion_Size'])/2, y_c + int(row['Lesion_Size'])/2
    # inverse sign for x and y for correct visuallization in ohif (ras, lps)
    xmin, xmax = -xmin, -xmax
    ymin, ymax = -ymin, -ymax
    # zmin, zmax = z_c - int(row['Lesion_Size'])/2, z_c + int(row['Lesion_Size'])/2
    # # use lesion diameter/2 with rounding to determin all z coordinates of bboxes
    # reference_dcm_obj = listdir_nohidden(row['Seriespath_pp']) # get rid of .DS_Store
    # for t2w_slice in reference_dcm_obj:
    #     slice_loc = float(dcmread(os.path.join(row['Seriespath_pp'],t2w_slice)).SliceLocation)
    #     if (slice_loc <= zmax) & (slice_loc >= zmin):
    #         bbox_array.append([
    #                     (xmin, ymin, slice_loc),
    #                     (xmax, ymin, slice_loc),
    #                     (xmax, ymax, slice_loc),
    #                     (xmin, ymax, slice_loc),
    #                     (xmin, ymin, slice_loc),
    #         ])
    # bbox_array = np.array(bbox_array)
    bbox_list = [
                (xmin, ymin, z_c),
                (xmax, ymin, z_c),
                (xmax, ymax, z_c),
                (xmin, ymax, z_c),
                (xmin, ymin, z_c),
    ]
    return bbox_list

def get_closest_axt2_dcm_path(
        row: pd.Series
    ) -> str:
    ''' 
        desc
            Same procedure as in project_point_on_axt2w_slice function for getting nearest slice.
        args
        return
    '''
    # get studypath to axt2 image series per study and patient
    reference_dcm_obj = listdir_nohidden(row['Seriespath_pp']) # get rid of .DS_Store
    # get list of all dcm slice objects
    dcm_obj_list = [dcmread(os.path.join(row['Seriespath_pp'],reference_dcm_file)) for reference_dcm_file in reference_dcm_obj]
    # get z-coordinate from table
    pos_items = re.search(r"\ ?(-?[0-9]*\.[0-9]*)\ +(-?[0-9]*\.[0-9]*)\ +(-?[0-9]*\.[0-9]*)", row['pos'])
    x_c = pos_items.group(1)
    y_c = pos_items.group(2)
    z_c = pos_items.group(3)
    # get slice that is closest to the z-coordinate of the biopsy coordinate
    closest_slice = find_closest_element(dcm_obj_list, z_c)
    path = [os.path.join(row['Seriespath_pp'],reference_dcm_file) for reference_dcm_file in reference_dcm_obj if float(dcmread(os.path.join(row['Seriespath_pp'],reference_dcm_file)).SliceLocation) == float(closest_slice.SliceLocation)][0]
    return path

def get_dcm_foruid(
        row: pd.Series
    ) -> str:
    ''' 
        desc
        args
        return
    '''
    closest_axt2_for = None
    closest_axt2_for = dcmread(row['closest_axt2_dcm_path']).FrameOfReferenceUID
    return closest_axt2_for

def create_ohif_labels(
        df_annotations: pd.DataFrame
    ) -> pd.DataFrame:
    ''' 
        desc
        args
        return
    '''
    # create new columns with projected biopsy targets for OHIF visualizations
    df_annotations.insert(loc=5, column='fid', value=df_annotations.apply(create_ohif_fid, axis=1))
    df_annotations.insert(loc=6, column='pos', value=df_annotations.apply(create_ohif_pos, axis=1))
    df_annotations.insert(loc=7, column='pos_projected', value=df_annotations.apply(project_point_on_axt2w_slice, axis=1))
    # df_annotations.insert(loc=8, column='bbox_projected', value=df_annotations.apply(create_bbox_array, axis=1))
    df_annotations.insert(loc=9, column='closest_axt2_dcm_path', value=df_annotations.apply(get_closest_axt2_dcm_path, axis=1))
    df_annotations.insert(loc=10, column='FrameOfReferenceUID', value=df_annotations.apply(get_dcm_foruid, axis=1))
    return df_annotations