'''
The followuing code will extract the nodule patches; however, many of the 
extracted images are black so those images should be manually removed 
from the training set.
Also, directories has to be changed accordingly.

'''
import SimpleITK as sitk
import numpy as np
from glob import glob
import pandas as pd
import scipy.ndimage
import os
import array
import math

try:
    from tqdm import tqdm 
except:
    print('tqdm not found')
    tqdm = lambda x: x
# import traceback

workspace = './'


class nodules_crop(object):
    def __init__(self, workspace):
        """param: workspace"""
        self.workspace = workspace
        luna_path = "C:/Users/mail2/Desktop/LUNA_16_dataset/"
        self.all_patients_path = os.path.join(self.workspace, luna_path + "subset2/subset2/")
        print("all_patients_path: %s" % (self.all_patients_path))

        self.nodules_npy_path = "C:/Users/mail2/Desktop/LUNA_16_dataset/nodule_cubes/npy/"
        self.all_annotations_mhd_path = "C:/Users/mail2/Desktop/LUNA_16_dataset/nodule_cubes/mhd/"
        self.all_candidates_mhd_path = "C:/Users/mail2/Desktop/LUNA_16_dataset/nodule_cubes/mhd/"
        self.ls_all_patients = glob(self.all_patients_path + "*.mhd")

        self.df_annotations = pd.read_csv(self.workspace + "csv_files/annotations.csv")
        self.df_annotations["file"] = self.df_annotations["seriesuid"].map(
            lambda file_name: self.get_filename(self.ls_all_patients, file_name))
        self.df_annotations = self.df_annotations.dropna()
        self.df_candidates = pd.read_csv(self.workspace + "csv_files/candidates.csv")
        self.df_candidates["file"] = self.df_candidates["seriesuid"].map(
            lambda file_name: self.get_filename(self.ls_all_patients, file_name))
        self.df_candidates = self.df_candidates.dropna()

    def set_window_width(self, image, MIN_BOUND=-1000.0):
        image[image < MIN_BOUND] = MIN_BOUND
        return image

    def resample(self, image, old_spacing, new_spacing=[1, 1, 1]):
        resize_factor = old_spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = old_spacing / real_resize_factor
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
        return image, new_spacing

    def write_meta_header(self, filename, meta_dict):
        header = ''
        # do not use tags = meta_dict.keys() because the order of tags matters
        tags = ['ObjectType', 'NDims', 'BinaryData',
                'BinaryDataByteOrderMSB', 'CompressedData', 'CompressedDataSize',
                'TransformMatrix', 'Offset', 'CenterOfRotation',
                'AnatomicalOrientation',
                'ElementSpacing',
                'DimSize',
                'ElementType',
                'ElementDataFile',
                'Comment', 'SeriesDescription', 'AcquisitionDate', 'AcquisitionTime', 'StudyDate', 'StudyTime']
        for tag in tags:
            if tag in meta_dict.keys():
                header += '%s = %s\n' % (tag, meta_dict[tag])
        f = open(filename, 'w')
        f.write(header)
        f.close()

    def dump_raw_data(self, filename, data):
        """ Write the data into a raw format file. Big endian is always used. """
        # Begin 3D fix
        data = data.reshape([data.shape[0], data.shape[1] * data.shape[2]])
        # End 3D fix
        rawfile = open(filename, 'wb')
        a = array.array('f')
        for o in data:
            a.fromlist(list(o))
        # if is_little_endian():
        #    a.byteswap()
        a.tofile(rawfile)
        rawfile.close()

    def write_mhd_file(self, mhdfile, data, dsize):
        assert (mhdfile[-4:] == '.mhd')
        meta_dict = {}
        meta_dict['ObjectType'] = 'Image'
        meta_dict['BinaryData'] = 'True'
        meta_dict['BinaryDataByteOrderMSB'] = 'False'
        meta_dict['ElementType'] = 'MET_FLOAT'
        meta_dict['NDims'] = str(len(dsize))
        meta_dict['DimSize'] = ' '.join([str(i) for i in dsize])
        meta_dict['ElementDataFile'] = os.path.split(mhdfile)[1].replace('.mhd', '.raw')
        self.write_meta_header(mhdfile, meta_dict)
        pwd = os.path.split(mhdfile)[0]
        if pwd:
            data_file = pwd + '/' + meta_dict['ElementDataFile']
        else:
            data_file = meta_dict['ElementDataFile']
        self.dump_raw_data(data_file, data)

    def save_annotations_nodule(self, nodule_crop, name_index):
        np.save(os.path.join(self.nodules_npy_path, "nodule.1%06d.npy" % (name_index)), nodule_crop)
        # np.save(self.nodules_npy_path + str(1) + "_" + str(name_index) + '_annotations' + '.npy', nodule_crop)
        # self.write_mhd_file(self.all_annotations_mhd_path + str(1) + "_" + str(name_index) + '_annotations' + '.mhd', nodule_crop,nodule_crop.shape)

   
    def save_candidates_nodule(self, nodule_crop, name_index, cancer_flag):
        # np.save(self.nodules_npy_path + str(cancer_flag) + "_" + str(name_index) + '_candidates' + '.npy', nodule_crop)
        np.save(os.path.join(self.nodules_npy_path, "%01d%06dcandidates.npy" % (cancer_flag, name_index)),
                nodule_crop)
        # self.write_mhd_file(self.all_candidates_mhd_path + str(cancer_flag) + "_" + str(name_index) + '.mhd', nodule_crop,nodule_crop.shape)

    def get_filename(self, file_list, case):
        for f in file_list:
            if case in f:
                return (f)

    def annotations_crop(self):
        for patient in enumerate(tqdm(self.ls_all_patients)):
            patient = patient[1]
            print(patient)

            if patient not in self.df_annotations.file.values:
                print('Patient ' + patient + 'Not exist!')
                continue
            patient_nodules = self.df_annotations[self.df_annotations.file == patient]
            full_image_info = sitk.ReadImage(patient)
            full_scan = sitk.GetArrayFromImage(full_image_info)
            origin = np.array(full_image_info.GetOrigin())[::-1]  
            old_spacing = np.array(full_image_info.GetSpacing())[::-1]  
            image, new_spacing = self.resample(full_scan, old_spacing)  
            print('Resample Done')
            for index, nodule in patient_nodules.iterrows():
                nodule_center = np.array([nodule.coordZ, nodule.coordY, nodule.coordX])  
                v_center = np.rint((nodule_center - origin) / new_spacing)  
                v_center = np.array(v_center, dtype=int)
        
                if nodule.diameter_mm < 5:
                    window_size = 7
                elif nodule.diameter_mm < 10:
                    window_size = 9
                elif nodule.diameter_mm < 20:
                    window_size = 15
                elif nodule.diameter_mm < 25:
                    window_size = 17
                elif nodule.diameter_mm < 30:
                    window_size = 20
                else:
                    window_size = 22
                zyx_1 = v_center - window_size 
                zyx_2 = v_center + window_size + 1
                nodule_box = np.zeros([45, 45, 45], np.int16)  # ---nodule_box_size = 45
                img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]  
                img_crop = self.set_window_width(img_crop)  # 
                zeros_fill = math.floor((45 - (2 * window_size + 1)) / 2)
                try:
                    nodule_box[zeros_fill:45 - zeros_fill, zeros_fill:45 - zeros_fill,
                    zeros_fill:45 - zeros_fill] = img_crop  #nodule_box
                except:
                    # f = open("log.txt", 'a')
                    # traceback.print_exc(file=f)
                    # f.flush()
                    # f.close()
                    continue
                nodule_box[nodule_box == 0] = -1000  
                self.save_annotations_nodule(nodule_box, index)
            print('Done for this patient!\n\n')
        print('Done for all!')
        
    
if __name__ == '__main__':
    nc = nodules_crop(workspace)
    nc.annotations_crop()