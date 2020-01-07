import os
import sys
import xml.etree.ElementTree as ET
import glob

indir = 'C:\\Users\\kctoa\\Desktop\\VScode\\AI_Project\\PokemonData\\data\\Annotations\\'
outdir = 'C:\\Users\\kctoa\\Desktop\\VScode\\AI_Project\\PokemonData\\data\\Images\\'
file_name = 'dataset.txt'

def xml_to_txt(indir,outdir):
    os.chdir(indir)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations)+'*.xml')

    f_w = open(outdir + '\\' + file_name, 'w')

    for i, file in enumerate(annotations):
        in_file = open(file)
        tree=ET.parse(in_file)
        root = tree.getroot()
        filename = root.find('filename').text
        for obj in root.iter('object'):
                current = list()
                name = obj.find('name').text

                xmlbox = obj.find('bndbox')
                xn = xmlbox.find('xmin').text   
                xx = xmlbox.find('xmax').text
                yn = xmlbox.find('ymin').text
                yx = xmlbox.find('ymax').text

                f_w.write(filename +', '+xn+', '+yn+', '+xx+', '+yx+', ')
                f_w.write(name+'\n')

xml_to_txt(indir, outdir)