"""
This file contains functions utilized in the collection and labeling of Google Earth images for use as EO data in
the dark-ships project.

@author: rdalrymple
"""
import os
from PIL import Image
from xml.dom import minidom
from zipfile import ZipFile
from bs4 import BeautifulSoup
from tqdm import tqdm

INPUT_DIR = os.path.join(os.getcwd(), '..', 'data_inputs')


def parse_kmz(kmz_fname: str) -> list:
    """
    This function unzips a kmz file and parses the resulting kml for name, latitude, longitude, meter width, date,
    and time estimate for the Google Earth Placemark.
    @author: rdalrymple

    Parameters
    ----------
    kmz_fname: str containing file path to kmz file

    Returns
    -------
    list containing a dict for every Placemark object within the kmz file. Each dict contains name, date, time,
    meter width, and lat/lon bounds of the place of interest.
    """
    kmz = ZipFile(kmz_fname, 'r')
    kml = kmz.open('doc.kml', 'r').read()
    bs_data = BeautifulSoup(kml, "xml")
    pm_list = []
    for pm in bs_data.find_all('kml:Placemark'):
        pm_dict = {'name': pm.find('kml:name').get_text()}
        pm_dict['date'] = pm_dict['name'][-10:].replace('_', '-')
        text = pm.find('kml:description').get_text()
        time = text[:text.find('UTC') + 5].replace(':', '')
        hour = int(time[:time.find(' ')])
        pm_dict['time'] = '{:04d}{}'.format(hour, time[time.find(' '):])
        pm_dict['meter_width'] = text[text.find('\n\n') + 2:text.find('m')]
        text = text[text.find('m') + 3:]
        pm_dict['lat1'] = text[:text.find(',')]
        pm_dict['lon1'] = text[text.find(',') + 2: text.find('\n')]
        text = text[text.find('\n') + 1:]
        pm_dict['lat2'] = text[:text.find(',')]
        pm_dict['lon2'] = text[text.find(',') + 2:]
        pm_list.append(pm_dict)

    return pm_list


def format_metadata_as_voc(google_earth_data_dir: str = os.path.join(INPUT_DIR, 'google_earth_EO')):
    """
    This function loops through the Google Earth dataset and takes all information from the images, kmz files, and
    ship truth labels and formats it into a VOC formatted xml file to be used in YOLO training. The VOC formatted xml
    files will be placed in the 'annotations' directory under the overarching directory provided.

    Parameters
    ----------
    google_earth_data_dir: str
        Path to directory containing all Google Earth data. This directory should contain places, images, ship_labels,
        and annotations folders. Within each of the places, images, and ship_labels folders should be 6 sets and within
        each set should be 10 batches of data. The annotations folder is where the VOC formatted xml files will be
        placed.

    @author: rdalrymple
    """
    places_dir = os.path.join(google_earth_data_dir, 'places')
    images_dir = os.path.join(google_earth_data_dir, 'images')
    ship_labels_dir = os.path.join(google_earth_data_dir, 'ship_labels')
    annotations_dir = os.path.join(google_earth_data_dir, 'voc_annotations')

    for set_dir in tqdm(os.listdir(places_dir), desc='Compiling metadata for sets of images'):
        for batch in os.listdir(os.path.join(places_dir, set_dir)):
            placemarks = parse_kmz(os.path.join(places_dir, set_dir, batch))
            for pm in placemarks:
                root = minidom.Document()
                ann = root.createElement('annotation')
                root.appendChild(ann)

                annotator = root.createElement('annotator')
                annotator_text = root.createTextNode('mjerge@riversideresearch.org')
                annotator.appendChild(annotator_text)
                ann.appendChild(annotator)

                source = root.createElement('source')
                database = root.createElement('database')
                database_text = root.createTextNode('GoogleEarth')
                database.appendChild(database_text)
                source.appendChild(database)

                dataset_collector = root.createElement('dataset_collector')
                dataset_collector_text = root.createTextNode('rdalrymple@riversideresearch.org')
                dataset_collector.appendChild(dataset_collector_text)
                source.appendChild(dataset_collector)
                ann.appendChild(source)

                folder = root.createElement('folder')
                folder_text = root.createTextNode('images/' + set_dir + '/' + batch[:-4])
                folder.appendChild(folder_text)
                ann.appendChild(folder)

                # print('{}/{}/{}'.format(set_dir, batch[:-4], pm['name']))
                filename = root.createElement('filename')
                filename_text = root.createTextNode(pm['name'] + '.jpg')
                filename.appendChild(filename_text)
                ann.appendChild(filename)

                date = root.createElement('date')
                date_text = root.createTextNode(pm['date'])
                date.appendChild(date_text)
                ann.appendChild(date)

                time = root.createElement('time_estimate')
                utc_offset = int(pm['time'][pm['time'].find('UTC')+3:]) * 100
                time_str = pm['time'][:pm['time'].find('UTC')+3]
                time_text = root.createTextNode(time_str + '{:+05d}'.format(utc_offset))
                time.appendChild(time_text)
                ann.appendChild(time)

                img_f = os.path.join(images_dir, set_dir, batch[:-4], pm['name'] + '.jpg')
                img = Image.open(img_f)
                size = root.createElement('size')
                width = root.createElement('width')
                width_text = root.createTextNode(str(img.size[0]))
                width.appendChild(width_text)
                size.appendChild(width)
                height = root.createElement('height')
                height_text = root.createTextNode(str(img.size[1]))
                height.appendChild(height_text)
                size.appendChild(height)
                depth = root.createElement('depth')
                depth_text = root.createTextNode('3')
                depth.appendChild(depth_text)
                size.appendChild(depth)
                ann.appendChild(size)

                m_width = root.createElement('meters_per_pixel')
                m_width_text = root.createTextNode('%.3f' % (float(pm['meter_width']) / img.size[0]))
                m_width.appendChild(m_width_text)
                ann.appendChild(m_width)

                ll_bounds = root.createElement('lat_lon_bounds')
                lat1 = root.createElement('lat1')
                lat1_text = root.createTextNode(pm['lat1'])
                lat1.appendChild(lat1_text)
                ll_bounds.appendChild(lat1)

                lon1 = root.createElement('lon1')
                lon1_text = root.createTextNode(pm['lon1'])
                lon1.appendChild(lon1_text)
                ll_bounds.appendChild(lon1)

                lat2 = root.createElement('lat2')
                lat2_text = root.createTextNode(pm['lat2'])
                lat2.appendChild(lat2_text)
                ll_bounds.appendChild(lat2)

                lon2 = root.createElement('lon2')
                lon2_text = root.createTextNode(pm['lon2'])
                lon2.appendChild(lon2_text)
                ll_bounds.appendChild(lon2)
                ann.appendChild(ll_bounds)

                if int(set_dir[-1:]) <= 3:
                    lbl_f = os.path.join(ship_labels_dir, set_dir, batch[:-4], pm['name'] + '.txt')
                    with open(lbl_f, 'r') as lbl_f:
                        for line in lbl_f:
                            vessel_type, x_c, y_c, w, h = list(map(float, line.split(' ')))
                            x_c = int(x_c * img.size[0])
                            y_c = int(y_c * img.size[1])
                            w = int(w * img.size[0])
                            h = int(h * img.size[1])
                            x1 = x_c - int(w/2)
                            y1 = y_c - int(h/2)
                            x2 = x_c + int(w/2)
                            y2 = y_c + int(h/2)

                            obj = root.createElement('object')
                            v_type = root.createElement('class')
                            v_type_text = root.createTextNode(str(int(vessel_type)))
                            v_type.appendChild(v_type_text)
                            obj.appendChild(v_type)

                            bbox = root.createElement('bndbox')
                            xmin = root.createElement('xmin')
                            xmin_text = root.createTextNode(str(x1))
                            xmin.appendChild(xmin_text)
                            bbox.appendChild(xmin)

                            ymin = root.createElement('ymin')
                            ymin_text = root.createTextNode(str(y1))
                            ymin.appendChild(ymin_text)
                            bbox.appendChild(ymin)

                            xmax = root.createElement('xmax')
                            xmax_text = root.createTextNode(str(x2))
                            xmax.appendChild(xmax_text)
                            bbox.appendChild(xmax)

                            ymax = root.createElement('ymax')
                            ymax_text = root.createTextNode(str(y2))
                            ymax.appendChild(ymax_text)
                            bbox.appendChild(ymax)
                            obj.appendChild(bbox)
                            ann.appendChild(obj)

                xml_str = root.toprettyxml(indent="\t")
                with open(os.path.join(annotations_dir, set_dir, batch[:-4], pm['name'] + '.xml'), "w") as f:
                    f.write(xml_str)
