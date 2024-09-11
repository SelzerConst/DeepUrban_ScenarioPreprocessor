import os
import shutil
from tqdm import tqdm
import time
import logging
import json
import numpy as np
from lxml import etree
import xml.etree.ElementTree as ET

            
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import dill
from scipy.interpolate import interp1d
import lanelet2

from pathlib import Path

from crdesigner.common.config.general_config import general_config
from crdesigner.common.config.lanelet2_config import lanelet2_config
from crdesigner.common.config.opendrive_config import open_drive_config
from crdesigner.map_conversion.map_conversion_interface import opendrive_to_lanelet

from ds_utils.generic import load_json, load_frames, load_frame, Object

def get_recording_name(scenario_name):
    parts = scenario_name.split('_')
    return parts[1] 


def read_split(src_dir):
    # read in all .json files in the src_dir/split
    split_files = [f for f in os.listdir(os.path.join(src_dir, '../split')) if f.endswith('.json')]
    
    # Split file name is split_<location>_<split>.json
    # safe the data in a dictionary with the location as key
    split = {}
    for file in split_files:
        location = file.split('_')[1]
        split[location] = json.load(open(os.path.join(src_dir, '../split', file)))
    
    # create a dict of dict of dict with the structure split[location][recoding_name][scenario_name] = train/val/test
    sorted_split = {}
    
    for location in split.keys():
        sorted_split[location] = {}
        for recording in split[location].keys():
            unique_recording_name_part = get_recording_name(recording)
            if unique_recording_name_part not in sorted_split[location]:
                sorted_split[location][unique_recording_name_part] = {}
            sorted_split[location][unique_recording_name_part][recording] = split[location][recording]
    
    return sorted_split, split_files


# Given a recording name and the path to the data directory, return a list of frames for the recording.
def get_frames(recording_name: str, data_dir: str) -> list:
    """
    Args:
        recording_name: Name of the recording
        data_dir: Path to the data directory
        
    Returns:
        frames: List of frames (frame_id, file_name, objects)
    """
    # Replace with the actual path to the annotation directory
    annotations_dir = os.path.join(data_dir, 'annotations', recording_name)

    # Load annotation data
    annotations_meta = load_json(os.path.join(annotations_dir, 'annotations_meta.json'))
    frames = load_frames(annotations_dir, annotations_meta['annotations'])

    return frames

def get_frame_data(location, unique_recording, src_dir):
    # get all recording with the unique_recording part
    recording = [f for f in os.listdir(os.path.join(src_dir, location, 'release', 'data', 'annotations')) if unique_recording in f]
    return get_frames(recording[0], os.path.join(src_dir, location, 'release', 'data'))


# Get all frames from list of frame_ids
def get_frames_from_frame_ids(frames: list, frame_ids: list) -> list:
    """
    Args:
        frames: List of frames
        frame_ids: List of frame_ids
        
    Returns:
        frames: List of frames with frame_ids in frame_ids
    """
    return [frame for frame in frames if frame['frame_id'] in frame_ids]




def flatten_frames(frames: list, scenario_name: str) -> dict:
    """
    Args:
        frames: List of frames
        
    Returns:
        frames: with object structure flattened
    """  
    
    # Create a dictionary to store the flattened frames
    flattened_frames = {}
    
    # Iterate through the frames
    for frame in frames:
        # Get the frame_id
        frame_id = frame['frame_id']
        
        # Add the frame_id to the dictionary
        flattened_frames[frame_id] = {}
        
        # Iterate through the objects in the frame
        for obj in frame['objects']:
            # Get the track_id
            track_id = obj.track_id
            
            # Add the track_id to the dictionary
            flattened_frames[frame_id][track_id] = {}
            
            # Add the translation to the dictionary
            flattened_frames[frame_id][track_id]['translation'] = obj.translation
            
            # Add the rotation to the dictionary
            flattened_frames[frame_id][track_id]['rotation'] = obj.rotation
            
            # Add the dimension to the dictionary
            flattened_frames[frame_id][track_id]['dimension'] = obj.dimension
            
            # Add the velocity to the dictionary
            flattened_frames[frame_id][track_id]['velocity'] = obj.velocity
            
            # Add the angular_velocity to the dictionary
            flattened_frames[frame_id][track_id]['angular_velocity'] = obj.angular_velocity
            
            # Add the acceleration to the dictionary
            flattened_frames[frame_id][track_id]['acceleration'] = obj.acceleration
            
            # Add the category_id to the dictionary
            flattened_frames[frame_id][track_id]['category_id'] = obj.category_id
            
            # Add the ego_vehicle to the dictionary
            if track_id == int(scenario_name.split('_')[2]):
                flattened_frames[frame_id][track_id]['ego_vehicle'] = 1
            else:
                flattened_frames[frame_id][track_id]['ego_vehicle'] = 0
                
    return flattened_frames


def get_scenario_data(frames, scenario_name):
    
    # create a list of frame_ids from the scenario_name
    parts = scenario_name.split('_')
    start_frame_id = int(parts[-2])
    end_frame_id = int(parts[-1])
    frame_ids = list(range(start_frame_id, end_frame_id + 1, 2))
    
    # get the frames from the frame_ids
    scenario_frames = get_frames_from_frame_ids(frames, frame_ids)
    
    # flatten the frames and 
    flattend_scenario_frames = flatten_frames(scenario_frames, scenario_name)
    
    return flattend_scenario_frames
 
 
 
 
##########################################################
########### Function to interpolate scenarios ############
##########################################################


def upscale_data(track_data, step_size=4):
    interpolated_track_data = {}

    for track_id, frames in track_data.items():
        # Extract the existing frame indices and sort them
        frame_ids = sorted(frames.keys())
        
        # Skip tracks with less than 2 frames
        if len(frame_ids) < 2:
            continue
        
        # Initialize the interpolated data structure for this track
        interpolated_track_data[track_id] = {}

        # Store constant attributes (assuming it's consistent across all frames)
        category_id = frames[frame_ids[0]]['category_id']
        ego_vehicle = frames[frame_ids[0]]['ego_vehicle']
        dimension = frames[frame_ids[0]]['dimension']

        # Interpolate between each pair of consecutive frames
        for i in range(len(frame_ids) - 1):
            start_frame = frame_ids[i]
            end_frame = frame_ids[i + 1]
            num_steps = end_frame - start_frame
            
            # Prepare new frame indices between the current and next frame (interpolating by 4)
            new_frame_ids = np.linspace(start_frame, end_frame, num=num_steps * step_size, endpoint=False)
            
            # Iterate through attributes that need interpolation
            attributes = ['translation', 'velocity', 'acceleration'] # , 'rotation', 'angular_velocity']
            
            for attr in attributes:
                # Extract data for this attribute at the current and next frame
                attr_start_data = np.array(frames[start_frame][attr])
                attr_end_data = np.array(frames[end_frame][attr])
                
                # Create interpolation function for this segment
                f_interp = interp1d([start_frame, end_frame], [attr_start_data, attr_end_data], axis=0, kind='linear')
                
                # Apply interpolation function to the new frame indices
                interpolated_values = f_interp(new_frame_ids)
                
                # Store interpolated values in the new dictionary
                for j, frame_id in enumerate(new_frame_ids):
                    if frame_id not in interpolated_track_data[track_id]:
                        interpolated_track_data[track_id][frame_id] = {}
                    interpolated_track_data[track_id][frame_id][attr] = interpolated_values[j].tolist()
                
            # Duplicate the category_id for each new frame
            for frame_id in new_frame_ids:
                if frame_id not in interpolated_track_data[track_id]:
                    interpolated_track_data[track_id][frame_id] = {}
                interpolated_track_data[track_id][frame_id]['category_id'] = category_id
                interpolated_track_data[track_id][frame_id]['ego_vehicle'] = ego_vehicle
                interpolated_track_data[track_id][frame_id]['dimension'] = dimension

        # Ensure that the original frames are also included
        for frame_id in frame_ids:
            if frame_id not in interpolated_track_data[track_id]:
                interpolated_track_data[track_id][frame_id] = frames[frame_id]
    
    adjusted_interpolated_track_data = {}
    
    # Adjust the frame indices to be intergers
    for track_id, frames in interpolated_track_data.items():
        adjusted_interpolated_track_data[track_id] = {}
        for frame_id, frame_data in frames.items():
            adjusted_interpolated_track_data[track_id][int(frame_id*step_size)] = frame_data
            

    return adjusted_interpolated_track_data

##########################################################
########### Function to downsample scenarios #############
##########################################################

def downsample_data(tracks, downsample_factor=5):
    # Filtered dictionary to be returned
    downsampled_tracks = {}

    # Iterate over each track_id in the original dictionary
    for track_id, frames in tracks.items():
        # Initialize a sub-dictionary for the filtered frames
        downsampled_tracks[track_id] = {}

        # Sort the frame ids to ensure consistent results
        sorted_frame_ids = sorted(frames.keys(), key=int)

        # Iterate over sorted frame ids and take every element which is divisible by 5
        for frame_id in sorted_frame_ids:
            if frame_id % downsample_factor == 0:
                downsampled_tracks[track_id][int(frame_id/downsample_factor)] = frames[frame_id]
    


    return downsampled_tracks

##########################################################
######## Function to switch the focus of the data ########
##########################################################

def swap_dictionary_nesting(source_data):
    """
    Swaps the first and second levels of nesting in a nested dictionary.
    
    Parameters:
    - source_data: Dict, the original nested dictionary to be modified.
    
    Returns:
    - Dict, a new dictionary with swapped nesting levels.
    """
    swapped_data = {}

    # Iterate over all items in the source data
    for first_key, first_level_data in source_data.items():
        for second_key, item_data in first_level_data.items():
            if second_key not in swapped_data:
                swapped_data[second_key] = {}
            swapped_data[second_key][first_key] = item_data

    return swapped_data



##########################################################
############# Function to save scenarios #################
##########################################################

def save_scenarios(output_dir: str, scenarios: dict, location: str, interpolate: bool = True) -> None:
    """
    Args:
        scenario_dir: Directory to save the scenarios
        scenarios: Dictionary with scenarios
        location: Location of the dataset
        interpolate: Whether to interpolate the scenarios
    """
    
    # Create the location directory
    scenario_dir = os.path.join(output_dir, location)
    # create the location directory if it does not exist
    os.makedirs(scenario_dir, exist_ok=True)
    
    # track the progress of the saving process with location where it is saved
    pbar = tqdm(total=len(scenarios), desc='Saving scenarios')
    
    # Iterate through the scenarios 
    for scenario_id, scenario_data in scenarios.items():
        
        # Modify the scenario data to have frame_id starting from 0 to n
        modified_scenario_data = {}
        for i, (frame_id, frame_data) in enumerate(scenario_data.items()):
            modified_frame_id = i
            modified_scenario_data[modified_frame_id] = frame_data
        
        
        
        # frames should be adjustable to the wanted scene_ts (instead of 0.08 sec it should be 0.1 sec)        
        if interpolate:
            start_time = time.time()
            
            # swap the focus of the data
            agent_focused_dict = swap_dictionary_nesting(modified_scenario_data)
            

            # upsample the agent focused dict to an frame_rate that is reachable by 0.08 and 0.1 sec            
            # find reachable frame_rate by 0.08 and 0.1 sec
            current_frame_rate = 0.08
            wanted_frame_rate = 0.1
            upsample_frame_rate = 0.02
            upsample_ratio = int(current_frame_rate/upsample_frame_rate)
            downsample_ratio = int(wanted_frame_rate/upsample_frame_rate)
            
            
            upsampled_agent_focused_dict = upscale_data(agent_focused_dict, upsample_ratio)
            
            interpolated_scenario_dict = downsample_data(upsampled_agent_focused_dict, downsample_ratio)
            
            
            # swap the focus of the data back
            interpolated_scenario_data = swap_dictionary_nesting(interpolated_scenario_dict)
            
            modified_scenario_data = interpolated_scenario_data
            
            end_time = time.time()
            logging.debug(f'Interpolated scenarios in {end_time - start_time} seconds')
            
        
        
        ##########################################################
        ############# Save scenarios as dill file ################
        ##########################################################
        
        # Create a dill file for the scenario
        scenario_file = f'{scenario_id}.dill'
        scenario_file_path = os.path.join(scenario_dir, scenario_file)
        
        # Convert the scenario data to a Pandas DataFrame
        df = pd.DataFrame.from_dict(modified_scenario_data)
        
        # Save the scenario as dill file
        df.to_pickle(scenario_file_path)
            
        # Update the progress bar
        pbar.update(1) 
        
    # # Close the progress bar
    # pbar.close()
    
        
    
    
def preprocess_data(src_dir, output_dir, interpolate=True):
    
    split, split_files = read_split(src_dir)
    # set up logging for terminal output
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    
    # create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # loop over each location in split
    for location in split.keys():
        
        logging.info(f'Processing location {location}')
        
        # loop over recodring name part in split[location]
        for unique_recording in split[location].keys():
            scenarios = {}
            
            frames = get_frame_data(location, unique_recording, src_dir)
            for scenario_name in split[location][unique_recording].keys():
            
                scenarios[scenario_name] = get_scenario_data(frames, scenario_name)
            
            save_scenarios(output_dir, scenarios, location, interpolate)
                
                
def preprocess_maps(src_dir, output_dir, interpolate):
    
    # create the output directory if it does not exist in the output_dir parent directory
    
    output_dir_maps = os.path.join(os.path.dirname(output_dir), 'maps')
    
    
    # create the meta file if it does not exist
    if not os.path.exists(os.path.join(output_dir_maps,'location_meta.json')):
        with open(os.path.join(output_dir_maps, 'location_meta.json'), 'w') as f:
            f.write('{}')
            
    if interpolate:
        with open(os.path.join(output_dir_maps, 'location_meta.json'), 'r+') as f:
            data = json.load(f)
            data['dt'] = 0.1
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
    else:
        with open(os.path.join(output_dir_maps, 'location_meta.json'), 'r+') as f:
            data = json.load(f)
            data['dt'] = 0.08
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
    
    # get all locations in the src_dir
    for location in os.listdir(src_dir):
        if not os.path.isdir(os.path.join(src_dir, location)):
            continue
    
        # Maps are in src_dir/<loctions>/release/data/map/ *.xodr
        map_path = os.path.join(src_dir, location, 'release', 'data', 'map')
        map_files = [f for f in os.listdir(map_path) if f.endswith('.xodr')]
        # data_meta.json in the data folder
        data_meta = load_json(os.path.join(src_dir, location, 'release', 'data', 'data_meta.json'))
        map_meta = data_meta['locations'][0]
        latitude = map_meta['latitude']
        longitude = map_meta['longitude']
        
        input_file = Path(os.path.join(map_path, map_files[0]))
        output_file = Path(os.path.join(output_dir_maps, location + '.osm'))
        
        # Check if the output file exists is not create it empty
        if not os.path.exists(output_file):
            with open(output_file, 'w') as f:
                f.write('')
                
        
        # Convert the xodr file to osm file
        
        # define configs
        general_config_ = general_config
        opendrive_config_ = open_drive_config
        lanelet2_config_ = lanelet2_config
        
        proj_string = f'+proj=tmerc +lat_0={latitude} +lon_0={longitude} +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
        
        add_proj_string_to_opendrive_config(input_file, proj_string)
        
        general_config_.affiliation = 'Munich University of Applied Sciences'
        general_config_.author = 'Constantin Selzer'
        general_config_.map_name = location
        general_config_.proj_string_cr = proj_string
        
        
        opendrive_config_.proj_string_odr = proj_string
        opendrive_config_.filter_types.append('intersection')
        opendrive_config_.lanelet_types_backwards_compatible = True
        
        lanelet2_config_.supported_lanelet2_subtypes.append('intersection')
        lanelet2_config_.use_local_coordinates = True
        

        # conversion
        opendrive_to_lanelet(
            input_file=input_file,
            output_file=str(output_file),
            odr_config=opendrive_config_,
            general_config=general_config_,
            lanelet2_config=lanelet2_config_,
            )
        
        # Remove relations with right_of_way subtype from the OSM file
        
        # Load the OSM file as an XML tree
        tree = ET.parse(output_file)
        root = tree.getroot()

        # Remove all relations with right_of_way subtype
        relations_to_remove = []
        for relation in root.findall('relation'):
            for tag in relation.findall('tag'):
                if tag.attrib.get('k') == 'subtype' and tag.attrib.get('v') == 'right_of_way':
                    relations_to_remove.append(relation)

        # Remove the identified relations from the root
        for relation in relations_to_remove:
            root.remove(relation)

        # Save the updated XML back to an OSM file
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        
        
        
        # add map longitude and latitude in a json file called map_meta.json in the same directory as the output map
        map_meta = {
            'latitude': latitude,
            'longitude': longitude
        }
        with open(os.path.join(output_dir_maps, 'location_meta.json'), 'r+') as f:
            data = json.load(f)
            data[location] = map_meta
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
        
        print("loaded")
            
def add_proj_string_to_opendrive_config(opendrive_file, proj_string):
    """
    Add the projection string to the opendrive config.
    """
    
    with open(opendrive_file, 'r+') as f:
        root_node = etree.parse(f)
        
        for elem in root_node.getiterator():
            if elem.tag == 'header':
                for sub_elem in elem.iter():
                    if sub_elem.tag == 'geoReference':
                        sub_elem.text = proj_string
                        break
                break
        
        # Move the file pointer to the beginning before writing
        f.seek(0)

        # Write the updated XML back to the file
        f.write(etree.tostring(root_node, pretty_print=True, encoding='UTF-8', xml_declaration=True).decode('UTF-8'))

        # Truncate the file to ensure no old data remains
        f.truncate()
        