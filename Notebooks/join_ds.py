import os
import shutil
import yaml
from tqdm import tqdm

def get_new_list():
    # This function should return the new list of labels
    select_labels = [ "Warning-STOP", "Warning-Yield", "Indication-One-way traffic", "Indication-Pedestrian crossing", "Prohibition-No entry", "Prohibition-Speed limit 100 km/h", "Prohibition-Speed limit 120 km/h", "Prohibition-Speed limit 20 km/h", "Prohibition-Speed limit 30 km/h", "Prohibition-Speed limit 40 km/h", "Prohibition-Speed limit 50 km/h", "Prohibition-Speed limit 60 km/h", "Warning-Roadworks", "Prohibition-Speed limit 80 km/h"]
    select_labels_2 = ['Warning-Yield', 'Dead end', 'Indication-Pedestrian crossing', 'Indication-One-way traffic', 'End of priority road', "Warning-Roadworks", 'End of construction works', 'End of play street', 'Warning-STOP', 'Play street', 'Prohibition-No entry', 'Road closure', 'Red traffic light', 'Green traffic light']  # class names
    new_labels = []
    for i in select_labels:
        if i not in new_labels:
            new_labels.append(i)
    for i in select_labels_2:
        if i not in new_labels:
            new_labels.append(i)
    return new_labels

def get_old_to_new_mapping_dataset1():
    # This function should return a dictionary mapping old class IDs to new class IDs for dataset1
    old_labels = [ "Warning-STOP", "Warning-Yield", "Warning-Danger", "Warning-Landslide", "Warning-Intersection where you have priority", "Warning-Wild animals crossing", "Warning-Children crossing", "Warning-Pedestrian crossing", "Warning-Speed bumps", "Warning-Roundabout", "Warning-Dual carriageway", "Warning-Slippery road", "Warning-Priority road", "Warning-Series of bends", "Warning-Roadworks", "Warning-Right turn", "Warning-Left turn", "Red light", "Green light", "Indication-Highway", "Indication-One-way traffic", "Indication-Parking", "Indication-Pedestrian crossing", "Indication-Bus stop", "Prohibition-No stopping and parking", "Prohibition-No overtaking", "Prohibition-No U-turn", "Prohibition-No right turn", "Prohibition-No left turn", "Prohibition-No entry", "Prohibition-No parking", "Prohibition-Speed limit 100 km/h", "Prohibition-Speed limit 120 km/h", "Prohibition-Speed limit 20 km/h", "Prohibition-Speed limit 30 km/h", "Prohibition-Speed limit 40 km/h", "Prohibition-Speed limit 50 km/h", "Prohibition-Speed limit 60 km/h", "Prohibition-Speed limit 80 km/h", "Mandatory-Continue right", "Mandatory-Continue left", "Mandatory-Continue straight", "Mandatory-Continue straight or turn right", "Mandatory-Continue straight or turn left", "Mandatory-Turn right", "Mandatory-Turn left", "Mandatory-Turn around the roundabout"]

    new_labels = get_new_list()
    return {old_labels.index(label): new_labels.index(label) for label in old_labels if label in new_labels}

def get_old_to_new_mapping_dataset2():
    # This function should return a dictionary mapping old class IDs to new class IDs for dataset1
    old_labels = ['Speed limit 50', 'Bike path', 'Dangerous intersection', 'Priority at next intersection', 'Passage restriction', 'Uneven road', 'No parking', 'Priority road', 'Parking', 'Parking cars only', 'Bikes crossing road', 'No overtaking', 'Children crossing road', 'Bike path crossing', 'No stopping between 1 and 15', 'No stopping', 'Warning-Yield', 'Parking disability', 'Parking trucks only', 'Dead end', 'Speed bump', 'Danger', 'Indication-Pedestrian crossing', 'Farm animals on road', 'Multiple steep turns starting turn right', 'Priority at this bottleneck', 'Road max width 2.5m', 'Indication-One-way traffic', 'Steep turn left', 'Obstacle! road narrows on both sides', 'End of priority road', "Warning-Roadworks", 'End of construction works', 'Steep turn right', 'Parking bus only', 'Parking on sidewalk', 'No turn right', 'Obstacle! road narrows on left', 'Left only', 'Obstacle! road narrows on right', 'End of play street', 'Warning-STOP', 'Straight or turn left', 'Multiple steep turns starting turn left', 'Bikes and pedestrians only', 'Maximum weight 5.5T', 'Bike path and pedestrians separated', 'No bikes', 'No turn left', 'No delivery cars', 'Parking on right', 'No stopping between 16 and 31', 'Play street', 'Roundabout', 'Prohibition-No entry', 'Straight only', 'Road closure', 'Slippy road', 'Traffic light ahead', 'Max height 3.5m', 'Oncoming traffic has priority', 'Red traffic light', 'Yellow traffic light', 'Green traffic light']  # class names

    new_labels = get_new_list()
    return {old_labels.index(label): new_labels.index(label) for label in old_labels if label in new_labels}

def merge_datasets(dataset1_train, dataset1_test, dataset1_val, dataset2_train, dataset2_test, dataset2_val, output_train, output_test, output_val):
    if not os.path.exists(output_train):
        os.makedirs(output_train)
    if not os.path.exists(output_test):
        os.makedirs(output_test)
    if not os.path.exists(output_val):
        os.makedirs(output_val)
    
    # Merge train sets
    merge_split(os.path.join(dataset1_train, 'images'), os.path.join(dataset1_train, 'labels'),
                os.path.join(dataset2_train, 'images'), os.path.join(dataset2_train, 'labels'),
                os.path.join(output_train, 'images'), os.path.join(output_train, 'labels'),
                filter_annotations_dataset1, filter_annotations_dataset2)
    
    # Merge test sets
    merge_split(os.path.join(dataset1_test, 'images'), os.path.join(dataset1_test, 'labels'),
                os.path.join(dataset2_test, 'images'), os.path.join(dataset2_test, 'labels'),
                os.path.join(output_test, 'images'), os.path.join(output_test, 'labels'),
                filter_annotations_dataset1, filter_annotations_dataset2)
    
    # Merge validation sets
    merge_split(os.path.join(dataset1_val, 'images'), os.path.join(dataset1_val, 'labels'),
                os.path.join(dataset2_val, 'images'), os.path.join(dataset2_val, 'labels'),
                os.path.join(output_val, 'images'), os.path.join(output_val, 'labels'),
                filter_annotations_dataset1, filter_annotations_dataset2)

def merge_split(images1_path, labels1_path, images2_path, labels2_path, output_images_path, output_labels_path, filter_func1, filter_func2):
    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)
    if not os.path.exists(output_labels_path):
        os.makedirs(output_labels_path)
    
    # Copy images and labels from the first dataset
    copy_files(images1_path, labels1_path, output_images_path, output_labels_path, filter_func1, get_old_to_new_mapping_dataset1())
    
    # Copy images and labels from the second dataset
    copy_files(images2_path, labels2_path, output_images_path, output_labels_path, filter_func2, get_old_to_new_mapping_dataset2())

def copy_files(images_path, labels_path, output_images_path, output_labels_path,
               filter_func, mapping):
    for file_name in os.listdir(labels_path):
        if file_name.endswith('.txt'):
            try:
                with open(os.path.join(labels_path, file_name), 'r') as file:
                    annotations = file.readlines()
            except FileNotFoundError:
                print(f"Warning: File {file_name} not found. Skipping.")
                continue
            
            good_annotations = filter_func(annotations)
            if good_annotations:
                # Copy the image only if there are good annotations
                image_file = file_name.replace('.txt', '.jpg')  # Assuming images are in .jpg format
                try:
                    shutil.copy(os.path.join(images_path, image_file), output_images_path)
                except FileNotFoundError:
                    print(f"Warning: Image file {image_file} not found. Skipping.")
                    continue
                
                with open(os.path.join(output_labels_path, file_name), 'w') as file:
                    updated_annotations = update_annotations(good_annotations, mapping)
                    file.writelines(updated_annotations)

def filter_annotations_dataset1(annotations):
    # Implement your logic to filter good annotations for dataset1
    return [ann for ann in annotations if is_good_annotation_dataset1(ann)]

def filter_annotations_dataset2(annotations):
    # Implement your logic to filter good annotations for dataset2
    return [ann for ann in annotations if is_good_annotation_dataset2(ann)]

def is_good_annotation_dataset1(annotation):
    # Define criteria for good annotations in dataset1
    parts = annotation.split()
    if len(parts) == 0:
        return False
    class_id = int(parts[0])
    labels = [ "Warning-STOP", "Warning-Yield", "Warning-Danger", "Warning-Landslide", "Warning-Intersection where you have priority", "Warning-Wild animals crossing", "Warning-Children crossing", "Warning-Pedestrian crossing", "Warning-Speed bumps", "Warning-Roundabout", "Warning-Dual carriageway", "Warning-Slippery road", "Warning-Priority road", "Warning-Series of bends", "Warning-Roadworks", "Warning-Right turn", "Warning-Left turn", "Red light", "Green light", "Indication-Highway", "Indication-One-way traffic", "Indication-Parking", "Indication-Pedestrian crossing", "Indication-Bus stop", "Prohibition-No stopping and parking", "Prohibition-No overtaking", "Prohibition-No U-turn", "Prohibition-No right turn", "Prohibition-No left turn", "Prohibition-No entry", "Prohibition-No parking", "Prohibition-Speed limit 100 km/h", "Prohibition-Speed limit 120 km/h", "Prohibition-Speed limit 20 km/h", "Prohibition-Speed limit 30 km/h", "Prohibition-Speed limit 40 km/h", "Prohibition-Speed limit 50 km/h", "Prohibition-Speed limit 60 km/h", "Prohibition-Speed limit 80 km/h", "Mandatory-Continue right", "Mandatory-Continue left", "Mandatory-Continue straight", "Mandatory-Continue straight or turn right", "Mandatory-Continue straight or turn left", "Mandatory-Turn right", "Mandatory-Turn left", "Mandatory-Turn around the roundabout"]
    select_labels = [ "Warning-STOP", "Warning-Yield", "Indication-One-way traffic", "Indication-Pedestrian crossing", "Prohibition-No entry", "Prohibition-Speed limit 100 km/h", "Prohibition-Speed limit 120 km/h", "Prohibition-Speed limit 20 km/h", "Prohibition-Speed limit 30 km/h", "Prohibition-Speed limit 40 km/h", "Prohibition-Speed limit 50 km/h", "Prohibition-Speed limit 60 km/h", "Warning-Roadworks", "Prohibition-Speed limit 80 km/h"]
    return labels[class_id] in select_labels

def is_good_annotation_dataset2(annotation):
    # Define criteria for good annotations in dataset2
    parts = annotation.split()
    if len(parts) == 0:
        return False
    class_id = int(parts[0])
    labels = ['Speed limit 50', 'Bike path', 'Dangerous intersection', 'Priority at next intersection', 'Passage restriction', 'Uneven road', 'No parking', 'Priority road', 'Parking', 'Parking cars only', 'Bikes crossing road', 'No overtaking', 'Children crossing road', 'Bike path crossing', 'No stopping between 1 and 15', 'No stopping', 'Warning-Yield', 'Parking disability', 'Parking trucks only', 'Dead end', 'Speed bump', 'Danger', 'Indication-Pedestrian crossing', 'Farm animals on road', 'Multiple steep turns starting turn right', 'Priority at this bottleneck', 'Road max width 2.5m', 'Indication-One-way traffic', 'Steep turn left', 'Obstacle! road narrows on both sides', 'End of priority road', "Warning-Roadworks", 'End of construction works', 'Steep turn right', 'Parking bus only', 'Parking on sidewalk', 'No turn right', 'Obstacle! road narrows on left', 'Left only', 'Obstacle! road narrows on right', 'End of play street', 'Warning-STOP', 'Straight or turn left', 'Multiple steep turns starting turn left', 'Bikes and pedestrians only', 'Maximum weight 5.5T', 'Bike path and pedestrians separated', 'No bikes', 'No turn left', 'No delivery cars', 'Parking on right', 'No stopping between 16 and 31', 'Play street', 'Roundabout', 'Prohibition-No entry', 'Straight only', 'Road closure', 'Slippy road', 'Traffic light ahead', 'Max height 3.5m', 'Oncoming traffic has priority', 'Red traffic light', 'Yellow traffic light', 'Green traffic light']  # class names
    select_labels = ['Warning-Yield', 'Dead end', 'Indication-Pedestrian crossing', 'Indication-One-way traffic', 'End of priority road', "Warning-Roadworks", 'End of construction works', 'End of play street', 'Warning-STOP', 'Play street', 'Prohibition-No entry', 'Road closure', 'Red traffic light', 'Green traffic light']  # class names
    return labels[class_id] in select_labels

def update_annotations(annotations, mapping):
    updated_annotations = []
    for ann in annotations:
        parts = ann.split()
        class_id = int(parts[0])
        if class_id in mapping:
            new_class_id = mapping[class_id]
            parts[0] = str(new_class_id)
            updated_annotations.append(' '.join(parts) + '\n')
        else:
            print(f"Warning: class_id {class_id} is not in the mapping")
    return updated_annotations

def generate_data_yaml(output_dir):
    new_labels = get_new_list()
    data = {
        'train': os.path.join(output_dir, 'train/images'),
        'val': os.path.join(output_dir, 'val/images'),
        'test': os.path.join(output_dir, 'test/images'),
        'nc': len(new_labels),
        'names': new_labels
    }
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as file:
        yaml.dump(data, file)

# Paths to your datasets
dataset1_train = 'train'
dataset1_test = 'test'
dataset1_val = 'valid'
dataset2_train = 'datasets/combined_yolo_annotations/train'
dataset2_test = 'datasets/combined_yolo_annotations/test'
dataset2_val = 'datasets/combined_yolo_annotations/val'
output_train = 'New_DS/train'
output_test = 'New_DS/test'
output_val = 'New_DS/val'
output_dir = 'New_DS'

merge_datasets(dataset1_train, dataset1_test, dataset1_val, dataset2_train, dataset2_test, dataset2_val, output_train, output_test, output_val)
generate_data_yaml(output_dir)