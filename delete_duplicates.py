import cv2
import imutils
import os
import datetime
from collections import defaultdict
import time
from concurrent.futures import ThreadPoolExecutor

from imaging_int import *



def remove_duplicates(img_path):
    """
    Removes duplicate images based on comparison scores.

    Args:
        img_path (str): Path to the directory containing the images.

    Returns:
        None
    """
    camera_ids, camera_id_dict = process_stamps(img_path)

    # Resize images and store original sizes
    original_size = resize_images(img_path)

    images_to_remove = []
    
    with ThreadPoolExecutor() as executor:
        
        for camera_id in camera_ids:
            print(camera_id)
            
            image_names = camera_id_dict[camera_id]
            image_dict = {}

            # Sort images by hour and store in a dictionary
            for image in image_names:
                file_time_str = image.split("__")[-1].split(".")[0]
                file_time = datetime.datetime.strptime(file_time_str, "%H_%M_%S")
                hour = file_time.hour
                
                if hour in image_dict:
                    image_dict[hour].append(image)
                else:
                    image_dict[hour] = [image]
            
            # Compare images in one hour with each other 
            # and store names of duplicates with a higher score
            
            for hour, hourly_images in image_dict.items():
                scores = []
                if len(hourly_images) >= 2:
                    
                    for i in range(len(hourly_images) - 1):
                        file1 = hourly_images[i]
                        img1 = cv2.imread(os.path.join(img_path, file1)) if file1 is not None else None
                        
                        if img1 is not None:
                            # img1 = cv2.resize(img1, (640, 480))
                            test_img_1 = preprocess_image_change_detection(img1)
                        else:
                            continue

                        for j in range(i + 1, len(hourly_images)):
                            file2 = hourly_images[j]

                            if (file1 is not None and file2 is not None):
                                img2 = cv2.imread(os.path.join(img_path, file2))

                                if img2 is not None:
                                    # img2 = cv2.resize(img2, (640, 480))
                                    test_img_2 = preprocess_image_change_detection(img2)
                                else:
                                    continue

                                test_img_2 = preprocess_image_change_detection(img2)
                                score, res_cnts, thresh = compare_frames_change_detection(test_img_1, test_img_2, 100)
                                
                                threshold_score = 50000
                                if score < threshold_score:
                                    images_to_remove.append(os.path.join(img_path, file2))
                                    hourly_images[j] = None
                                scores.append(score)

            # Compare images in one hour with images in the previous and next 2 hours
            #and store the names of duplicates with a lower score

            for hour, hourly_images in image_dict.items():
                for near_hour in range(hour - 2, hour + 3):
                    if near_hour == hour:
                        continue
                    if near_hour in image_dict:
                        near_hourly_images = image_dict[near_hour]
                        for i in range(len(hourly_images)):
                            file1 = hourly_images[i]
                            img1 = cv2.imread(os.path.join(img_path, file1)) if file1 is not None else None
                        
                            if img1 is not None:
                                test_img_1 = preprocess_image_change_detection(img1)
                            else:
                                continue

                            for j in range(len(near_hourly_images)):
                                file2 = near_hourly_images[j]

                                if (file1 is not None and file2 is not None):
                                    img2 = cv2.imread(os.path.join(img_path, file2))

                                    if img2 is not None:
                                        test_img_2 = preprocess_image_change_detection(img2)
                                    else:
                                        continue

                                    test_img_2 = preprocess_image_change_detection(img2)
                                    score, res_cnts, thresh = compare_frames_change_detection(test_img_1, test_img_2, 100)
                                    
                                    threshold_score = 1000
                                    if score < threshold_score:
                                        images_to_remove.append(os.path.join(img_path, file2))
                                        near_hourly_images[j] = None
                                    scores.append(score)
        
        #Remove duplicate images
        for image_path in images_to_remove:
            os.remove(image_path)
        
        #Resize images back to original size
        for file in os.listdir(img_path):
            if file.endswith(".png"):
                img = cv2.imread(os.path.join(img_path, file))
                if img is not None:
                    img = cv2.resize(img, original_size[file], interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(img_path, file), img)


def resize_images(img_path):
    """
    Resizes images in the specified directory to a fixed size and saves the original sizes.

    Args:
        img_path (str): Path to the directory containing the images.

    Returns:
        original_size (dict): Dictionary mapping image filenames to their original sizes.
    """
    original_size = {}
    for image in os.listdir(img_path):
        if image.endswith(".png") and image is not None:
            img = cv2.imread(os.path.join(img_path, image))
            if img is not None:
                original_size[image] = (img.shape[1], img.shape[0])
                img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)   
                cv2.imwrite(os.path.join(img_path, image), img) 
    return original_size

def process_stamps(img_path):
    """
    Processes the stamps on the images and renames them if necessary.

    Args:
        img_path (str): Path to the directory containing the images.

    Returns:
        camera_ids (set): Set of unique camera IDs.
        camera_id_dict (defaultdict): Dictionary mapping camera IDs to a list of image filenames.
    """
    camera_ids = set()
    camera_id_dict = defaultdict(list)

    for image in os.listdir(img_path):
        if image.endswith(".png"):
            camera_id = image.split("-")[0] if ("-" in image) else image.split("_")[0]
            camera_ids.add(camera_id)
            
            if "-" in image:
                timestamp = int(image.split("-")[1].split(".")[0])
                timestamp /= 1000 

                dt = datetime.datetime.fromtimestamp(timestamp)
                formatted_datetime = dt.strftime("%Y_%m_%d__%H_%M_%S")
            
                # Rename the file with this format: {camera_id}_{formatted_datetime}.png
                os.rename(os.path.join(img_path, image), os.path.join(img_path, camera_id + "_" + formatted_datetime + ".png"))
                camera_id_dict[camera_id].append(camera_id + "_" + formatted_datetime + ".png")
            else:
                camera_id_dict[camera_id].append(image)

    return camera_ids, camera_id_dict


#Uncomment to run the script

# def main():
#     img_path = r"C:\Users\divis\Downloads\dataset"
#     remove_duplicates(img_path)

# if (__name__ == "__main__"):
#     start_time = time.time()
#     main()
#     print("--- %s seconds ---" % (time.time() - start_time))
    