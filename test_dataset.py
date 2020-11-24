"""
This file can be used to test the siamese network trained using the face_train.ipynb
"""

from tqdm import trange
import os
import keras as keras
import os, random
from PIL import Image
from matplotlib import pyplot
import numpy as np
from datetime import datetime
import pytz
from pytz import timezone
format = "%d-%m-%Y_%H-%M-%S"
from keras.models import load_model


class configuration():

  """ A simple class to manage the configuration"""

  dataset_type = 1   #  1 or 2  ---> more about dataset_type can be found in the README.txt file
  testing_dataset_path = "datasets\\scface"
  model_path = "test_model\\siamese_network_02-04-2020_15-11-49_thres_[0.52510107]_.h5"
  model_threshold = 0.52510107   #provide the treshold of the trained model
  
  split = 10  # total number of splits to make from total samples
  batch_size = None  #will be set automatically in the program -> (total samples/split)
  image_size = None  #will be set automatically
  
  trials = 10  # number of trials to perform
  
  results_file_name = "results_sc.txt"  # name of the file where the results would be stored



def get_images(img0_paths, img1_paths):
  """ A simple function to open images and append it in the array. 
      Returns the array of images"""
  
  img0_arr = []
  img1_arr = []

  def return_image(img_path):
    """ Open the image and return the image """

    img = Image.open(img_path).convert('RGB')
    img = img.resize((configuration.image_size,configuration.image_size))
    img = (np.asarray(img, dtype=np.float32))/255.0
    
    return img
  
  for i in range(len(img0_paths)):  
    img0_arr.append( return_image(img0_paths[i]) )
    img1_arr.append( return_image(img1_paths[i]) )
  
  return img0_arr, img1_arr



def get_folder_list(path):
  """ Returns the list of folder paths in the directory. """
  folders = []

  for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    folders.append(folder_path)
  #print(folders)

  return folders



def get_image_pair_paths_type1(folders):
  """ 
  returns 3 lists.
  1st and 2nd lists contains paths to images in the pair. Thus image paths of ith pair can be accessed by list1[i], list2[i].
  3rd list contains the tags for the respective pair.  0 means the pair is same and 1 means the pair is different.
  """ 
  
  image1_paths = []
  image2_paths = []
  tags = []

  print("Generating image pair paths: ")
  for i in trange(len(folders)):
    current_folder_path = folders[i]
    for current_image_name in os.listdir(current_folder_path):
      choice = random.randint(0,1)
      if choice == 0:
        image1_paths.append(os.path.join(folders[i], current_image_name))
        image2_name = random.choice(os.listdir(folders[i]))
        image2_paths.append(os.path.join(folders[i], image2_name))
        tags.append(choice)
      
      else:
        folder1 = folders[i]
        while True:
          folder2 = random.choice(folders)
          if folder1 != folder2 :
            break
        
        image1_paths.append(os.path.join(folder1, current_image_name))
        image2_name = random.choice(os.listdir(folder2))
        image2_paths.append(os.path.join(folder2, image2_name))
        tags.append(choice)

  combined_zip = list(zip(image1_paths, image2_paths, tags))
  random.shuffle(combined_zip)
  image1_paths[:], image2_paths[:], tags[:] = zip(*combined_zip)

  return image1_paths, image2_paths, tags




def get_image_pair_paths_type2(folders):
  """ 
  returns 3 lists.
  1st and 2nd lists contains paths to images in the pair. Thus image paths of ith pair can be accessed by list1[i], list2[i].
  3rd list contains the tags for the respective pair.  0 means the pair is same and 1 means the pair is different.
  """ 

  image1_paths = []
  image2_paths = []
  tags = []

  print("Generating image pair paths: ")
  for i in range(len(folders)):
    current_folder_path = folders[i]
    current_subfolder = random.choice(os.listdir(current_folder_path))
    current_subfolder_path = os.path.join(current_folder_path, current_subfolder)
    img1_path = os.path.join( current_subfolder_path, random.choice(os.listdir(current_subfolder_path)))
    
    choice = random.randint(0,1)

    if choice == 0:
      current_subfolder = random.choice(os.listdir(current_folder_path))
      current_subfolder_path = os.path.join(current_folder_path, current_subfolder)
      img2_path = os.path.join( current_subfolder_path, random.choice(os.listdir(current_subfolder_path)))
    
    elif choice == 1:
      current_folder_path = random.choice(folders)
      current_subfolder = random.choice(os.listdir(current_folder_path))
      current_subfolder_path = os.path.join(current_folder_path, current_subfolder)
      img2_path = os.path.join( current_subfolder_path, random.choice(os.listdir(current_subfolder_path)))
    
    image1_paths.append(img1_path)
    image2_paths.append(img2_path)
    tags.append(choice)

  combined_zip = list(zip(image1_paths, image2_paths, tags))
  random.shuffle(combined_zip)
  image1_paths[:], image2_paths[:], tags[:] = zip(*combined_zip)

  return image1_paths, image2_paths, tags



def check(path1, path2, label):
  """
  prints the batch statistics about the percent of same pairs and different pairs
  """

  count_same = count_diff = false_count = 0
  for i in range(len(label)):
      if path1[i].split("\\")[2] == path2[i].split("\\")[2]:
          if label[i] == 0:
              count_same += 1
          else:
              false_count += 1
      else:
          if label[i] == 1:
              count_diff += 1
          else:
              false_count +=1
  total = count_same + count_diff + false_count
  print("same pairs {} ({:0.2f}%) | diff pairs {} ({:0.2f}%)".format(count_same, ((count_same*100)/total), count_diff, ((count_diff*100)/total)))
  file.writelines("same pairs {} ({:0.2f}%) | diff pairs {} ({:0.2f}%)\n".format(count_same, ((count_same*100)/total), count_diff, ((count_diff*100)/total)))

  return None



def calc_accuracy(tag, prediction, threshold):
  """
  Returns the accuracy of the predictions and also number of correct and false classifications
  """

  correct_classify = false_classify = 0
  for i in range(0,len(tag)):
    if prediction[i] >= threshold:
      prediction[i] = 1
    else:
      prediction[i] = 0
    if prediction[i] == tag[i]:
      correct_classify = correct_classify + 1
    else:
      false_classify = false_classify + 1

  accuracy = (correct_classify * 100)/(len(tag))

  return accuracy, correct_classify, false_classify


#load the saved model and print the summary.
loaded_model = load_model(configuration.model_path, compile = False)
loaded_model.summary()

#set the image size from the input size of the model
configuration.image_size = loaded_model.input_shape[0][1]

#open file for writing the results
file = open(configuration.results_file_name,'w',encoding='utf-8')

testing_acc = []
for i in range(0,configuration.trials):
  print("="*60)
  file.writelines("="*30)
  file.writelines("\n")
  print("Run {}".format(i))
  file.writelines(f"Run {i}\n")

  # get the paths of all the folders in the directory
  folders = get_folder_list(configuration.testing_dataset_path)
  
  # generate image pair paths and tags based upon the dataset type
  if configuration.dataset_type == 2:
    img0_paths, img1_paths, tags = get_image_pair_paths_type2(folders)
  else:
    img0_paths, img1_paths, tags = get_image_pair_paths_type1(folders)
  
  print("Total samples: ",len(img0_paths))
  configuration.batch_size = int(len(tags)/configuration.split)  # set the batch size
  print("Batch size: ",configuration.batch_size)

  # print the statistics for each batch
  file.writelines("Batch Statistics:\n")
  for i in range(0, len(tags), configuration.batch_size):
      check(img0_paths[i:i+configuration.batch_size], img1_paths[i:i+configuration.batch_size], tags[i:i+configuration.batch_size])

  file.writelines("\n")
  batch_acc = []
  correct_predict = false_predict = 0

  for i in trange(0, len(tags), configuration.batch_size):
      img0, img1 = get_images(img0_paths[i: i+configuration.batch_size], img1_paths[i: i+configuration.batch_size])
      predictions = loaded_model.predict([img0,img1])
      
      # as the model has two outputs 
      # 1st output is auxiliary output
      # 2nd output is the final output of the model
      # the predictions list contains two values for each pair. Each value corresponding to the auxiliary output and final output respectively.
      # So considering only the final output
      predictions = predictions[1]
      
      # calculate accuracy
      accuracy, correct, wrong = calc_accuracy(tags[i: i+configuration.batch_size], predictions, configuration.model_threshold)
      batch_acc.append(accuracy)
      correct_predict += correct
      false_predict += wrong
      

  # write the results to the file
  for i in range(0,len(batch_acc)):
      print("batch {} | size_of_batch {} | acc {}".format(i,configuration.batch_size,batch_acc[i]))
      file.writelines("batch {} | size_of_batch {} | acc {}\n".format(i,configuration.batch_size,batch_acc[i]))

  print(f"\ntotal images {correct_predict + false_predict} | correct predict {correct_predict} | false predict {false_predict}")
  file.writelines(f"\ntotal images {correct_predict + false_predict} | correct predict {correct_predict} | false predict {false_predict}\n")
  
  testing_acc.append(np.mean(batch_acc))
  print("mean acc over {} batches: {}".format(len(batch_acc), np.mean(batch_acc)))
  file.writelines("mean acc over {} batches: {}\n".format(len(batch_acc), np.mean(batch_acc)))
  
  print("="*30)
  file.writelines("="*30)

# write the mean accuracy over the multiple trials
print("="*30)
file.writelines("="*30)
print(f"Average accuracy over {configuration.trials} trials is {np.mean(testing_acc)}")
file.writelines(f"\n\nAverage accuracy over {configuration.trials} trials is {np.mean(testing_acc)}\n\n")
print("="*30)
file.writelines("="*60)

# close the file
file.close()
