"""
Generates the roc curves and the true_positive_rate and the false_positive_rate values for different datasets
"""

from tqdm import trange
import os
import keras as keras
from keras.models import load_model
import os, random
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

class configuration():
  """
  A simple class to manage the configuration
  """

  model_path = "test_model\\siamese_network_02-04-2020_15-11-49_thres_[0.52510107]_.h5"
  
  no_of_samples = 500  # total number of samples to consider
  
  image_size = None  #will be set automatically
  
  trials = 1  # total number of trials to perform



def get_images(img0_paths, img1_paths):
  """ 
  A simple function to open images and append it in the array. 
  Returns the array of images
  """

  img0_arr = []
  img1_arr = []

  def return_image(img_path):
    """ Open the image and return the image """

    img = Image.open(img_path).convert('RGB')
    img = img.resize((configuration.image_size,configuration.image_size))
    img = (np.asarray(img, dtype=np.float32))/255.0
    
    return img
  
  print("Accessing the images")
  for i in trange(len(img0_paths)):  
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
  for i in trange(configuration.no_of_samples):
    
    if i < configuration.no_of_samples // 2:
      choice = 0
    else:
      choice = 1
    
    current_folder_path1 = random.choice(folders)
    path1 = os.path.join(current_folder_path1, random.choice(os.listdir(current_folder_path1)))
    image1_paths.append(path1)
    
    if choice == 0:
      path2 = os.path.join(current_folder_path1, random.choice(os.listdir(current_folder_path1)))
      image2_paths.append(path2)
    
    elif choice == 1:
      while True:
        current_folder_path2 = random.choice(folders)
        if current_folder_path1 != current_folder_path2:
          break
      path2 = os.path.join(current_folder_path2, random.choice(os.listdir(current_folder_path2)))
      image2_paths.append(path2)
    
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
  for i in trange(configuration.no_of_samples):
    
    current_folder_path1 = random.choice(folders)
    current_subfolder = random.choice(os.listdir(current_folder_path1))
    current_subfolder_path = os.path.join(current_folder_path1, current_subfolder)
    img1_path = os.path.join( current_subfolder_path, random.choice(os.listdir(current_subfolder_path)))
    
    if i < int(configuration.no_of_samples/2):
      choice = 0
    else:
      choice = 1

    if choice == 0:
      current_subfolder = random.choice(os.listdir(current_folder_path1))
      current_subfolder_path = os.path.join(current_folder_path1, current_subfolder)
      img2_path = os.path.join( current_subfolder_path, random.choice(os.listdir(current_subfolder_path)))
    
    elif choice == 1:
      while True:
        current_folder_path2 = random.choice(folders)
        if current_folder_path1 != current_folder_path2:
          break
      current_subfolder = random.choice(os.listdir(current_folder_path2))
      current_subfolder_path = os.path.join(current_folder_path2, current_subfolder)
      img2_path = os.path.join( current_subfolder_path, random.choice(os.listdir(current_subfolder_path)))
    
    image1_paths.append(img1_path)
    image2_paths.append(img2_path)
    tags.append(choice)

  combined_zip = list(zip(image1_paths, image2_paths, tags))
  random.shuffle(combined_zip)
  image1_paths[:], image2_paths[:], tags[:] = zip(*combined_zip)

  return image1_paths, image2_paths, tags





def batch_statistics(path1, path2, label):
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
              print(f"false: {path1[i]} | {path2[i]} | {label[i]}")
      else:
          if label[i] == 1:
              count_diff += 1
          else:
              false_count +=1
              print(f"false: {path1[i]} | {path2[i]} | {label[i]}")
  total = count_same + count_diff + false_count
  print("total pairs: {} | same pairs {} ({:0.2f}%) | diff pairs {} ({:0.2f}%)".format((count_same + count_diff + false_count),count_same, ((count_same*100)/total), count_diff, ((count_diff*100)/total)))
  
  
  return None



def get_metrices(tags, predictions, threshold):
  """
  calculates the values of confision matrix
  Returns the true_positive_rate, false_positive_rate and accuracy
  """
  true_positive = false_positive = false_negative = true_negative = 0
  for i in range(len(tags)):
    if predictions[i] <= threshold:
      if tags[i] == 0:
        true_positive +=1
      elif tags[i] == 1:
        false_positive +=1
    elif predictions[i] > threshold:
      if tags[i] == 0:
        false_negative +=1
      elif tags[i] == 1:
        true_negative += 1
  
  total_true_positive = true_positive + false_negative
  total_true_negative = false_positive + true_negative
  predicted_positive = true_positive + false_positive
  predicted_negative = false_negative + true_negative

  true_positive_rate = true_positive/total_true_positive
  false_positive_rate = false_positive/total_true_negative
  accuracy = (true_positive + true_negative)/len(tags)

  return true_positive_rate, false_positive_rate, accuracy



def get_tpr_fpr_auc(dataset_name, testing_dataset_path, dataset_type):
  """
  writes tpr and fpr in the file and plots the roc curve
  """

  mean_tpr= []
  mean_fpr = []
  mean_acc = []

  # open file for writing the results 
  file = open(f"{dataset_name}_roc.txt", 'w', encoding='utf-8')
  for trial in range(configuration.trials):
    print("="*60)
    print("Run: ", trial+1)
    folders = get_folder_list(testing_dataset_path)
    if dataset_type == 2:
      img0_paths, img1_paths, tags = get_image_pair_paths_type2(folders)
    else:
      img0_paths, img1_paths, tags = get_image_pair_paths_type1(folders)

    # print batch statistics
    print("#"*60)
    print("Batch Statistics:")
    batch_statistics(img0_paths, img1_paths, tags)
    print("#"*60)

    img0, img1 = get_images(img0_paths, img1_paths)
    print("Predicting:")
    predictions = loaded_model.predict([img0,img1])
    
    # as the model has two outputs 
    # 1st output is auxiliary output
    # 2nd output is the final output of the model
    # the predictions list contains two values for each pair. Each value corresponding to the auxiliary output and final output respectively.
    # So considering only the final output
    predictions = predictions[1]


    tpr_arr = []
    fpr_arr =[]
    accuracy_arr = []
    
    # create a threshold array consisting values --> ( 0, 0.01, 0.02, 0.03, ... ..., 0.98, 0.99, 1.00 )
    thresholds = list(np.arange(0.0,1.01,0.01))

    # for each threshold value get the tpr and fpr
    for i in thresholds:
      tpr, fpr, accuracy = get_metrices(tags, predictions, i)
      tpr_arr.append(tpr)
      fpr_arr.append(fpr)
      accuracy_arr.append(accuracy)

    # calculate the auc score
    auc_score = (np.trapz(tpr_arr,fpr_arr))
    print("auc score: ",auc_score)

    # find the best threshold
    best_threshold = thresholds[np.argmax(accuracy_arr)]
    print("Best threshold: ",best_threshold)

    # calculate mean tpr and mean for each threshold for multiple trials
    if len(mean_tpr) == 0:
      mean_tpr = tpr_arr
      mean_fpr = fpr_arr
      mean_acc = accuracy_arr
    else:
      mean_tpr = [(mean_tpr[i] + tpr_arr[i]) for i in range(len(tpr_arr))]
      mean_fpr = [(mean_fpr[i] + fpr_arr[i]) for i in range(len(fpr_arr))]
      mean_acc = [(mean_acc[i] + accuracy_arr[i]) for i in range(len(accuracy_arr))]


    print("="*60)
    print("\n")

  mean_tpr = [x/configuration.trials for x in mean_tpr]
  mean_fpr = [x/configuration.trials for x in mean_fpr]
  mean_acc = [x/configuration.trials for x in mean_acc]
  mean_auc = (np.trapz(mean_tpr,mean_fpr))
  mean_threshold = thresholds[np.argmax(mean_acc)]
  mean_auc = np.mean(mean_auc)

  # write the mean tpr and fpr values in the file
  file.writelines("="*100)
  file.writelines(f"\nmean auc {mean_auc} | mean thres {mean_threshold}\n")
  file.writelines("="*100)
  file.writelines("\n")

  for i in range(len(mean_acc)):
    file.writelines(f"\nthres: {thresholds[i]:0.2f} | tpr: {mean_tpr[i]:0.2f} | fpr: {mean_fpr[i]:0.2f}")
    print(f"thres: {thresholds[i]:0.2f} | tpr: {mean_tpr[i]:0.2f} | fpr: {mean_fpr[i]:0.2f}")
  print(f"mean auc {mean_auc} | mean thres {mean_threshold}")
  
  file.close()

  # plot the roc curves for the dataset
  plt.subplots(1, figsize=(10,10))
  plt.xlim(0,1)
  plt.ylim(0,1)
  plt.title('Receiver Operating Characteristic')
  plt.plot(mean_fpr, mean_tpr, label = f"{dataset_name} - auc: {mean_auc}")
  plt.plot([0,1],[0,1],'--',color = 'orange')
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.legend(loc = 'lower right')
  plt.savefig(f"{dataset_name}_roc.png")
  plt.show()

  return mean_tpr, mean_fpr, mean_acc, mean_auc, mean_threshold



# load the saved model 
loaded_model = load_model(configuration.model_path, compile = False)
loaded_model.summary()  # print the summary of the model
configuration.image_size = loaded_model.input_shape[0][1]   # set the image size


# call the function get_tpr_fpr_auc(dataset_name, path_of_dataset, dataset_type)
lfw_tpr, lfw_fpr, lfw_acc, lfw_auc, lfw_threshold = get_tpr_fpr_auc("lfw", "datasets\\lfw", 1)

scface_tpr, scface_fpr, scface_acc, scface_auc, scface_threshold = get_tpr_fpr_auc("scface", "datasets\\scface", 1)

yt_faces_tpr, yt_faces_fpr, yt_faces_acc, yt_faces_auc, yt_faces_threshold = get_tpr_fpr_auc("youtube_faces", "datasets\\aligned_images_DB", 2)

# for plotting the roc curves together
plt.subplots(1, figsize=(10,10))
plt.xlim(0,1)
plt.ylim(0,1)
plt.title('Receiver Operating Characteristic')
plt.plot(scface_fpr, scface_tpr, color = 'purple', label = f"SCface - auc: {scface_auc}")
plt.plot(yt_faces_fpr, yt_faces_tpr, color = 'blue', label = f"Youtube Faces - auc: {yt_faces_auc}")
plt.plot(lfw_fpr, lfw_tpr, color = 'magenta', label = f"LFW - auc: {lfw_auc}")
plt.plot([0,1],[0,1],'--',color = 'orange')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.savefig("roc_combined.png")
plt.show()