----------------------------------------------------------------------------------------------------------------------------------------------
This README file provides the necessary instructions for running the codes for training and testing the siamese network for facial recognition.
----------------------------------------------------------------------------------------------------------------------------------------------

Note*

1)  The dataset directory having the                            | 2)  The dataset directory having the 
    following structure is considered as  " type = 1 ":         |     following structure is considered as  " type = 2 ":
                                                                |
example:                                                        |     example:
        lfw                                                     |            youtube_faces
        |                                                       |                |
        |---subject_1                                           |                |---subject_1
        |   |---image_1.jpg                                     |                |       |---sub_folder_1
        |   |---image_2.jpg                                     |                |       |       |---image_1.jpg
        |   |---    ...                                         |                |       |       |---image_2.jpg
        |                                                       |                |       |       |---    ...
        |---subject_2                                           |                |       |       |---    ...
        |   |---image_1.jpg                                     |                |       |
        |   |---image_2.jpg                                     |                |       |---sub_folder_2
        |   |---    ...                                         |                |       |       |---image_1.jpg
        |   |---    ...                                         |                |       |       |---image_2.jpg
        |                                                       |                |       |       |---    ...
        ...                                                     |                |       |       |---    ...
        ...                                                     |                |       |   ...
        ...                                                     |                |       |   ...
                                                                |                |
                                                                |                |---subject_2
                                                                |                |       |---sub_folder_1
                                                                |                |       |       |---image_1.jpg
                                                                |                |       |       |---image_2.jpg
                                                                |                |       |       |---    ...
                                                                |                |       |       |---    ...
                                                                |                |       |
                                                                |                |       |---sub_folder_2
                                                                |                |       |       |---image_1.jpg
                                                                |                |       |       |---image_2.jpg
                                                                |                |       |       |---    ...
                                                                |                |       |       |---    ...
                                                                |                |       |   ...
                                                                |                |       |   ...
                                                                |                ...
                                                                |                ...
                                                                |                ...
              

>>  Thus dataset directories having subfolders are considered as type 2.
    Set the dataset_type value in the configuration class accordingly.


----------------------------------------------------------------------------------------------------------------------------------------------

###########################################
#           face_train.ipynb              #
###########################################

To run the above file in Google Colab on gpu, first create a empty directory structure in Google Drive as follows:

Facial recognition
    |
    |---codes                           <--- put the code (face_train.ipynb) in this folder
    |
    |---datasets                        
    |      |
    |      |---sc_dataset
    |      |      |---training          <--- put the folders of the subjects used for training in this folder
    |      |      |---testing           <--- put the folders of the subjects used for testing in this folder
    |      |
    |      |---custom_test_images       <--- put your own test images (if any)
    |
    |---logs                            


Keep the folders of the subjects used for training and testing as shown above in the directory tree.

After training, the "logs" folder will contain all the result files (i.e. train_loss_graph.jpg, accuracy_graph.jpg  etc.).

For GPU configuration
1) Open the file (face_train.ipynb) using Google Colab.
2) From the menu bar go to ( Runtime --> Change runtime type ) and select GPU as Hardware accelerator. 
 
Change the configuration settings in the configuration class in the code.
----------------------------------------------------------------------------------------------------------------------------------------------



----------------------------------------------------------------------------------------------------------------------------------------------
###########################################         ###########################################
#             test_dataset.py             #         #               auc_roc.py                #
###########################################         ###########################################

For the above codes to run create the empty directory structure as follows:

test_datasets           <--- put the codes (test_dataset.py  and  auc_roc.py) in this folder
    |
    |---datasets        <--- put the datasets to test(lfw, scface, youtube_faces) in this folder
    |
    |---test_model      <--- put the trained model in this folder


The requirements for the programs is given in requirements.txt file

Install the requirements by running the following command in cmd:
python -m pip install -r path_to_requirements.txt

*** Installing TensorFlow on the pc requires additional installation of softwares.
    Please refer to the official website for complete installation.
    
    >> https://www.tensorflow.org/install        (for installation of TensorFlow)
    >> https://www.tensorflow.org/install/gpu    (for additional software requirements for TensorFlow)

Change the configuration settings in the configuration class in the code.
----------------------------------------------------------------------------------------------------------------------------------------------
