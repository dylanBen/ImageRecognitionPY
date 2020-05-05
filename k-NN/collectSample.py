import shutil
import os

sample_size = 50
train_folder = '../data/train/'
focus_train_folder = '../data/focus_train/'
focus_normal = os.path.join(focus_train_folder, 'NORMAL')
focus_pneumonia = os.path.join(focus_train_folder, 'PNEUMONIA')

def createFocusFolder():
    if os.path.exists(focus_train_folder):
        shutil.rmtree(focus_train_folder)
    os.mkdir(focus_train_folder)
    os.mkdir(focus_normal)
    os.mkdir(focus_pneumonia)

def getSample():
    normal_folder = os.path.join(train_folder, 'NORMAL')
    normal_filenames = os.listdir(normal_folder)
    pneumonia_folder = os.path.join(train_folder, 'PNEUMONIA')
    pneumonia_filenames = os.listdir(pneumonia_folder)
    bacteria_counter = 0
    virus_counter = 0

    for index, filename in enumerate(normal_filenames):
        if index == sample_size:
            break
        shutil.copy(os.path.join(normal_folder, filename), focus_normal)

    for filename in pneumonia_filenames:
        if bacteria_counter >= sample_size/2 and virus_counter >= sample_size/2:
            break
        if 'bacteria' in filename and bacteria_counter < sample_size/2:
            shutil.copy(os.path.join(pneumonia_folder, filename), focus_pneumonia)
            bacteria_counter += 1
        elif 'virus' in filename and virus_counter < sample_size/2:
            shutil.copy(os.path.join(pneumonia_folder, filename), focus_pneumonia)
            virus_counter += 1

def respectStructure():
    return os.path.exists(os.path.join(train_folder, 'NORMAL')) and os.path.exists(os.path.join(train_folder, 'PNEUMONIA'))

def launchSampling(newSample):
    if(newSample):
        sample_size = newSample
    if respectStructure():
        print("Create destination folder (focus train)")
        createFocusFolder()
        print("collect sample from sources (train folder)")
        getSample()
        print("Sampling Finish")
    else:
        print("Please respect the structure for extraction")

launchSampling(sample_size)
