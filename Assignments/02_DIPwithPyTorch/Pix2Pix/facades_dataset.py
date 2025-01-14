import torch
from torch.utils.data import Dataset
import cv2
import os


# current_work_dir = os.path.dirname(__file__)  # 当前文件所在的目录

class FacadesDataset(Dataset):
    def __init__(self, list_file):
        """
        Args:
            list_file (string): Path to the txt file with image filenames.
        """
        # list_file=os.path.join(current_work_dir, list_file)
        # os.chdir("./Assignments/02_DIPwithPyTorch/Pix2Pix")
        # print(os.getcwd())
        # print(list_file)

        # Read the list of image filenames
        with open(list_file, 'r') as file:
            self.image_filenames = [line.strip() for line in file]
        
    def __len__(self):
        # Return the total number of images
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Get the image filename
        img_name = self.image_filenames[idx]
        # img_name=os.path.join(current_work_dir, img_name)

        img_color_semantic = cv2.imread(img_name)
        # print(img_name)
        # Convert the image to a PyTorch tensor
        image = torch.from_numpy(img_color_semantic).permute(2, 0, 1).float()/255.0 * 2.0 -1.0
        image_rgb = image[:, :, :256]
        image_semantic = image[:, :, 256:]
        return image_rgb, image_semantic
    
# train_dataset = FacadesDataset(list_file='train_list.txt')

# print(os.getcwd())

# img1, label1 = train_dataset[123] 

# print(img1)

    