# Database:
* COV19-CT Database: first 51 CT scans in the database were used from each class for training, and full validation set was used for validation.
* Pneumonia CT images dataset was also used for Pneumonia cases [https://data.mendeley.com/datasets/3y55vgckg6/1](https://data.mendeley.com/datasets/3y55vgckg6/1) <br/>
  - 4 CT scans (test-59, 61, 62, and 63) were used for training  <br/>
  - 2 CT scans (test-64 and 65) were used for validation  <br/>
* The dataset used for training and validation is a combination of the above mentions.


# Methodology:
*	Augmentation were deployed on the Pneumonia class aiming at balancing classes with the CT images in the COV19CT dtabase. 
* Augmentation focused on zooming, flipping, and rotation; 2000 images from the existing 199 slices of Common Pneumonia were generated. The augmentator at https://github.com/mdbloice/Augmentor was used to perform the task.
* The method trained is the one used during the first run of the MIA-COVID19 Workshop, 2021 [here](https://github.com/IDU-CVLab/COV19D)
