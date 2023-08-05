# Database:
* Pneumonia CT images are at [https://data.mendeley.com/datasets/3y55vgckg6/1](https://data.mendeley.com/datasets/3y55vgckg6/1)
* The dataset used for training and testing the method is is a combination of the COV19-CT-DB and the Pnomonia CT images.
* 4 CT scans (test-59, 61, 62, and 63) of the Common Pnomonia were used for training.
* 2 CT scans (test-64 and 65) were used for validation.

# Methodology:
*	Augmentation were deployed on the Pneumonia class aiming at balancing classes in the dataset. 
* Augmentation focused on zooming, flipping, and rotation. 2000 images from the existing 199 slices of Common Pneumonia were addeeed. The augmentation library is at https://github.com/mdbloice/Augmentor was used to perform the task.
* The first 51 CT scans were used from COV19-CT-DB database for training; in each covid and non-covid cases (CT's from 0 to 50).
* The method is our COV19D first run of the competition 2021, [here](https://github.com/IDU-CVLab/COV19D)
