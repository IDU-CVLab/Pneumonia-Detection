# Database:

* The dataset used for verifying the COVID-19 methods on Pneumonia cases from CT images is at [https://data.mendeley.com/datasets/3y55vgckg6/1](https://data.mendeley.com/datasets/3y55vgckg6/1)
* The method to be verified is our COV19D first run of the competition 2021, [here](https://github.com/IDU-CVLab/COV19D)
* The dataset used for training the CNN model to include Pnumonia cases is a combination of the COV19-CT-DB first run and the Pnomonia cases mentioned above.
* 4 CT scans (test-59, 61, 62, and 63) of the Common Pnomonia were used for training. Oversampling with 28 duplicate copied was applied on the training set
* 2 CT scans (test-64 and 65) were used for validation.

# Methodology:
*	Augmentation were deployed on the Pneumonia class aiming at balancing classes in the dataset. 
* For data augmentation, and Augmentation method added 2000 images from the existing 199 slices of Common Pneumonia by applying flipping at different degrees and Zooming. The augmentation library at https://github.com/mdbloice/Augmentor was used to perform the task.
* The first 51 CT scans were used from COV19-CT-DB database for training; both in covid and non-covid cases (CT's from 0 to 50 were used).
* The method tried is our COVID first run in 2021.
