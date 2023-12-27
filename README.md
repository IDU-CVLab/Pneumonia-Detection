# Dataset:
* COV19-CT Database was used for COVID and healthy cases. The first 51 CT scans in each class were used for training, and full validation set was used for validation.
* For Pneumonia cases, a publicly available dataset at [https://data.mendeley.com/datasets/3y55vgckg6/1](https://data.mendeley.com/datasets/3y55vgckg6/1) was used; 4 CT scans (test-59, 61, 62, and 63) were used for training, and 2 CT scans (test-64 and 65) were used for validation.  
* The dataset used in this method is a combination of the above-mentioned.


# Methodology:
*	Augmentation were deployed on the Pneumonia class aiming at classes balance. Augmentation focused on zooming, flipping, and rotation; 2000 images from the existing 199 slices of Common Pneumonia were generated. using 'augmentator' at https://github.com/mdbloice/Augmentor.  
* Image processing the CNN model training used for Pnumonia cases is similar to the onle we applied during the first run of the MIA-COVID19 Workshop, 2021 [here](https://github.com/IDU-CVLab/COV19D). Thus this work allow for extending our solution from just COVID-19 detection to Pnumonia cases.  
* For full details of the method and the results, refer to our paper in the [IMCIDU Congress Proceedings](https://imcidu.idu.edu.tr/index.php/kongre-kitaplari/?lang=en), 2023. (Paper ID 442).  
  
# Citation
This study is included in the university's 2023 booklets, following the [5th International Medical Congress of Izmir Democracy UNiversity (IMCIDU) with ID-442/OP](https://imcidu.idu.edu.tr/index.php/kongre-kitaplari/?lang=en).

