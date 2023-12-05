# DeepEchoAnnulusDetection

Real-time mitral annulus segmentation from 3D TEE using EfficientNetB0 to predeict Foruier coefficient representation of annulus curve.

### Recommended citation:

 Patrick Carnahan, Apurva Bharucha, Mehdi Eskandari, Daniel Bainbridge, Elvis C. S. Chen, and Terry M. Peters "Real-time mitral annulus segmentation from 4D transesophageal echocardiography using deep learning regression", Proc. SPIE 12464, Medical Imaging 2023: Image Processing, 124641X (3 April 2023); https://doi.org/10.1117/12.2653618 
 

### Usage Examples
#### Data Format
Data is expected to be structured using 3 subfolders, train, test, val. Images are expected to having matching names for imaage and correspodning label, with images ending in US and labels ending in annulus.csv. File format should be .nii or .nii.gz for images. An example of the data structure is here:
```
data
|-- test
|   |-- Patient1-annulus.csv
|   |-- Patient1-US.nii
|   |-- Patient2-annulus.csv
|   |-- Patient2-US.nii
|-- train
|   |-- 001-00-annulus.csv
|   |-- 001-00-US.nii
|   |-- 004-00-annulus.csv
|   |-- 004-00-US.nii
|-- val
|   |-- 001_f1-annulus.csv
|   |-- 001_f1-US.nii.gz
|   |-- 001_f2-annulus.csv
|   |-- 001_f2-US.nii.gz
```

Output from both training and validation will be saved in a directory "runs" created in the data directory alongside train, test and val.

#### Training from epoch 0

```
python TEEAD.py train -data "PATH_TO_DATA"
```
where the data folder corresponds to the top level titled "data" in the example above.

#### Training from a checkpoint

```
python TEEAD.py train -data "PATH_TO_DATA" -load "PATH_TO_CHECKPOINT"
```
where the data folder corresponds to the top level title "data" in the example above and the checkpoint is a .pt file from a prior training run.

#### Running Validation

Validation can be run on either the val images or the test images, with performance metrics being output and model segmentations being saved.

For validation using the val image set:
```
python TEEAD.py validate "PATH_TO_MODEL" -data "PATH_TO_DATA"
```
For validation using the test image set:
```
python TEEAD.py validate "PATH_TO_MODEL" -data "PATH_TO_DATA" -use_test
```
where the data folder corresponds to the top level title "data" in the example above and the model is either a .pt file or a .md file from a prior training run.

#### Running Segmentation

In both training and validation it is expected that images will have corresponding ground truth labels. To use TEEAD in inference mode on images without ground truth labels, use the segment option. The segment option does not require the directory structure from above, and instead expects to be passed the folder directly containing the target images, and will perform inference on all .nii files. Resulting segmentations will be saved in a subdirectory "out" created in the target image directory.

```
python TEEAD.py segment "PATH_TO_MODEL" "PATH_TO_TARGET_IMAGES"
```