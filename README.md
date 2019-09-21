## A Fully Convolutional Neural Network for Document Segmentation in Real-World Complex Backgrounds

This is an implementation of Halved U-net (HU-net) on Python 3, Keras and Tensorflow. The training dataset is the Extended Smartdoc Dataset. The proposed model is based on U-net Architeture, which we halved the number of parameters.

### Edge detection Train:

Architecture used: HU-net
Input Image Dimensions: 512x512 (cinza - 1 channel)
Dimensions Output Image: 512x512 (binarizada - 1 channel)
Pre-trained model provided: pre_trained_model.hdf5


### Instructions for performing the training:

Enter the command below in cmd

python "run_unet_gpu.py" -munet --train-folder="D:/Ricardo/Datasets/all_tra_files" --validation-folder="D:/Ricardo/Datasets/vall_files" --gpu="1" --bs="4" --train-steps="1" --valid-steps="1" --no-aug --train-samples="18598" --valid-samples="6165" --lr="0.0001"

  - Set Amount of epochs;
  - Set location where new .hdf5 (trained model) will be saved
  - Define the location of results during training (output_refined)


### Instructions for converting .hdf5 to .pb:

Run the script: generator_hdf5_pb_pbtxt.py

Set:
     - hdf5 path
     - Destination path of .pb and .pbtxt files

### Instruções para executar a inferência do arquivo .pb

Run the script: inferenceTestPB.py
Define:
     - Location where output will be saved
     - Location of the images to be inferred
     - Locating the .pb
     - Name of .pb

#### Extra information

  - Input layer name: input_1:0
  - Output layer name: conv2d_19/Sigmoid

### Citation

Use this bibtex to cite this repository:

bibtex for cite