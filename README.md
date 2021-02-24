## HU-PageScan: a fully convolutional neural network for document page crop

This is an implementation of HU-PageScan on Python 3, Keras and Tensorflow. The training dataset is the Extended Smartdoc Dataset. The proposed model is based on U-net Architecture, which we halved the number of parameters.

### Edge detection Train:

* Architecture used: HU-PageScan
* Input Image Dimensions: 512x512 (gray - 1 channel)
* Dimensions Output Image: 512x512 (binary - 1 channel)
* Pre-trained model provided: pre_trained_model.hdf5


### Instructions for performing the training:

Enter the command below in cmd
```
python "run_hupagescan.py" -munet --train-folder="D:/Ricardo/Datasets/all_tra_files" 
       --validation-folder="D:/Ricardo/Datasets/vall_files" --gpu="1" --bs="4" --train-steps="1" 
       --valid-steps="1" --no-aug --train-samples="18598" --valid-samples="6165" --lr="0.0001"
```
- Set Amount of epochs;
- Set a location where new .hdf5 (trained model) will be saved
- Define the location of results during training (output_refined)



### Instructions for converting .hdf5 to .pb:

Run the script: generator_hdf5_pb_pbtxt.py

Set:
- hdf5 path
- Destination path of .pb and .pbtxt files

### Instructions for performing .pb file inference

Run the script: inferenceTestPB.py
Define:
- The location where the output will be saved
- Location of the images to be inferred
- Locating the .pb
- Name of .pb

#### Extra information

  - Input layer name: input_1:0
  - Output layer name: conv2d_19/Sigmoid

### Citation

Use this BibTeX  to cite this repository:

```
@ARTICLE{
   author = {Ricardo Batista das Neves Junior},
   author = {Estanislau Lima},
   author = {Byron L.D. Bezerra},
   author = {Cleber Zanchettin},
   author = {Alejandro H. Toselli},
   ISSN = {1751-9659},
   language = {English},
   title = {HU-PageScan: a fully convolutional neural network for document page crop},
   journal = {IET Image Processing},
   year = {2020},
   month = {December},
   publisher ={Institution of Engineering and Technology},
   copyright = {© The Institution of Engineering and Technology},
   url = {https://digital-library.theiet.org/content/journals/10.1049/iet-ipr.2020.0532}
}
```