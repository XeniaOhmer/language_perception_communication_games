# language_perception_communication_games

This code accompanies the arXiv preprint: 
Xenia Ohmer, Michael Marino, Michael Franke, Peter König. (2021). Mutual influence between language and perception in multi-agent communication games. *arXiv:2112.14518*


We use a setup where two agents, a sender and a receiver, play a reference game. The agents have a vision module which 
is initialized with a pretrained CNN, and a language module which is initialized randomly. The agents are then trained on 
a reference game. Different training setups are used (see Figure 3 in the manuscript).

* Influence of perception on language / evolutionary analysis: vision modules are fixed, only the language modules are trained 
* Influence of language on perception: vision and language modules are trained simultaneously. 
    * language learning: the sender weights (and hence the messages) are completely fixed, only the receiver is trained 
    * language emergence: both agents are trained 

The README explains 1) how to get the 3D shapes data set, 2) how to pretrain the CNNs, 3) how to train the agents on the reference game, and 4) where to find the results and analyses reported in the paper. The pretrained CNNs that we use in the paper are included in 'share/'. If you would like to work with these models, and do not intend to train your own models, you can skip steps 1) and 2). 



## Data 

We use the 3d shapes data set (Burgess & Kim, 2018), which you can download using

    wget https://storage.cloud.google.com/3d-shapes/3dshapes.h5

or from [here](https://console.cloud.google.com/storage/browser/3d-shapes;tab=objects?prefix=&forceOnObjectsSortingFiltering=false]).

Put the data into a folder for the CNN pretraining to work. If you do not intend to train your own CNNs but only want to use the models we are using, you do not need to download the original data. The pretrained models can be found in 'share/'. 



## CNN pretraining 


Provided the file '3dshapes.h5' is in the main directory with the train_cnn.py file, the default training experiments 
(with dual-trait conditions set to equal weighting) can be run by calling train_cnn.py without arguments, i.e. by running

python train_cnn.py

or by providing the path to the 3dshapes file via command line as

python train_cnn.py -d PATH_TO_DATAFILE

in a terminal with the appropriate python packages installed. By default, the script will train a CNN consisting of 2
convolutional layers with 32 channels each and two fully connected layers with 16 nodes each. Models are built using
the GenericNet class in the ./models folder. A CNN with different parameters can be trained by defining a dictionary
following the format in utils.train.load_default_model_params(). The default parameters dictionary is defined as -  

        model_params['conv_depths'] = [32,32]
        model_params['fc_depths'] = [16,16]
        model_params['conv_pool'] = [True, False]

Each entry in 'conv_depths' and 'fc_depths' specifies the channel dimension, with one layer resulting for each entry.
The 'conv_pool' entry specifies whether or not to pool after each convolutional layer. For further customization with
the GenericNet class see 'models/genericnet.py'. Alternatively, any arbitrary keras model can be used by replacing the
'model' variable, provided input and output dimensions are consistent. In order to change the dataset, replace the
load_data() function in 'train_cnn.py' and update the 'input_shape' and 'num_classes' variables. The 'sf_list' variable
defines the smoothing parameters that will be applied. The outer training loop controls which traits will be enforced,
resulting in one training run per trait per smoothing factor.

Below is a list of other parameters of interest that may be set via command line

    -s      smoothing parameter - [0, 1) : amount of value removed from true class and distributed to related classes.
                                           Setting this to 0 corresponds to the normal one-hot case

    -t      trait parameter - {scale, color, shape, all, color-shape, color-size, shape-size} 
                                          : perceptual trait to be enforced by label smoothing, where all means color-shape-size

    -dw     dualweight parameter - (0, 1) : how to weigh traits in the case of dual trait enforcement, with the value
                                            specified applied to the first trait and 1 minus that value applied to the 
                                            second

For example, if one were to use the command

python train_cnn.py -d data.h5 -s 0.6 -t color-size -dw 0.4 -p params.pkl

This would result in a training run using the data file "data.h5" in the working directory, a smoothing factor
of 0.6, enforcing the perceptual traits color and size with a weighting of 0.4 for the color attribute and 0.6
for the size (or scale) attribute, with a CNN model specified according to a dictionary loaded from a file
named params.pkl in the working directory (which must have keys corresponding to the keyword arguments in the
GenericNet class, as mentioned above).

The trained CNNs, that we are using, can be found in 'share/'. 

The functions for analyzing the CNN similarities are under 'utils/vision_analysis.py' and the plots in 'vision_analysis.ipynb'.


## Training on the communication game 

The agents can be trained on the communication game using 'train_refgame.py'. There are several command line arguments you can use, help can be found in the file. 

#### How results are saved 

Results are saved in a folder 'results/' in the main directory. The name under which results are stored are specified by the command line arguments *--mode* giving the subfolder and *--name* giving the subsubfolder. E.g. *--mode language_emergence_basic --run default* will store the results for the first run under 'results/language_emergence_basic/default0/' and so on. Another subfolder is created automatically depending on vocab size and message length, such that the results are finally stored in folders like 'results/language_emergence_basic/default0/vs4_ml3/'. In each of these folders, we save the parameters of the run, a log file monitoring the training progress, the training and test rewards, sender and receiver loss, message length, and potentially classification accuracies. 


#### Training for the different setups 

For training, download and unzip '3Dshapes_subset.zip' from the associated OSF project and move the folder into the main directory. It contains the train / test split that is also used for the CNN pretraining. 

For all simulations you need to specify the following command line arguments: 
* --sim_sender (sender vision bias)
* --sim_receiver (receiver vision bias)
* --sf_sender (smoothing factor of sender CNN)
* --sf_receiver (smoothing factor of receiver CNN)
* --mode (subfolder in the results)
* --run (subsubfolder in the results)

The biases that can be provided are *default*, *color*, *scale*, *shape*, *all*, as well as *color-scale*, *color-shape*, *scale-shape* for the mixed bias conditions. The smoothing factors are reported as for example *0-6* for sigma=0.6. Depending on the training setup you need to specify additional arguments. 


**1. Influence of language on perception / evolutionary analysis**

*Example*:\
python train_refgame.py --sim_sender color --sim_receiver default --sf_sender 0-6 --sf_receiver 0-0 --mode language_emergence_basic --run color_default

All other parameters have the right default values. In the control simulations where only two out of three attributes are relevant, the irrelevant attribute is provided with *--irrelevant_attribute* (color, scale, or shape).


**2. Influence of perception on language - language learning**

*Example*:\
python train_refgame.py --sim_sender all --sim_receiver default --sf_sender 0-8 --sf_receiver 0-0 --train_vision_receiver True 
--classification True --load_sender_epoch 150 --mode language_learning_train_vision --run all_default --n_epochs 25

You need to specify a bias and smoothing factor for an already trained sender. The sender's weights are then loaded from 'results/language_emergence_basic/'. You also need to indicate that the vision module of the receiver is trained (*--train_vision_receiver True*) and that it is also trained on the classification task (*--classification True*). *--load_sender_epoch* indicates which training epoch to load from the pretrained sender, and implicitly that the sender is fixed (final epoch 150). 


**3. Influence of perception on language - language emergence**

*Example*\
python train_refgame.py --sim_sender default --sim_receiver default --sf_sender 0-0 --sf_receiver 0-0 --train_vision_sender True --train_vision_receiver True --classification True --mode language_emergence_train_vision --run default

Again, training of the vision modules and training on the classification task need to be indicated. 




## Analyses and results 

All results can be found in the 'results.zip' folder of the associated OSF project. In case you would like to run the analyses, download and unzip them, then move them into the main directory of the repository. (We separated them because they add up to ~3GB.)

The different subfolder contain the following results: 
* language_emergence_basic: influence of perception on language / evolutionary analysis
* language_learning_train_vision: influence of language on perception, language learning scenario
* language_emergence train_vision: influence of language on perception, language emergence scenario
* language_emergence_color_irrelevant, language_emergence_shape_irrelevant, language_emergence_scale_irrelevant: control simulations (see evolutionary analysis results)

These results are analyzed using the jupyter notebooks in the main directors:
* label_smoothing_analysis: analyze the effects of label smoothing on the CNN representations for different conditions
* training_analysis: analyze training and test rewards
* three_way_entropy_analysis: show that an entropy analysis of the sender side (input objects - messages) is symmetric to an entropy analysis of the receiver side (messages - selected objects), which is why we only analyze the sender. 
* language_analysis: analyze effectiveness scores 
* vision_analysis: analyze RSA scores (i.e. biases) of the vision modules
* evolutionary_analysis: generate and analyze the reward matrices for the evolutionary analysis
* mixed_models_grid_seach: evaluate the grid search over mixed-bias models, and identify the best ones for the evolutionary control simulations

All reported values, bootstrapped confidence intervals and plots in the paper can be found in these notebooks. 

