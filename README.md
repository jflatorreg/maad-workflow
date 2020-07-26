# MAAD Workflow
MAAD's Work Flux are the steps (scripts) to follow to perform automatic classification of sounds of interest. These steps are based on the **MAAD** method.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install MAAD library.

```bash
pip install upgrade -i https://test.pypi.org/simple/scikit-maad
```

## Usage

```python
from maad.rois import find_rois_cwt
from maad import sound

s, fs = sound.load('./templates/BETA-_20161006_002000_section.wav') # loads a signal of interest as a floating point time series s.
                                                                    # get the sample rate of the signal fs.
                                               
rois = find_rois_cwt(s, fs, 
                     flims = (1000,4000),             # frequency limits of the regions of interest.
                     tlen = 0.3,                      # time length of the regions of interest.
                     th = 0.0001,                     # threshold to filter the signal
                     display=True, figsize=(13,6))
```

![Alt text](Example/Example.png?raw=true "Title")

## Workflow

* [1_detection_and_feature_extraction.ipynb](1_detection_and_feature_extraction.ipynb) Detection of regions of interest and extraction of characteristics from regions of interest. 


* [2_data_processing.ipynb](2_data_processing.ipynb) The regions of interest are labeled and the training and test sets are created.


* [3_models_training.ipynb](3_models_training.ipynb) Supervised learning models are trained. It's found the hyperparameters of the models: logistic regression, support vector machine, random forest, multilayer perceptron.


* [4_data_analysis.ipynb](4_data_analysis.ipynb) Visualize and analyze data.


* [5_data_augmentation.ipynb](5_data_augmentation.ipynb) Generate the files with background sound.


* [6_data_augmentation_features_extraction.ipynb](6_data_augmentation_features_extraction.ipynb) Extract the characteristics of the augmented data.


* [7_data_augmentation_data_processing.ipynb](7_data_augmentation_data_processing.ipynb) The regions of interest of augmented data are labeled and the training and test sets are created.


* [8_data_augmentation_models_training.ipynb](8_data_augmentation_models_training.ipynb) Supervised learning models are trained.


* [9_data_analysis_data_augmentation.ipynb](9_data_analysis_data_augmentation.ipynb) Visualize and analyze data.


## Author

Developed by **Juan Felipe Latorre Gil**, you can contact me by [email](mailto:jflatorre@unal.edu.co): <jflatorreg@unal.edu.co> or [git](https://github.com/jflatorreg).

This work was based on the preliminary work of the **MAAD** method carried out by [Juan Sebastián Ulloa Chacón](mailto:julloa@humboldt.org.co) (<julloa@humboldt.org.co>). Thanks for the help.

## License
[MIT](https://choosealicense.com/licenses/mit/)
