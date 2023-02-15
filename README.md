# BATMAN
## Overview
Welcome to the BATMAN repository for the code used in BATMAN: a Brain-like Approach for Tracking Maritime Activity and Nuance.
## Getting Started
Clone the repository and navigate into root directory of the project:
```
cd batman
```
We recommended the use of a virtual environment to utilize this code.
Use the following commands to create a virtual environment through python.
```
python3 -m virtualenv batmanenv
batmanenv\Scripts\activate
```
Or you could create a Conda environment
```
conda create --new batman python=3.8
```
Once you have a virtual environment, please use pip to install the requirements,
as well as install this repository as a package.
```
pip install -r requirements.txt -e .
```

Now code can be imported from the `batman` package directly, for example: `from batman.data_utils.api_utils import get_weather_data`

Once the repository has been set up, there is an IPython notebook called `tutorial_notebook.ipynb` which is a good place
to try out the code available and become familiar with the repository.

## File structure of `batman/` package
`data_utils` directory contains python functions and classes for reading, manipulating, and comparing AIS data.
It also contains some functions related to STK ship files, for generating new versions of the MDoDS dataset (see link in
references).

`behavioral_classifier` contains the code used to train and and classifiy the behavior of ships.
The training and testing data is generated from code in the `AIS_processing` folder.

`AIS_processing` contains all files regarding ship pairing, AIS dataset generation, and other AIS needs.

## Citations
Please consider citing our open-access paper with the following bibtex citation, if you find our repo useful.
```bibtex
@inproceedings{jones2023batman,
  title={BATMAN: a Brain-like Approach for Tracking Maritime Activity and Nuance},
  author={Jones, Alexander and Koehler, Stephan and Jerge, Michael and Graves, Mitchell and King, Bayley and Dalrymple, Richard and Freese, Cody and Von Albade, James},
  year={2023},
  pages={TBD},
  journal={Sensors},
  doi={TBD},
  volume={TBD},
  article_number={TBD},
  url={TBD},
  pubmedid={TBD},
  issn={TBD},
  doi={TBD},
  abstract={As commercial geospatial intelligence data becomes more widely available, algorithms using artificial intelligence need created to analyze it. 
  	Maritime traffic is annually increasing in vol-ume, and with it the number of anomalous events that would be of interest to law enforcement agencies, governments, and militaries.
    This work proposes a data fusion pipeline that uses a mixture of artificial intelligence and traditional algorithms to identify ships at sea and classify their behavior.
    A fusion process of visual spectrum satellite imagery and automatic identifica-tion system (AIS) data was used to identify ships.
    Further, this fused data was further integrated with additional information about the ship’s environment to help classify ship behavior to a meaningful degree.
    This type of contextual information included things such as exclusive eco-nomic zone boundaries, locations of pipelines and undersea cables, and the local weather.
    Be-haviors such as illegal fishing, transshipment, and spoofing are identified by the framework using freely or cheaply accessible data from places such as Google Earth, the United States Coast Guard, etc.
    The pipeline is the first of its kind to go beyond the typical ship identification pro-cess to help aid analysts in identifying tangible behaviors and reducing the human workload.
  }
  
}
```
