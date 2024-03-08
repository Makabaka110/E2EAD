# End-to-End Autonomous Driving

This project aims to create a model for autonomous driving using a deep learning approach. The model is trained with data collected from a driving simulator and can control a virtual car in the simulator environment.

## Installation

**Environment Preparation**

1. Install python3.8 
2. Install all dependencies of the project
```bash 
pip install -r requirements.txt
```
3. Install torch from website https://pytorch.org/, run the command like:
```bash 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


**Simulator preparation:**
This project uses the Udacity Autonomous Driving Simulator, which can be downloaded and installed from the official Udacity repository or website.



### Training Data Acquisition

Our data acquisition process is designed to capture a wide range of driving behaviors and control inputs.

1. **Data Types**

We collect two types of data based on the input method:

- **Keyboard Collected Data**: This data is discrete and discontinuous, reflecting the typical on-off keypresses.
- **Xbox Controller Collected Data**: Using the controller allows for continuous and stable input, better mimicking real-world driving.

2. **Driving Behavior**

The data encompasses various driving behaviors to ensure the model can handle different scenarios:

- **Counterclockwise and Clockwise**: To teach the model about making turns in both directions.
- **Large Angles**: Specifically collected when the vehicle is near the edge of the road, requiring significant steering adjustments to correct the course.
- **Stay in the Middle of the Road**: Data showing stable driving behavior, with the vehicle centered between the road markings.

<img width="594" alt="5e19e224d7bd5058798fdaf65e27815" src="https://github.com/Makabaka110/E2EAD/assets/55959544/d42c74a5-8dd4-4bf7-8294-3e69c184ed05">



### Network Design

Our model is built upon the ResNet50 architecture, leveraging its deep residual learning framework for efficient feature extraction. 

We redesign the FC layer in other to predict steering angles from the input image. The structure is display as the picture below:

![network structure drawio](https://github.com/Makabaka110/E2EAD/assets/55959544/02dab8cb-153d-49b3-8dc1-072f2af412d1)





### Project Hierarchy
- **root folder**: The main project directory.
  - **data**: Directory used to store the collected raw data from the simulator.
  - **dataloader**: Contains scripts for loading and batching the training data.
  - **models**: Stores the model parameter files (e.g., `.pth` files) after training.
  - **network**: Contains the neural network architecture definitions.
  - **utils**: Includes utility scripts for the project, such as data preprocessing, visualization, and Excel file processing.
  - **config.py**: Stores configuration settings for the network, including hyperparameters, data file paths and so on.
  - **drive.py**: Used to interface with the simulator for model evaluation.
  - **train.py**: Script for training the model with the collected data.



### Evaluation in simulator
![](https://github.com/Makabaka110/E2EAD/blob/main/assets/performance.GIF)

### Thesis reference
```
@misc{bojarski2016end,
      title={End to End Learning for Self-Driving Cars}, 
      author={Mariusz Bojarski and Davide Del Testa and Daniel Dworakowski and Bernhard Firner and Beat Flepp and Prasoon Goyal and Lawrence D. Jackel and Mathew Monfort and Urs Muller and Jiakai Zhang and Xin Zhang and Jake Zhao and Karol Zieba},
      year={2016},
      eprint={1604.07316},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```
