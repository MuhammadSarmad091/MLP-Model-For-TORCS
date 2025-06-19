**TORCS Racing MLP Controller**

This repository contains the code and resources for a multilayer perceptron (MLP) model developed to control a racing car in the TORCS simulator. The model is trained on a large dataset (\~390,000 samples) to predict key control outputs—steering, acceleration, brake, gear, and clutch—based on game telemetry.

---

## Project Overview

* **Objective**: Build and train an MLP that accurately maps TORCS game state inputs to control outputs, enabling autonomous driving within the TORCS environment.
* **Data**: \~390k rows of gameplay telemetry, including speed, track position, sensor readings, and corresponding control labels.
* **Model**: Fully connected (dense) neural network using TensorFlow/Keras.
* **Pipeline**:

  1. **Preprocessing**: Clean and normalize raw telemetry data.
  2. **Training**: Configure, train, and evaluate the MLP on the preprocessed dataset.
  3. **Deployment**: Integrate the trained model into a driver interface to interact with TORCS.

---

## Repository Structure

```text
├── Driver/
  #All files are related to the communication with the game i.e. message exchanges b/w the game and the trained model
│
├── Model/
│   ├── preprocessing_and_training.ipynb  # Jupyter notebook with preprocessing & training code
│   ├── trained_model.h5                  # Saved Keras MLP model
│   └── scaler_input.pkl                  # Pickled scaler for input normalization
│
└── README.md                # This file
```

---

## Driver Folder

Contains all scripts required to bridge between the TORCS simulator and the trained MLP model:

You can run the driver code by 
   ```bash
   python pyclient.py
   ```
---

## Model Folder

Holds the data preprocessing, model training code, and the resulting artifacts:

* **preprocessing\_and\_training.ipynb**: Jupyter notebook that:

  1. Loads raw telemetry data.
  2. Applies cleaning steps (missing value handling, filtering).
  3. Scales inputs using a `StandardScaler`.
  4. Defines an MLP architecture with multiple dense layers to predict steering angle, acceleration, brake pressure, gear selection, and clutch engagement.
  5. Trains the model, tracks metrics, and saves the best weights.

* **trained\_model.h5**: The final Keras model file, ready for inference within the driver.

* **scaler\_input.pkl**: Pickled `StandardScaler` object used to normalize incoming telemetry before model prediction.

---

## Usage

1. **Install Dependencies**:

   ```bash
   pip install tensorflow pandas scikit-learn
   ```
2. **Start TORCS** (ensure the simulator is running and configured for network control).
3. **Run Driver**:
   Navigate to Driver folder and run the following command by opening the terminal in that folder:
   ```bash
   python pyclient.py
   ```

The driver will continuously receive telemetry from the TORCS server, normalize inputs, predict control commands, and send them back to the game in real time.

---

## License

This project is released under the MIT License. Feel free to use, modify, and distribute.
