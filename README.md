# TinyML ECG-to-PPG Conversion using LSTM on Arduino Nano 33 BLE Sense

This repository contains a **TinyML implementation of an LSTM (Long Short-Term Memory)** neural network running on the Arduino Nano 33 BLE Sense. 

The model performs real-time translation of Electrocardiogram (ECG) signals into Photoplethysmogram (PPG) waveforms. This specific version utilizes **offline CSV data playback**, allowing for consistent testing and demonstration of the inference engine without requiring attached analog sensors.

## 🚀 Features

* **On-Device Inference:** Runs a custom LSTM model entirely on the ARM Cortex-M4 processor.
* **Optimized Math:** Uses fixed-point arithmetic (scaling factor 1000) to bypass expensive floating-point operations, optimized for the Arduino Nano's architecture.
* **Data Playback:** Simulates live sensor input using pre-loaded high-resolution ECG data stored in Flash memory (`PROGMEM`).
* **Signal Processing:** Includes real-time peak detection and heart rate estimation (BPM).
* **Bluetooth Low Energy (BLE):** Streams ECG, predicted PPG, and Heart Rate data to connected central devices (phones/tablets).
* **Fault Tolerance:** Includes fallback signal generation if model confidence drops.

## 🛠 Hardware Requirements

* **Arduino Nano 33 BLE Sense**
* *Note: No external sensors are required for this version as it uses stored test data.*

## 📂 Project Structure

* `arduino_nano_csv_playback.ino`: The main entry point. Handles setup, the main loop, BLE advertising, and data orchestration.
* `arduino_nano_tinyml_model.cpp/h`: The core inference engine. Implements the LSTM cell forward pass, gate computations (Sigmoid/Tanh approximations), and signal post-processing.
* `arduino_nano_model_weights.cpp`: Contains the pre-trained weights and biases exported from PyTorch, converted to fixed-point integers.
* `ecg_data_array.h`: A C-header file containing 10,000 samples of raw ECG data for playback.

## ⚙️ Technical Details

### The Model
The system uses a lightweight LSTM architecture designed for edge constraints:
* **Input Size:** 1 (ECG Sample)
* **Hidden Size:** 16 units
* **Sequence Length:** 16
* **Optimization:** Custom matrix multiplication kernels optimized for SRAM usage.

### Signal Pipeline
1.  **Preprocessing:** Raw ECG data (from Flash) is normalized.
2.  **Inference:** Data is fed into the LSTM layers.
3.  **Post-processing:** Output is smoothed using a moving average filter.
4.  **Analysis:** Peak detection algorithms calculate real-time Heart Rate.
5.  **Transmission:** Data is packaged and sent via BLE notifications.

## 💻 How to Run

1.  **Install Dependencies:**
    * Download the Arduino IDE.
    * Install the **Arduino Mbed OS Nano Boards** package via Board Manager.
    * Install the **ArduinoBLE** library via Library Manager.

2.  **Upload:**
    * Open `arduino_nano_csv_playback.ino`.
    * Connect your Nano 33 BLE Sense.
    * Select the board and port.
    * Click **Upload**.

3.  **Monitor:**
    * **Serial Monitor:** Open at `115200` baud to see text-based logs and heart rate stats.
    * **BLE:** Use apps like **nRF Connect** or **LightBlue** to scan for a device named `TinyML-CSV`. You can subscribe to the characteristics to see the streaming waveforms.

## 📊 Performance

* **Sampling Rate:** 360 Hz
* **Memory Usage:** ~40KB for data storage (Flash), efficient SRAM usage for buffers.
* **Inference Time:** Optimized to run within the sampling interval.

## 📜 License

This project is open-source. Please check individual file headers for specific licensing information.
