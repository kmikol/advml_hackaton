# Environmental Model Deployment Instructions

This guide provides step-by-step instructions to preprocess the dataset, optimize the model, and deploy it to an Arduino board for environmental monitoring.

---

## Step 1: Preprocess the Dataset

Run the following command to preprocess the dataset:

```bash
make environmental
```

### What This Does:
- **Dataset Preprocessing**: Cleans and prepares the raw data for training.
- **Hyperparameter Optimization**: Runs optimization with quantization to find the best model parameters.
- **Model Selection**: Selects the best model that meets the design constraints.
- **Model Training**: Trains the selected model and saves it to the model registry and to the arduino project folder.

---

## Step 2: Deploy the Model to Arduino

Once the model is trained, it needs to be deployed to the Arduino board. Follow these steps:

1. Open the Arduino IDE.
2. Navigate to the project directory:
   ```
   ./src/advml_hackaton/environmental/arduino
   ```
3. Open the project files in the Arduino IDE.
4. Upload the project to the Arduino board.

### Notes:
- Ensure that the Arduino board is connected to your computer.
- Verify the correct board and port settings in the Arduino IDE before uploading.

---

## Step 3: Monitor Outputs via Bluetooth

To monitor the outputs from the Arduino board via Bluetooth, run the following command:

```bash
make monitor_ble
```

### What This Does:
- Connects to the Arduino board via Bluetooth.
- Streams real-time data from the board to your terminal.

### Tips:
- Ensure that Bluetooth is enabled on your computer.
- Keep the Arduino board powered on and within range.

---

By following these steps, you will successfully deploy the environmental model and monitor its outputs in real time. If you encounter any issues, refer to the troubleshooting section in the project documentation or reach out to the development team.