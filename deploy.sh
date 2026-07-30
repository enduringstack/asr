#!/bin/bash
# Install the built ASR HAP to a device.
# Usage: ./deploy.sh [device_serial]

DEVICE="${1:-5XM0125A10000251}"
HAP_PATH="entry/build/default/outputs/default/entry-default-signed.hap"
MODEL_DIR="/Users/cannkit/ASR/entry/src/main/resources/rawfile/models"

echo "=== Deploying to device: $DEVICE ==="

# Check device is connected
echo "Checking device..."
hdc list targets | grep -q "$DEVICE"
if [ $? -ne 0 ]; then
    echo "ERROR: Device $DEVICE not found!"
    echo "Available devices:"
    hdc list targets
    exit 1
fi

echo ""
echo "=== Checking packaged model files ==="
for model in model-streaming-fixed-floatmask.om units.txt model_metadata.json; do
    if [ ! -f "$MODEL_DIR/$model" ]; then
        echo "ERROR: $model not found at $MODEL_DIR/$model"
        exit 1
    fi
    echo "Found $model"
done

# Install/update HAP
echo ""
echo "=== Installing HAP ==="
if [ -f "$HAP_PATH" ]; then
    hdc -t $DEVICE install -r "$HAP_PATH"
    echo "HAP installed successfully"
else
    echo "WARNING: HAP not found at $HAP_PATH"
    echo "Please build the project first in DevEco Studio"
fi

echo ""
echo "=== Deployment complete ==="
echo "Launch the app on device to test"
