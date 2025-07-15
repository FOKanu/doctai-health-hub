#!/bin/bash

# Medical Image Data Preparation Script
# ====================================

echo "🏥 Medical Image Data Preparation Pipeline"
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

# Install required packages
echo "📦 Installing required packages..."
pip3 install -r data_preparation_requirements.txt

# Set source and output directories
SOURCE_DIR="/Users/francis/Downloads/MRI_CT_XRAY_Datset"
OUTPUT_DIR="./organized_medical_dataset"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ Source directory not found: $SOURCE_DIR"
    echo "Please update the SOURCE_DIR variable in this script or move your dataset to the correct location."
    exit 1
fi

echo "📁 Source directory: $SOURCE_DIR"
echo "📁 Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the data preparation pipeline
echo "🚀 Starting data preparation..."
python3 data_preparation_pipeline.py \
    --source_dir "$SOURCE_DIR" \
    --output_dir "$OUTPUT_DIR"

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Data preparation completed successfully!"
    echo "📊 Check the organized dataset in: $OUTPUT_DIR"
    echo "📋 Check the log file: data_preparation.log"
    echo ""
    echo "📈 Dataset Summary:"
    if [ -f "$OUTPUT_DIR/metadata/dataset_metadata.json" ]; then
        echo "   - Metadata file created"
        echo "   - Check metadata/dataset_metadata.json for details"
    fi
    echo ""
    echo "🎯 Next steps:"
    echo "   1. Review the cleaned images in the output directory"
    echo "   2. Verify the train/validation/test splits"
    echo "   3. Use the organized dataset for ML training"
else
    echo "❌ Data preparation failed. Check the log file for details."
    exit 1
fi
