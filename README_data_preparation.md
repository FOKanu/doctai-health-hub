# Medical Image Data Preparation Pipeline

This pipeline organizes and cleans medical image datasets (MRI, CT, X-ray) for machine learning training by removing watermarks, labels, and text that could negatively impact model performance.

## 🎯 Features

### **1. Dataset Organization**
- **Unified Structure**: Organizes all datasets into a consistent format
- **Modality Separation**: Separates MRI, CT, and X-ray images
- **Proper Splits**: Creates train/validation/test splits (70%/15%/15%)
- **Class Mapping**: Standardizes class names across datasets

### **2. Image Cleaning**
- **Text Removal**: Detects and removes text overlays using EasyOCR or morphological operations
- **Watermark Detection**: Identifies and removes semi-transparent watermarks
- **Label Cleaning**: Removes medical labels and annotations
- **Inpainting**: Uses advanced inpainting to fill removed regions naturally

### **3. Quality Assurance**
- **Metadata Generation**: Creates comprehensive dataset statistics
- **Logging**: Detailed logging of all operations
- **Error Handling**: Robust error handling and recovery
- **Progress Tracking**: Real-time progress updates

## 📁 Directory Structure

After processing, your dataset will be organized as follows:

```
organized_medical_dataset/
├── train/
│   ├── mri/
│   │   ├── normal_t1/
│   │   ├── normal_t2/
│   │   ├── tumor/
│   │   └── ...
│   ├── ct/
│   │   ├── normal/
│   │   ├── tumor/
│   │   ├── cyst/
│   │   └── ...
│   └── xray/
│       ├── normal/
│       ├── pneumonia/
│       ├── tuberculosis/
│       └── ...
├── validation/
│   ├── mri/
│   ├── ct/
│   └── xray/
├── test/
│   ├── mri/
│   ├── ct/
│   └── xray/
└── metadata/
    └── dataset_metadata.json
```

## 🚀 Quick Start

### **Option 1: Using the Shell Script (Recommended)**

```bash
# Make the script executable
chmod +x run_data_preparation.sh

# Run the data preparation
./run_data_preparation.sh
```

### **Option 2: Manual Execution**

```bash
# Install dependencies
pip3 install -r data_preparation_requirements.txt

# Run the pipeline
python3 data_preparation_pipeline.py \
    --source_dir "/Users/francis/Downloads/MRI_CT_XRAY_Datset" \
    --output_dir "./organized_medical_dataset"
```

### **Option 3: Skip Image Cleaning (Faster)**

```bash
python3 data_preparation_pipeline.py \
    --source_dir "/Users/francis/Downloads/MRI_CT_XRAY_Datset" \
    --output_dir "./organized_medical_dataset" \
    --skip_cleaning
```

## 📋 Requirements

### **System Requirements**
- Python 3.8 or higher
- 8GB+ RAM (for large datasets)
- 10GB+ free disk space

### **Python Dependencies**
```
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
tqdm>=4.65.0
easyocr>=1.7.0
pathlib2>=2.3.7
```

## ⚙️ Configuration

You can customize the pipeline behavior by editing `data_preparation_config.json`:

### **Image Cleaning Settings**
```json
{
  "image_cleaning": {
    "enabled": true,
    "text_detection": {
      "confidence_threshold": 0.5,
      "use_easyocr": true
    },
    "watermark_detection": {
      "contrast_threshold": 10,
      "min_area": 100,
      "max_area": 10000
    }
  }
}
```

### **Dataset Splits**
```json
{
  "dataset_splits": {
    "train_ratio": 0.7,
    "validation_ratio": 0.15,
    "test_ratio": 0.15,
    "random_seed": 42,
    "stratify": true
  }
}
```

## 🔧 Advanced Usage

### **Custom Source Directory**
```bash
python3 data_preparation_pipeline.py \
    --source_dir "/path/to/your/dataset" \
    --output_dir "./custom_output"
```

### **Process Only Specific Modalities**
Edit the `organize_all_datasets()` method in `data_preparation_pipeline.py` to comment out unwanted modalities:

```python
def organize_all_datasets(self):
    # Organize each modality
    self.organize_mri_dataset()      # Comment out to skip MRI
    self.organize_ct_datasets()      # Comment out to skip CT
    self.organize_xray_dataset()     # Comment out to skip X-ray
```

### **Custom Class Mapping**
Modify the `class_mapping` section in `data_preparation_config.json` to customize class names.

## 📊 Output Files

### **1. Organized Dataset**
- Clean, organized images in train/validation/test splits
- Consistent naming and structure
- Ready for ML training

### **2. Metadata File**
```json
{
  "dataset_info": {
    "name": "Medical Imaging Trio Dataset",
    "total_images": 15000,
    "modalities": ["mri", "ct", "xray"]
  },
  "class_distribution": {
    "normal": {"train": 1000, "validation": 150, "test": 150},
    "tumor": {"train": 800, "validation": 120, "test": 120}
  },
  "modality_distribution": {
    "mri": {"train": 3000, "validation": 450, "test": 450},
    "ct": {"train": 8000, "validation": 1200, "test": 1200},
    "xray": {"train": 4000, "validation": 600, "test": 600}
  }
}
```

### **3. Log File**
Detailed logging of all operations in `data_preparation.log`.

## 🎯 Supported Datasets

### **MRI Datasets**
- Brain tumor classification (45 classes)
- T1, T1C+, T2 sequences
- Normal vs. tumor detection

### **CT Datasets**
- Brain CT (tumor, aneurysm, cancer)
- Lung CT (cancer detection)
- Kidney CT (normal, cyst, tumor, stone)
- Brain stroke CT (normal, bleeding, ischemia)
- COVID-19 CT (COVID-19 vs. non-COVID-19)

### **X-Ray Datasets**
- Chest X-ray (17 diseases)
- Pneumonia, tuberculosis, COVID-19
- Cardiomegaly, atelectasis, emphysema
- Fractures, pneumothorax, effusions

## 🔍 Quality Control

### **Before Training**
1. **Review Sample Images**: Check cleaned images for quality
2. **Verify Splits**: Ensure proper class distribution
3. **Check Metadata**: Review dataset statistics
4. **Test Loading**: Verify images load correctly

### **Common Issues**
- **Text Not Removed**: Increase confidence threshold
- **Over-cleaning**: Reduce detection sensitivity
- **Memory Issues**: Process in smaller batches
- **Slow Processing**: Use `--skip_cleaning` for faster processing

## 🚨 Troubleshooting

### **EasyOCR Installation Issues**
```bash
# On macOS
brew install tesseract
pip3 install easyocr

# On Ubuntu
sudo apt-get install tesseract-ocr
pip3 install easyocr

# Alternative: Use basic text detection
# Edit config to set "use_easyocr": false
```

### **Memory Issues**
```bash
# Reduce batch size in config
"performance": {
  "batch_size": 16,
  "num_workers": 2
}
```

### **Permission Issues**
```bash
# Make script executable
chmod +x run_data_preparation.sh

# Check directory permissions
ls -la /Users/francis/Downloads/MRI_CT_XRAY_Datset
```

## 📈 Performance Tips

1. **Use SSD Storage**: Faster I/O for large datasets
2. **Increase RAM**: More memory for batch processing
3. **Use GPU**: Enable GPU acceleration if available
4. **Parallel Processing**: Adjust `num_workers` in config
5. **Skip Cleaning**: Use `--skip_cleaning` for faster processing

## 🤝 Contributing

To improve the pipeline:

1. **Report Issues**: Create detailed bug reports
2. **Suggest Features**: Propose new cleaning methods
3. **Add Datasets**: Extend support for new modalities
4. **Optimize Performance**: Improve processing speed

## 📄 License

This pipeline is part of the Health AI Hub project. Use responsibly and ensure compliance with medical data regulations.

## 🆘 Support

For issues or questions:
1. Check the log file: `data_preparation.log`
2. Review this README
3. Check the configuration file
4. Verify your dataset structure

---

**Happy Training! 🎉**
