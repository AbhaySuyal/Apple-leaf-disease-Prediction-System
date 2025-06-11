
# 🍏 Apple Leaf Disease Prediction using CNN

This project uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify apple leaf diseases. It helps identify healthy and infected apple leaves using image data. A Streamlit web app is provided to make predictions through a user-friendly interface.

## 🧠 Project Overview

- **Goal**: Detect diseases in apple leaves from images.
- **Model**: Convolutional Neural Network (CNN)
- **Frameworks**: TensorFlow, Keras, Streamlit
- **Accuracy**: ~88% on test set

---

## 📂 Dataset

- **Source**: [Kaggle - Plant Pathology 2021 FGVC8](https://www.kaggle.com/c/plant-pathology-2021-fgvc8/data?select=train.csv)
- **Classes**:
  - Apple Scab
  - Black Rot
  - Cedar Apple Rust
  - Healthy

You’ll need to download the dataset from Kaggle and extract the contents.

---

## 📦 Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```text
tensorflow==2.10.0
numpy
pandas
matplotlib
scikit-learn
opencv-python
streamlit
```

---

## 🏗️ Project Structure

```
apple-leaf-disease-prediction/
├── dataset/
│   └── train/ (images)
│   └── train.csv
├── model/
│   └── apple_leaf_model.h5
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Training the Model

To train the model from scratch:

```bash
python train_model.py
```

This will:
- Load images and labels
- Preprocess data
- Train a CNN
- Save the model as `model/apple_leaf_model.h5`

---

## 🌐 Streamlit Web App

To run the web application:

```bash
streamlit run app.py
```

Then open the URL provided in the terminal to upload an image and view predictions in the browser.

---

## 📸 Example Prediction

Upload a sample leaf image using the Streamlit interface. The model will return one of:

- ✅ Healthy
- ⚠️ Apple Scab
- ⚠️ Black Rot
- ⚠️ Cedar Apple Rust

---

## 🚀 Future Improvements

- Integrate mobile deployment using TensorFlow Lite
- Real-time prediction from mobile camera feed
- Add more data augmentation to improve generalization

---

## 🙌 Acknowledgements

- Kaggle for providing the dataset
- TensorFlow/Keras for deep learning tools
- Streamlit for the web app framework

---

## 📃 License

This project is open-source under the MIT License.
