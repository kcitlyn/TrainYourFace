# TrainMyFace
TrainMyFace is a cutting-edge, terminal-driven face recognition and training system leveraging the powerful combination of OpenCV and dlib. Seamlessly capture, train, and identify faces in real-time — all from your webcam — with a streamlined and privacy-focused approach. Ideal for developers, researchers, and hobbyists passionate about computer vision and biometric identification.

## 🚀 Features
Robust Face Training & Recognition: Capture face images on the fly using the spacebar, automatically process and store facial descriptors, and perform accurate recognition in live video streams.

Multi-Face Detection: Detect and identify multiple faces simultaneously with real-time bounding boxes and names displayed elegantly on-screen.

Highly Scalable: Train an unlimited number of identities, managing face data and personal attributes with secure JSON storage — ensuring privacy and customization.

Advanced Matching Algorithm: Employs Euclidean distance comparisons of 128-dimensional facial embeddings for precise and reliable identity matching.

Relationship Metadata Support: Beyond recognition, assign and store contextual relationships (e.g., family ties) to enhance personal identification layers.

## ⚙️ Installation & Setup
Requirements
- Setup a virtual environment in main/root folder
```
python3 venv venv
```
- Activate the virtual environment 
  - Windows
```
.\venv\scripts\activate
```
  - Linux/ Mac
```
source myenv/bin/activate
```
- Python 3.11.9 (or compatible earlier versions)
- CMake (required for dlib compilation)
```
pip install cmake
```
- A C++ build toolchain (common ones below):
  - Windows: Visual Studio Build Tools
  - Linux: GCC/G++
  - macOS: Xcode Command Line Tools

### Model Weights
Download the essential dlib models and place them in the face_detection/face_data/models/ directory:
- shape_predictor_5_face_landmarks.dat.bz2
  - This can be downloaded from the following URL: [http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2)
- dlib_face_recognition_resnet_model_v1.dat.bz2
  - This can be downloaded from the following URL: [http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)
Extract the .bz2 archives after downloading and move them to the proper folder.

### Install Dependencies
```
pip install -r requirements.txt
```
### 🎯 Usage
Launch the program from the project root directory:
```
python3 main.py
```
Choose between two interactive modes:
- Training Mode: Capture and label face images. Press the spacebar to take snapshots for model training.
- Identification Mode: Real-time detection and identification of faces with dynamic overlays showing recognized names.
All facial data and metadata are securely saved in JSON files, created on the first run, ensuring full data privacy and user control.

## 🙌 Credits
This project stands on the shoulders of giants, powered by:
- dlib: Industry-standard face detection and recognition.
- OpenCV: Advanced computer vision and image processing library.
Thank you to the open-source community for making these groundbreaking tools accessible.

## 📄 License
Licensed under the MIT License, granting you the freedom to use, modify, and distribute this project with minimal restrictions.

## 🔮 Future Personal Roadmap and Possibilities for Contributions
- Text-to-Speech Integration: Announce recognized individuals audibly for enhanced accessibility and interactivity.
- Rich Face Property Display: Add detailed contextual information and relationship tags for deeper personalization.
- User-Friendly GUI: Develop a sleek graphical interface for effortless operation beyond the terminal.
- More versatility for training options (for example, via file upload)
  - (in the works)

## 🤝 Contributing
Contributions, feature requests, and bug reports are warmly welcomed! Whether you’re enhancing core functionality or polishing user experience, your input is invaluable and super exciting!
Feel free to fork the repo, submit pull requests, or reach out directly via email or message for anything!
If TrainMyFace has helped you at all or you enjoyed using it, please consider giving this repository a ⭐️!