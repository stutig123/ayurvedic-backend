# 🌿 Plant Species Identifier API

This is a Flask-based deep learning API for identifying medicinal and common Indian plants using a trained MobileNetV2 model. The API takes an image as input and returns the predicted plant species along with a confidence score.

---

## 🚀 Features

- ✅ Image classification using a trained **MobileNetV2** model
- ✅ Predicts from 40+ Indian plant species
- ✅ Built with **Flask** and **TensorFlow**
- ✅ Simple API endpoint for frontend integration
- ✅ CORS-enabled for cross-origin requests

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/plant-identifier-api.git
cd plant-identifier-api
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
Or install manually:
```bash
pip install flask flask-cors tensorflow pillow numpy
```
3. Add the Model File
Create a folder called model and place your trained model file inside:
```bash
plant-identifier-api/
├── app.py
├── model/
│   └── mobilenetv2_final_model.keras
```
4. Run the Server
```bash
python app.py
The API will start on: http://localhost:5000
```
