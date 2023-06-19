from fastapi import FastAPI, HTTPException
from fastapi import File, UploadFile, Form
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from PIL import Image
import uvicorn

print('STARTING APP')

app = FastAPI()

try:
    model = load_model('model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.post('/predict')
async def predict(rooms: str = Form(...), meters: str = Form(...), image_file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model could not be loaded")
    try:
        image = Image.open(image_file.file)
        image = image.resize((180, 180))
        image = np.expand_dims(image, axis=0)
        print(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

    try:
        #print(data['rooms'])
        #print(data['meters'])
        tabular = np.expand_dims([rooms,meters], axis=0).astype(float)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing tabular: {e}")

    try:
        predictions = model.predict([image,tabular])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")

    try:
        # Obten la clase con mayor score
        predicted_class = np.argmax(predictions[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interpreting prediction: {e}")

    return {"predicted_class": int(predicted_class)}

if __name__ == "__main__":
    import uvicorn  
    uvicorn.run(app, host="0.0.0.0", port=80)
