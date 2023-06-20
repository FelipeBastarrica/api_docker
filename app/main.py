from fastapi import FastAPI, HTTPException
from fastapi import File, UploadFile, Form
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from PIL import Image
import uvicorn
import zipfile
import csv


print('STARTING APP')
classes = ["HOUSES","APARTMENT"]

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

@app.post('/predict_batch')
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Model could not be loaded")
    try:
        with zipfile.ZipFile(file.file, 'r') as zip_ref:
            zip_ref.extractall(file.file)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {e}")

    try:
        predictions_classes = []
        with open('table.csv') as file_obj:
            reader_obj = csv.reader(file_obj)
            for row in reader_obj:
                row_splitted = row.split(",")

                try:
                    image = Image.open("images/"+row_splitted)
                    image = image.resize((180, 180))
                    image = np.expand_dims(image, axis=0)
                    print(image)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

                try:
                    tabular = np.expand_dims([row_splitted[1],row_splitted[2]], axis=0).astype(float)

                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error processing tabular: {e}")

                try:
                    predictions = model.predict([image,tabular])
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")

                try:
                    # Obten la clase con mayor score
                    predictions_classes.append(np.argmax(predictions[0]))
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error interpreting prediction: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in CSV reader: {e}")



    return {"predicted_class": predictions_classes}


if __name__ == "__main__":
    import uvicorn  
    uvicorn.run(app, host="0.0.0.0", port=80)
