from fastapi import FastAPI, HTTPException
from fastapi import File, UploadFile, Form
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from PIL import Image
import uvicorn
import zipfile
import csv, os
import tempfile
import stat
import logging


logging.basicConfig(filename="app.log", level=logging.INFO)
logging.info("Starting APP")

classes = ["HOUSES","APARTMENT"]

app = FastAPI()

try:
    model = load_model('model.h5')
except Exception as e:
    logging.info(f"Error loading model: {e}")
    model = None

@app.get("/health")
async def health_check():
    logging.info("Health check endpoint accessed")
    return {"status": "OK"}

@app.post('/predict')
async def predict(rooms: str = Form(...), meters: str = Form(...), image_file: UploadFile = File(...)):
    logging.info("Executing Online Prediction")
    if not model:
        raise HTTPException(status_code=500, detail="Model could not be loaded")
        logging.info("Model could not be loaded")
    try:
        image = Image.open(image_file.file)
        image = image.resize((180, 180))
        image = np.expand_dims(image, axis=0)
        logging.info("Image loaded")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")
        logging.info(f"Error when opening image: {e}")
    try:
        tabular = np.expand_dims([rooms,meters], axis=0).astype(float)
        logging.info("Tabular data loaded")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing tabular: {e}")
        logging.info(f"Tabular data could not be loaded: {e}")
    try:
        predictions = model.predict([image, tabular])
        logging.info(f"Predictions: {predictions}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")
        logging.info(f"Model Predict Error: {e}")
    try:
        # Obten la clase con mayor score
        predicted_class = np.argmax(predictions[0])
        logging.info(f"Got class predicted: {predicted_class}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interpreting prediction: {e}")
        logging.info(f"Error while selecting predicted class: {e}")

    return {"predicted_class": classes[int(predicted_class)]}

def extract_zip(file_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(file_path))

@app.post('/predict_batch')
async def predict(file: UploadFile = File(...)):
    logging.info("Executing Batch Prediction")
    if not model:
        raise HTTPException(status_code=500, detail="Model could not be loaded")
    try:
        # Save the uploaded zip file
        file_path = f"{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
           
         # Extract the contents of the zip file
        extract_zip(file_path)
        logging.info("Zip file extracted")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {e}")
        logging.info("Error processing file")
    predictions_classes = []
    #os.chmod('table.csv', stat.S_IRWXU)
    with open('table.csv', encoding='utf-8-sig') as file_obj:
        reader_obj = csv.reader(file_obj)
        try: 
            for row in reader_obj:
                        
                try:
                    image = Image.open("images/"+row[0])
                    image = image.resize((180, 180))
                    image = np.expand_dims(image, axis=0)
                    logging.info("Image loaded")
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error processing image: {e}")
                    logging.info(f"Error when opening image: {e}")
                try:
                    tabular = np.expand_dims([str(row[1]),str(row[2])], axis=0).astype(float)
                    logging.info("Tabular data loaded")
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error processing tabular: {e}")
                    logging.info(f"Tabular data could not be loaded: {e}")
                try:
                    predictions = model.predict([image,tabular])
                    logging.info(f"Predictions: {predictions}")
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")
                    logging.info(f"Model Predict Error: {e}")
                try:
                    # Obten la clase con mayor score
                    predictions_classes.append(classes[int(np.argmax(predictions[0]))])
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error interpreting prediction: {e}")
                    logging.info(f"Error while selecting predicted class: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error after opening csv: {e}")     
            logging.info(f"Error after opening csv: {e}")  
    #except Exception as e:
        #raise HTTPException(status_code=500, detail=f"Error in CSV reader: {e}")

    return {"predicted_class": predictions_classes}


if __name__ == "__main__":
    import uvicorn  
    uvicorn.run(app, host="0.0.0.0", port=80)
