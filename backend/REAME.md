# Income Prediction API

This Flask API allows making predictions about individuals' income based on various characteristics.

!!!
The model is trained with scikit-learn version 1.6.0 not 1.3.0

Issue with the model:
- always predict <50K

!!!

## Available Endpoints

### 1. API Health Check
```bash
GET /api/health
```
Response:
```json
{
    "status": "ok"
}
```

### 2. Features List
```bash
GET /api/features
```
Returns the list of available features for prediction.

### 3. Simple Prediction
```bash
POST /api/predict
```
Returns a binary prediction (>50K or <=50K)

Request example:
```json
{
    "education_Bachelors": 1,
    "marital.status_Married": 1,
    "occupation_Exec-managerial": 1,
    "race_White": 1,
    "relationship_Husband": 1,
    "sex_Female": 0,
    "workclass_Private": 1,
    "age": 38,
    "capital.gain": 0,
    "capital.loss": 0,
    "hours.per.week": 45
}
```

You can also use the `apiExemple.json` file as an example input data.

### 4. Prediction Probabilities
```bash
POST /api/predict_proba
```
Returns probabilities for each class.

### 5. Model Information
```bash
GET /api/model_info
```
Returns information about the used model:
- Model type
- Training date
- Number of features
- Number of estimators
- Maximum depth

### 6. Feature Explanation
```bash
POST /api/explain
```
Returns the importance of each feature in the model.

## Usage Example

To test the API, you can use the `apiExemple.json` file as an example input data. Here's a curl request example:

```bash
curl -X POST http://localhost:5000/api/predict \
     -H "Content-Type: application/json" \
     -d @apiExemple.json
```

## Running the Server

### Using Docker
```bash
# Build the Docker image
docker build -t flask-mlops .

# Run the container
docker run -p 5000:5000 flask-mlops
```

The server will be accessible at `http://localhost:5000`
