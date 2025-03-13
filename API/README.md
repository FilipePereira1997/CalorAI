# Backend API

This is the backend API for the project. It is a RESTful API that is built using the Fast Api Framework.

Inside you will find the following:

### Backend Data
In here you find processed data explored previously in the "raw-dataset".

### APP
The main application that contains the API routes and the database models.

```
App 
│
├── core
│    ├── config.py      # reads and sets the app configurations 
│    └── events.py      # handles app broadcast events
├── db
│   ├── db.py           # setups the database
│   └── events.py       # proccess´s initial and final events of the app startup and shutdown
├── main.py
├── models
|   ├── data            # db models
│   ├── domain          # models to comunicate between modules
│   └── schemas         # API response and request models
├── optimization-module # standalone module for the optimization feature
├── prediction-module   # standalone module for the predection feature
├── routers             # declaration of endpoints
└── services            # logic services (ex: csv data reader)

```

#### Optimization Module
The standalone module that contains the optimization algorithm, using AI, you can find more info inside that folder [README.md](app/optimization_module/README.md).

#### Prediction Module
The standalone module that contains the prediction algorithm, you can find more info inside that folder [README.md](app/prediction_module/README.md).

## Requirements
Normally have the package requirements for your application in `requirements.txt` file.
Docker will install these using pip.

## Docker Run
 To fully run the application and the database you should run the following command:

```bash
docker-compose up --build
```

You can now access the API on `http://localhost:80`.
You should see a "Hello World" message if everything is working correctly.

and access to the documentation on `http://localhost:80/docs` or `http://localhost:80/redoc`.

**note**: First time is slow! it will populate the Database from scratch.
Wait for the message

**"INFO:     Application startup complete."** to appear in the logs.