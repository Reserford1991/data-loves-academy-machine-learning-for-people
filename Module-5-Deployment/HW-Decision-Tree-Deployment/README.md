# Streamlit Weather Prediction Demo

This project gives users an opportunity to test weather prediction model based on decision trees and deployed to Streamlit.
Weather can be predicted in several Ausralian cities.

You can test this app using URL https://module-5-deploymenth-hbhidj.streamlit.app
If you see a message "This app has gone to sleep due to inactivity. Would you like to wake it back up?" just hit the button "Yes, get this app back up!" and wait for some time.

## Project structure

- **csv/**: Data directory (`weatherAUS.csv`).
- **models/**: Directory with trained model.
- **app.py**: File to run strdeamlit app.
- **requirements.txt**: Required python packages list
- **train.ipynb**: Jupyter Notebook to train Random Forest model.

## Confuguration

Project uses python 3.12 and all libraries from `requirements.txt` file.

### Встановлення

1. **Clone Repository**:
   ```bash
   git clone https://github.com/Reserford1991/data-loves-academy-machine-learning-for-people.git
   cd Module-5-Deployment/HW-Decision-Tree-Deployment
   ```

2. **Create virtual environment** (not required, but would be a plus):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install all required libraries**:
   ```bash
   pip install -r requirements.txt
   ```

### Model training

In order to train Random Forest classifier, run Jupyter Notebook `train.ipynb`:

1. Open Jupyter Notebook `train.ipynb`.

2. Run all cells one by one and trainde model will be saved into `models/` directory.

### Streamlit app running

Run Streamlit app locally using this command:

```bash
streamlit run app.py
```

App will be available by address `http://localhost:8501`.
