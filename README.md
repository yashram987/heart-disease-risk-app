# heart-disease-risk-app
This repository contains my heart disease risk prediction app, created for my master’s dissertation. The model is built using the Health Survey for England dataset and XGBoost, combining both clinical information (such as blood pressure and cholesterol) and lifestyle factors (such as smoking, alcohol, and physical activity). I’ve added Explainable AI tools (SHAP and LIME) so you can see which factors most influence each prediction. The app also includes a rule-based system that turns these SHAP explanations into simple, personalised lifestyle tips based on NHS guidelines and prior knowledge from my Biomedical Sciences degree. 

In the app, you can explore predictions for a few example patients, see which factors drive their risk up or down using explainable AI, and also view tailored advice based on the SHAP explainations

App Link: https://heart-disease-risk-app-1.streamlit.app/
