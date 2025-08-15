#imports required modules
import streamlit as st
import numpy as np
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

st.set_page_config(page_title='Heart Disease Risk using Example Patients',layout='centered')#webpage title

#loads saved files
xgb=joblib.load('xgboostmodel.pkl')
features=joblib.load('featurenames.pkl')
limebase=joblib.load('limebase.pkl')
examples=joblib.load('examplecases.pkl')#example patients from notebook

#columns used in training model
numcols=['sbp','dbp','alcoholunits7d','weeklyactivitytotal','educend','bmi','cholhdlratio']
catcols=['sex','agegroup','diabetes','diabetesmed','walkpace','smokeever','cigarettetype','income3','topqual','marital','tenure','deprivation','urbanrural','region','hypertension']

#maps patient label to stored key
patientmap={'Patient 1':'high','Patient 2':'mid','Patient 3':'low'}
 
#build 1xn vector from patient data (handles numeric + categorical including "Missing")
def buildvector(vals):
    x=np.zeros(len(features),dtype=float) #empty vector for all features
    for k in numcols:
        if k in features and k in vals:
            try:x[features.index(k)]=float(vals[k])
            except:pass
    for k in catcols:
        v=str(vals.get(k,'Missing'))#default to 'Missing' if no value
        col=k+v
        if col in features:x[features.index(col)]=1.0
    return x

#lifestyle advice based on my previous knowledge through my biomedical science degree and NHS advice
def advice(v):
    tips=[]
    try:
        sbp=float(v.get('sbp',0));dbp=float(v.get('dbp',0))
        bmi=float(v.get('bmi',0))
        act=float(v.get('weeklyactivitytotal',0))
        alc=float(v.get('alcoholunits7d',0))
        ratio=float(v.get('cholhdlratio',0))
        cig=str(v.get('cigarettetype',''));smoke=str(v.get('smokeever',''))
        
        if sbp>=140 or dbp>=90:tips.append('Lower blood pressure: less salt, be active, take medication as advised by Doctor.')  #blood pressure
        else:tips.append('Good: Blood pressure is within target.')
        if bmi>=25:tips.append('Aim for a healthier weight through diet and exercise.') #BMI
        elif 18.5<=bmi<25:tips.append('Good: your healthy weight is lowering risk.')
        if act<150:tips.append('Do ≥150 mins/week exercise (or 75 mins of vigorous exercise).')#exercise
        else:tips.append('Good: your exercise level is helping reduce risk.')
        if alc>14:tips.append('Keep alcohol intake to ≤14 units/week with some alcohol‑free days.') #alcohol
        else:tips.append('Good: Alcohol intake is within guidance (≤14 units/week).')
        if ratio>5:tips.append('Improve cholesterol ratio by eating less saturated fat, more fibre & oily fish.')
        else:tips.append('Good: Your cholesterol ratio looks healthy.')

        # smoking
        if cig.startswith('Current') or (smoke=='Yes' and 'Used to' not in cig):tips.append('Stop smoking; use NHS stop‑smoking support.')
        elif cig.startswith('Used to') or smoke=='No':tips.append('Good: not smoking is lowering your risk.')
    except:pass
    if not tips:tips.append('No tips, maintain your healthy lifestyle.')
    return tips

#app ui
st.title('Heart Disease Risk Calculator Using Sample Patients')
who=st.radio('Choose patient',['Patient 1','Patient 2','Patient 3'],horizontal=True)
key=patientmap[who]
vals=examples[key]

if st.button('Predict'):
    st.markdown('### Patient details')
    st.write(vals)
    x=buildvector(vals).reshape(1,-1)
    p=float(xgb.predict_proba(x)[:,1][0])
    st.subheader(f'Estimated probability of existing Cardiovascular Disease: {p*100:.1f}%')
    
    #shap explanation
    st.markdown('### SHAP explanation')
    expl=shap.TreeExplainer(xgb)
    sv=expl.shap_values(x)
    try:
        svrow=sv[1][0];base=expl.expected_value[1]
    except Exception:
        svrow=sv[0];ev=expl.expected_value
        base=float(ev if np.ndim(ev)==0 else ev[0])
    ex=shap.Explanation(values=svrow,base_values=base,data=x[0],feature_names=features)
    shap.plots.waterfall(ex)
    st.pyplot(plt.gcf()) 
    
    #lime top features
    st.markdown('### LIME top features')
    lime=LimeTabularExplainer(limebase,feature_names=features,class_names=['No','Yes'],mode='classification')
    exp=lime.explain_instance(x[0],xgb.predict_proba)
    st.table({'feature':[k for k,_ in exp.as_list()],'weight':[float(v) for _,v in exp.as_list()]})

    #zips feature names with shap values
    featureimportance=sorted(zip(features, svrow), key=lambda x: abs(x[1]), reverse=True) #this is 
    zipshap=sorted(zip(features, svrow), key=lambda x: abs(x[1]), reverse=True)
    #defines which features are modifiable risk factors
    modifiable=['sbp','dbp','bmi','cholhdlratio','alcoholunits7d','weeklyactivitytotal','smokeever','cigarettetype']
    #keep only modifiable features with positive shap values as that increases CVD risk
    topmodifiable=[f for f, v in featureimportance if f in modifiable and v>0]
    #creates a dictionary with only those features from patient values
    targetvals={k:vals[k] for k in topmodifiable if k in vals}
    
    #advice section filtered by shap features so using XAI to for advice to be triggered
    st.markdown('### Personalised advice based on SHAP explainations (XAI)')
    for t in advice(targetvals):
        st.write(t)