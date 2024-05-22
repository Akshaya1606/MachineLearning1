import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

st.title("Emotion detection using Random Forest Algorithm")
emotfile=pd.read_csv(r"C:\Users\Akshaya\Desktop\VS_DataScience\emotion_dataset_raw.csv")
x=emotfile['Text'].values
y=emotfile['Emotion'].values
label_encoder = LabelEncoder()
y=label_encoder.fit_transform(y)
vectorizer = TfidfVectorizer(max_features=5000)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_tfidf, y_train)
y_pred = clf.predict(X_test_tfidf)

tabs=st.tabs(["Home","Accuracy","Applications"])
with tabs[0]:
    st.write("Welcome to user friendly Streamlit Application")
    st.write("Please enter the input to check the emotion")
    t=st.text_input("Enter the text")
    if st.button("Detect"):
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        tvec = vectorizer.transform([t])
        predi=clf.predict(tvec)
        deresult=label_encoder.inverse_transform(predi)[0]

        st.success("Emotion Detected")
        if deresult=='joy':
            st.write(":smile: Joy")
            st.image(r"C:\Users\Akshaya\Downloads\36409_hd.jpg")
        if deresult=='anger':
            st.write(":rage: Anger")
            st.image(r"C:\Users\Akshaya\Downloads\meditation-for-anger.jpg.d4a1fadcdc8a7f5d95eb0804b46659d7.jpg")
        if deresult=='disgust':
            st.write(":confounded: Disgust")
            st.image(r"C:\Users\Akshaya\Downloads\th.jpeg")
        if deresult=='fear':
            st.write(":cry: Fear")
            st.image(r"C:\Users\Akshaya\Downloads\th (1).jpeg")
        if deresult=='neutral':
            st.write(":neutral_face: Neutral")
            st.image(r"C:\Users\Akshaya\Downloads\th (2).jpeg")
        if deresult=='sadness':
            st.write(":worried: Sad")
            st.image(r"C:\Users\Akshaya\Downloads\th (3).jpeg")
        if deresult=='shame':
            st.write(":man-facepalming: Shame")
            st.image(r"C:\Users\Akshaya\Downloads\th (4).jpeg")
        if deresult=='surprise':
            st.write(":open_mouth: Surprise")
            st.image(r"C:\Users\Akshaya\Downloads\th (5).jpeg")

with tabs[1]:
    rep=classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    report_df = pd.DataFrame(rep).transpose()
    st.dataframe(report_df)
    st.write(f"**Accuracy Score: {accuracy_score(y_test,y_pred)}**")

with tabs[2]:
    st.header("Real Time Application")
    st.write(":headphones: Customer Service and Support")
    st.write(":iphone: Social Media Monitoring")
    st.write(":shopping_trolley: Product and Service Feedback")
    st.write(":shopping_bags: E-commerce and Retail")
    st.write(" :movie_camera: Youtube comment detection")

