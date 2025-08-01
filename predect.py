import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# App title
st.title("📊 Student Performance Predictor")
st.markdown("Upload a CSV file with student data and predict exam scores with Pass/Fail status.")

# File uploader
uploaded_file = st.file_uploader("📁 Upload your CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read the CSV file with a safer encoding
        data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

        # Check for required columns
        required_columns = ['name', 'study_hours', 'attendance', 'assignments', 'exam_score']
        if not all(col in data.columns for col in required_columns):
            st.error(f"❌ CSV must contain the columns: {', '.join(required_columns)}")
        else:
            st.success("✅ Data loaded successfully!")

            # Show data preview
            st.subheader("📄 Uploaded Data Preview")
            st.write(data.head())

            # Split data
            x = data[['study_hours', 'attendance', 'assignments']]
            y = data['exam_score']

            # Train model
            model = LinearRegression()
            model.fit(x, y)

            # Predict exam scores
            data['predicted_score'] = model.predict(x)

            # Add Pass/Fail status
            data['status'] = data['predicted_score'].apply(lambda x: 'Pass' if x >= 50 else 'Fail')

            # Show results
            st.subheader("📈 Prediction Results")
            st.dataframe(data[['name', 'predicted_score', 'status']])

            # Download button for results (encoding in ISO-8859-1 to avoid error)
            csv_result = data.to_csv(index=False).encode('ISO-8859-1')
            st.download_button(
                label="⬇ Download Result CSV",
                data=csv_result,
                file_name='predicted_results.csv',
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"❌ Error reading or processing file: {e}")

else:
    st.info("📌 Please upload a CSV file to begin.")