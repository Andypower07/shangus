  import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 경사하강법 함수 (다중 선형 회귀)
def gradient_descent(X, y, learning_rate=0.01, iterations=100):
    m, n = X.shape  # m: 데이터 수, n: 특성 수
    w = np.zeros(n)  # 가중치 초기화
    b = 0.0  # 편향 초기화
    losses = []

    for _ in range(iterations):
        # 예측값 계산
        y_pred = np.dot(X, w) + b
        # 손실 함수 (MSE)
        loss = (1/m) * np.sum((y_pred - y) ** 2)
        losses.append(loss)
        # 기울기 계산
        dw = (2/m) * np.dot(X.T, (y_pred - y))
        db = (2/m) * np.sum(y_pred - y)
        # 가중치와 편향 업데이트
        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b, losses

# Streamlit 앱
st.title("Student Grade Prediction with Gradient Descent")
st.write("Predict exam scores based on study hours, attendance rate, and assignment scores.")

# 사용자 입력 (사이드바)
st.sidebar.header("Model Parameters")
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
iterations = st.sidebar.slider("Iterations", 100, 1000, 500, 50)

# 샘플 데이터 생성
np.random.seed(42)
n_samples = 50
study_hours = np.random.uniform(1, 10, n_samples)  # 1~10시간
attendance_rate = np.random.uniform(50, 100, n_samples)  # 50~100%
assignment_scores = np.random.uniform(60, 100, n_samples)  # 60~100점
# 가정: 성적 = 5*공부시간 + 0.5*출석률 + 0.3*과제점수 + 10 + 노이즈
y = 5 * study_hours + 0.5 * attendance_rate + 0.3 * assignment_scores + 10 + np.random.normal(0, 5, n_samples)
X = np.column_stack((study_hours, attendance_rate, assignment_scores))

# 사용자 데이터 입력 옵션
st.write("Use sample data or input your own data points.")
use_sample_data = st.checkbox("Use Sample Data", value=True)

if not use_sample_data:
    user_data = st.text_area("Enter data (Study Hours, Attendance %, Assignment Score, Exam Score) one per line, e.g., '5,90,85,92'")
    if user_data:
        try:
            data = [line.split(",") for line in user_data.strip().split("\n")]
            X = np.array([[float(row[0]), float(row[1]), float(row[2])] for row in data])
            y = np.array([float(row[3]) for row in data])
        except:
            st.error("Invalid data format. Please enter as 'Study Hours,Attendance %,Assignment Score,Exam Score' per line.")
            X, y = np.array([]), np.array([])

# 예측 입력
st.header("Predict Exam Score")
study_hours_input = st.number_input("Study Hours (1-10)", min_value=1.0, max_value=10.0, value=5.0)
attendance_input = st.number_input("Attendance Rate (%) (50-100)", min_value=50.0, max_value=100.0, value=90.0)
assignment_input = st.number_input("Assignment Score (0-100)", min_value=0.0, max_value=100.0, value=85.0)

# 경사하강법 실행
if len(X) > 0 and len(y) > 0:
    if st.button("Train Model and Predict"):
        w, b, losses = gradient_descent(X, y, learning_rate, iterations)
        st.write(f"Learned Parameters: w1 (Study Hours) = {w[0]:.2f}, w2 (Attendance) = {w[1]:.2f}, w3 (Assignment) = {w[2]:.2f}, Bias = {b:.2f}")

        # 예측
        input_data = np.array([study_hours_input, attendance_input, assignment_input])
        predicted_score = np.dot(input_data, w) + b
        st.write(f"Predicted Exam Score: {predicted_score:.2f}")

        # 손실 그래프
        st.write("### Loss Over Iterations")
        fig, ax = plt.subplots()
        ax.plot(range(iterations), losses, color="green")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (MSE)")
        ax.set_title("Loss over Iterations")
        st.pyplot(fig)

        # 데이터와 예측 시각화 (공부 시간 vs 성적)
        st.write("### Study Hours vs Exam Score")
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], y, color="blue", label="Data Points")
        X_range = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
        y_range = w[0] * X_range + w[1] * np.mean(X[:, 1]) + w[2] * np.mean(X[:, 2]) + b
        ax.plot(X_range, y_range, color="red", label="Fitted Line")
        ax.scatter([study_hours_input], [predicted_score], color="green", s=100, label="Predicted Point")
        ax.set_xlabel("Study Hours")
        ax.set_ylabel("Exam Score")
        ax.legend()
        st.pyplot(fig)
else:
    st.write("Please provide valid data to train the model.")

# GitHub 및 Streamlit Cloud 배포 안내
st.write("""
### Deployment Instructions
1. **Save Code**: Save this code as `student_grade_prediction.py`.
2. **GitHub**:
   - Create a new GitHub repository (e.g., `student-grade-prediction`).
   - Create a `requirements.txt` with:
     ```
     streamlit
     numpy
     matplotlib
     ```
   - Push the code:
     ```bash
     git add student_grade_prediction.py requirements.txt
     git commit -m "Add student grade prediction app"
     git push origin main
     ```
3. **Streamlit Cloud**:
   - Go to [Streamlit Cloud](https://streamlit.io/cloud).
   - Connect your GitHub account and select the repository.
   - Specify `student_grade_prediction.py` as the main file.
   - Deploy the app.
4. **Run**: Access the app via the provided Streamlit Cloud URL.
""")
