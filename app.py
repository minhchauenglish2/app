import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import skew, kurtosis
import io
import base64 # Import this for potential future use, though not strictly needed for text download
import datetime # Để thêm thời gian vào báo cáo

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Phân loại học sinh",
    layout="wide", # Dùng layout rộng để có nhiều không gian hơn
    initial_sidebar_state="expanded" # Mở rộng sidebar mặc định
)

# Thêm một chút CSS tùy chỉnh để làm đẹp
st.markdown("""
<style>
/* Gradient background for the entire app */
.stApp {
    background: linear-gradient(135deg, #2A0845 0%, #6441A5 100%);
}

/* Main header styling */
.main-header {
    font-size: 2.5em;
    color: #fff;
    text-align: center;
    margin-bottom: 20px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

/* Card-like containers for content */
.content-card {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
    background: rgba(255, 255, 255, 0.1);
    padding: 10px;
    border-radius: 10px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: nowrap;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 7px;
    gap: 10px;
    padding: 0 20px;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255, 255, 255, 0.3);
    transform: translateY(-2px);
}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    background: rgba(138, 43, 226, 0.6);
    color: white;
    border-bottom: 3px solid #FFD700;
}

/* Button styling */
.stButton>button {
    background: linear-gradient(45deg, #8A2BE2, #9932CC);
    color: white;
    border-radius: 8px;
    padding: 10px 25px;
    font-size: 1em;
    font-weight: bold;
    border: none;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(138, 43, 226, 0.4);
}

/* DataFrames and other containers */
.stDataFrame {
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.05);
}

/* Text colors for better visibility */
.streamlit-expanderHeader, .stMarkdown, .stText, .stAlert, .stInfo, .stWarning, .stSuccess {
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)

# Add decorative header with image
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1 class='main-header'>Hệ thống phân loại học sinh <br>dựa trên thành tích và hành vi học tập</h1>
    <h2 class='main-header' style='font-size:1.5em;'>NHÓM 3</h2>
</div>
""", unsafe_allow_html=True)

# Add decorative icons using emoji instead of external images
st.markdown("""
<div class='content-card'>
    <div style='display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 20px;'>
        <div style='text-align: center;'>
            <div style='font-size: 50px;'>👨‍🏫</div>
            <p style='color: white;'>Huấn luyện</p>
        </div>
        <div style='text-align: center;'>
            <div style='font-size: 50px;'>📊</div>
            <p style='color: white;'>Phân tích</p>
        </div>
        <div style='text-align: center;'>
            <div style='font-size: 50px;'>🎯</div>
            <p style='color: white;'>Dự đoán</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Biến toàn cục để lưu trữ encoder và cột huấn luyện
# Sử dụng st.session_state để giữ trạng thái giữa các lần rerun của Streamlit
if 'grade_encoder' not in st.session_state:
    st.session_state['grade_encoder'] = None
if 'X_train_cols' not in st.session_state:
    st.session_state['X_train_cols'] = None
if 'rf_model' not in st.session_state:
    st.session_state['rf_model'] = None
if 'uploaded_train_file_content' not in st.session_state:
    st.session_state['uploaded_train_file_content'] = None # Lưu nội dung file để cache
if 'uploaded_predict_file_content' not in st.session_state:
    st.session_state['uploaded_predict_file_content'] = None # Lưu nội dung file để cache
if 'training_report_content' not in st.session_state:
    st.session_state['training_report_content'] = None # Lưu nội dung báo cáo huấn luyện


# Hàm tải và tiền xử lý dữ liệu
@st.cache_data(show_spinner=False) # Cache dữ liệu để không xử lý lại nếu file không đổi, ẩn spinner mặc định
def load_and_preprocess_data(uploaded_file_buffer, is_prediction_data=False):
    """
    Tải và tiền xử lý dữ liệu.
    Bao gồm ánh xạ Grade, xử lý missing value, one-hot encoding và xử lý skew/kurtosis.
    """
    df = None
    if uploaded_file_buffer is not None:
        try:
            df = pd.read_csv(uploaded_file_buffer)
        except Exception as e:
            st.error(f"Lỗi khi đọc file CSV: {e}. Vui lòng kiểm tra định dạng file.")
            return None, None
    else:
        # Chỉ tạo dữ liệu mẫu nếu không phải là dữ liệu dự đoán
        if not is_prediction_data:
            st.warning("Không có file dữ liệu được tải lên. Đang sử dụng dữ liệu mẫu giả lập.")
            data = {
                'G1': np.random.randint(0, 20, 200),
                'G2': np.random.randint(0, 20, 200),
                'G3': np.random.randint(0, 20, 200),
                'Final_Score': np.random.randint(0, 100, 200),
                'Quizzes_Avg': np.random.randint(0, 100, 200),
                'Sleep_Hours_per_Night': np.random.randint(4, 10, 200),
                'Assignments_Avg': np.random.randint(0, 100, 200),
                'Internet_Access_at_Home': np.random.choice(['Yes', 'No'], 200),
                'Parent_Education_Level': np.random.choice(['High School', 'College', 'University', 'Masters'], 200),
                'Attendance (%)': np.random.randint(50, 100, 200)
            }
            df = pd.DataFrame(data)
            for col in df.select_dtypes(include='object').columns:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mode()[0])
            for col in df.select_dtypes(include=np.number).columns:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mean())
        else: # Nếu là dữ liệu dự đoán nhưng không có file
            return None, None

    if df is None: # Trường hợp lỗi hoặc không có dữ liệu để xử lý
        return None, None

    st.subheader("📊 Dữ liệu gốc đã tải lên:")
    st.dataframe(df.head())
    st.write(f"Kích thước dữ liệu: `{df.shape[0]}` hàng, `{df.shape[1]}` cột")

    # Kiểm tra các cột cần thiết
    required_initial_cols = ['G1', 'G2', 'G3', 'Final_Score', 'Quizzes_Avg', 'Sleep_Hours_per_Night', 'Assignments_Avg', 'Internet_Access_at_Home', 'Parent_Education_Level', 'Attendance (%)']
    missing_initial_cols = [col for col in required_initial_cols if col not in df.columns]
    
    has_g_cols = all(col in df.columns for col in ['G1', 'G2', 'G3'])
    has_final_score = 'Final_Score' in df.columns

    if not has_g_cols and not has_final_score:
        st.error("Dữ liệu thiếu các cột điểm cần thiết (G1, G2, G3 hoặc Final_Score). Không thể tiếp tục phân tích.")
        return None, None

    if missing_initial_cols and (has_g_cols or has_final_score):
        st.warning(f"Dữ liệu có thể thiếu một số cột khuyến nghị: `{', '.join(missing_initial_cols)}`. Đang tiếp tục với các cột hiện có.")

    st.subheader("⚙️ Khám phá dữ liệu và Tiền xử lý:")

    # Ánh xạ điểm thành Grade
    def map_grade(score):
        if score < 30:
            return 'Weak'
        elif score < 40:
            return 'Average'
        elif score < 50:
            return 'Good'
        else:
            return 'Excellent'

    if has_g_cols:
        df['total_grade'] = df['G1'] + df['G2'] + df['G3']
    elif has_final_score:
        df['total_grade'] = df['Final_Score'] / 100 * 60
    
    df['Grade'] = df['total_grade'].apply(map_grade)
    st.write("Dữ liệu sau khi thêm cột 'Grade':")
    st.dataframe(df.head())
    
    # Biểu đồ phân phối Grade sau khi mapping
    st.write("Phân phối các loại Grade:")
    fig_grade_dist, ax_grade_dist = plt.subplots(figsize=(7, 5))
    sns.countplot(x='Grade', data=df, ax=ax_grade_dist, order=['Weak', 'Average', 'Good', 'Excellent'], palette='viridis')
    ax_grade_dist.set_title("Phân phối học lực")
    st.pyplot(fig_grade_dist)
    plt.close(fig_grade_dist)

    # Xử lý Missing Value (loại bỏ các hàng có NaN)
    initial_rows = df.shape[0]
    df.dropna(inplace=True)
    st.write(f"Đã loại bỏ `{initial_rows - df.shape[0]}` hàng chứa giá trị thiếu (NaN).")
    st.dataframe(df.head())

    # Category Encoding cho cột 'Grade'
    encoder = LabelEncoder()
    df['Grade_encoded'] = encoder.fit_transform(df['Grade'])
    
    # One-hot encoding cho các cột category khác
    object_columns = df.select_dtypes(include=['object', 'category']).columns.drop(['Grade'], errors='ignore')
    if len(object_columns) > 0:
        df = pd.get_dummies(df, columns=object_columns, drop_first=True, dtype='int')
        st.write("Dữ liệu sau khi One-Hot Encoding các cột phân loại:")
        st.dataframe(df.head())
    else:
        st.write("Không có cột kiểu object/category nào khác để thực hiện One-Hot Encoding.")

    # Phân tích phân phối và xử lý skewness/kurtosis
    st.subheader("📈 Phân tích phân phối và xử lý độ lệch/độ nhọn")
    numeric_cols = df.select_dtypes(include=np.number).columns.drop(['Grade_encoded', 'total_grade'], errors='ignore')
    
    # Biểu đồ ma trận tương quan sau tiền xử lý
    st.write("Ma trận tương quan của các biến số sau tiền xử lý:")
    if not numeric_cols.empty and len(numeric_cols) > 1:
        fig_corr_processed, ax_corr_processed = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr_processed)
        ax_corr_processed.set_title("Ma trận tương quan sau tiền xử lý")
        st.pyplot(fig_corr_processed)
        plt.close(fig_corr_processed)
    else:
        st.info("Không đủ cột số để vẽ ma trận tương quan sau tiền xử lý.")

    for col in numeric_cols:
        if col in df.columns:
            st.markdown(f"##### Cột: `{col}`")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"Phân phối gốc của cột `{col}`:")
                fig_hist, ax_hist = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax_hist, color='skyblue')
                ax_hist.set_title(f'Phân phối của {col}')
                st.pyplot(fig_hist)
                plt.close(fig_hist)

            sk = skew(df[col])
            kt = kurtosis(df[col])
            st.write(f"Độ lệch (Skewness): `{sk:.2f}`")
            st.write(f"Độ nhọn (Kurtosis): `{kt:.2f}`")

            original_skew = skew(df[col])
            transformation_applied = False
            if abs(original_skew) > 1:
                if original_skew > 1:
                    df[col] = np.log1p(df[col])
                    st.write(f"-> Đã áp dụng Log transformation cho cột `{col}` do độ lệch dương lớn.")
                    transformation_applied = True
                elif original_skew < -1:
                    st.write(f"-> Cột `{col}` có độ lệch âm. Không áp dụng biến đổi tự động.")
            
            if transformation_applied:
                with col2:
                    st.write(f"Phân phối của `{col}` sau xử lý độ lệch:")
                    fig_hist_skew, ax_hist_skew = plt.subplots()
                    sns.histplot(df[col], kde=True, ax=ax_hist_skew, color='lightcoral')
                    ax_hist_skew.set_title(f'Phân phối của {col} sau xử lý')
                    st.pyplot(fig_hist_skew)
                    plt.close(fig_hist_skew)
                st.write(f"Độ lệch (Skewness) sau xử lý: `{skew(df[col]):.2f}`")
            st.markdown("---") # Đường phân cách giữa các cột

    return df, encoder

# Hàm tạo báo cáo huấn luyện dưới dạng chuỗi Markdown/Text
def create_training_report_content(accuracy, confusion_matrix_data, classification_report_text, grade_classes):
    report_string = io.StringIO()
    
    report_string.write(f"# Báo cáo Huấn luyện và Đánh giá Mô hình Phân loại Học sinh\n\n")
    report_string.write(f"Ngày và giờ tạo báo cáo: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report_string.write("---\n\n")

    report_string.write("## 1. Tổng quan hiệu suất mô hình\n")
    report_string.write(f"- Accuracy trên tập kiểm tra: **{accuracy:.4f}**\n\n")

    report_string.write("## 2. Ma trận nhầm lẫn (Confusion Matrix)\n")
    report_string.write("Ma trận này cho thấy số lượng các dự đoán đúng và sai của mô hình cho từng lớp.\n")
    
    cm_df = pd.DataFrame(confusion_matrix_data, index=grade_classes, columns=grade_classes)
    report_string.write("```\n")
    report_string.write(cm_df.to_string())
    report_string.write("\n```\n\n")

    report_string.write("## 3. Báo cáo phân loại (Classification Report)\n")
    report_string.write("Báo cáo này cung cấp các chỉ số Precision, Recall, F1-score và Support cho từng lớp học lực.\n")
    report_string.write("```\n")
    report_string.write(classification_report_text)
    report_string.write("\n```\n\n")
    
    report_string.write("## 4. Ghi chú\n")
    report_string.write("Mô hình Random Forest đã được huấn luyện với các tham số và quy trình tiền xử lý được mô tả trong ứng dụng.\n")
    report_string.write("Kết quả này có thể thay đổi tùy thuộc vào dữ liệu đầu vào và các tham số mô hình.\n\n")
    report_string.write("---\n")
    report_string.write("Báo cáo được tạo tự động bởi Hệ thống phân loại học sinh (Nhóm 3).\n")

    return report_string.getvalue()


# --- Tạo các Tabs ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button[aria-selected="false"] {
        background-color: #E6E6FA;  /* Nền tím nhạt khi chưa chọn */
        color: black;   /* Chữ đen khi chưa chọn */
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #8A2BE2;  /* Nền tím đậm khi đã chọn */
        color: white;   /* Chữ trắng khi đã chọn */
    }
</style>
""", unsafe_allow_html=True)

tab_intro, tab_train_eval, tab_predict = st.tabs(["Giới thiệu 📚", "Huấn luyện & Đánh giá Mô hình 📈", "Dự đoán Dữ liệu Mới 🔮"])

with tab_intro:
    st.markdown("## Chào mừng đến với Hệ thống Phân loại Học sinh!")
    st.markdown("""
    <div class='content-card'>
        <p style="font-size: 1.1em; color: #E0FFFF;">
            Ứng dụng này được thiết kế để phân loại học lực của học sinh (<b>Weak, Average, Good, Excellent</b>) 
            dựa trên các yếu tố liên quan đến thành tích học tập và hành vi. Mục tiêu là giúp giáo viên và phụ huynh 
            có cái nhìn sâu sắc hơn về tình hình học tập của học sinh.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 Cách sử dụng ứng dụng:")
    st.markdown("""
    <ul style="list-style-type: square; color: #E0FFFF;">
        <li><b><span style="color: #FFD700;">Tab 'Huấn luyện & Đánh giá Mô hình'</span>:</b>
            <ul>
                <li>Tải lên file dữ liệu <code>CSV</code> của bạn (ví dụ: <code>student_data.csv</code>) để huấn luyện mô hình.</li>
                <li>Ứng dụng sẽ tự động thực hiện các bước tiền xử lý dữ liệu và huấn luyện mô hình Random Forest.</li>
                <li>Bạn sẽ thấy các báo cáo đánh giá hiệu suất mô hình như Accuracy, Confusion Matrix, và Classification Report.</li>
            </ul>
        </li>
        <li><b><span style="color: #FFD700;">Tab 'Dự đoán Dữ liệu Mới'</span>:</b>
            <ul>
                <li>Sau khi mô hình được huấn luyện, bạn có thể tải lên một file <code>CSV</code> chứa dữ liệu của các học sinh mới.</li>
                <li>Mô hình sẽ dự đoán học lực cho từng học sinh và hiển thị kết quả cùng phân phối học lực dự đoán.</li>
            </ul>
        </li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🧠 Về thuật toán Random Forest:")
    st.markdown("""
    <div class='content-card'>
        <p style="font-style: italic; color: #E0FFFF;">
        Random Forest là một thuật toán học máy mạnh mẽ, thuộc họ các thuật toán học tập ensemble. 
        Nó hoạt động bằng cách xây dựng nhiều cây quyết định trong quá trình huấn luyện và xuất ra 
        lớp là chế độ của các lớp (đối với bài toán phân loại) của các cây riêng lẻ. 
        Thuật toán này nổi tiếng với độ chính xác cao và khả năng xử lý tốt cả dữ liệu số và phân loại, 
        giúp nó trở thành lựa chọn lý tưởng cho bài toán phân loại học sinh này.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #E0FFFF; margin-top: 30px;'>Cảm ơn bạn đã sử dụng ứng dụng của chúng tôi!</p>", unsafe_allow_html=True)


with tab_train_eval:
    st.markdown("## 📊 Huấn luyện & Đánh giá Mô hình")
    st.write("Tại đây, bạn sẽ tải lên dữ liệu huấn luyện để xây dựng và đánh giá mô hình phân loại học sinh. Sau khi tải lên, quá trình tiền xử lý và huấn luyện sẽ tự động diễn ra.")
    
    uploaded_train_file = st.file_uploader(
        "**Bước 1: Tải lên file CSV dữ liệu học sinh để huấn luyện** (Ví dụ: `student_data.csv`)",
        type=["csv"],
        key="train_file_uploader_main"
    )

    df_processed = None
    grade_encoder = None

    if uploaded_train_file is not None:
        st.session_state['uploaded_train_file_content'] = uploaded_train_file.getvalue()
        
        with st.spinner("Đang đọc và tiền xử lý dữ liệu huấn luyện..."):
            df_processed, grade_encoder = load_and_preprocess_data(io.BytesIO(st.session_state['uploaded_train_file_content']))
        
        if df_processed is not None and grade_encoder is not None:
            st.session_state['grade_encoder'] = grade_encoder
            st.success("🎉 Tiền xử lý dữ liệu huấn luyện hoàn tất! Sẵn sàng huấn luyện mô hình.")

            st.markdown("### 2.1. Huấn luyện mô hình")
            X = df_processed.drop(['Grade', 'Grade_encoded', 'total_grade'], axis=1, errors='ignore')
            y = df_processed['Grade_encoded']

            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.dropna(axis=1)

            st.session_state['X_train_cols'] = X.columns.tolist()

            if len(X) < 2 or len(y.unique()) < 2:
                st.error("Dữ liệu không đủ để chia tập huấn luyện/kiểm tra hoặc chỉ có một lớp duy nhất. Vui lòng cung cấp thêm dữ liệu đa dạng hơn.")
                st.stop()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            st.write(f"Kích thước tập huấn luyện: `{X_train.shape}`")
            st.write(f"Kích thước tập kiểm tra: `{X_test.shape}`")
            
            with st.expander("Xem phân phối lớp trong tập huấn luyện"):
                st.write("Phân phối lớp trong tập huấn luyện:")
                st.bar_chart(pd.Series(y_train).map(dict(enumerate(grade_encoder.classes_))).value_counts())

            st.info("Đang huấn luyện mô hình Random Forest... 🚀")
            with st.spinner("Mô hình đang được huấn luyện... Quá trình này có thể mất vài giây."):
                rf_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    class_weight='balanced',
                    random_state=42
                )
                rf_model.fit(X_train, y_train)
                st.session_state['rf_model'] = rf_model
            st.success("✅ Mô hình đã được huấn luyện thành công!")

            st.markdown("### 2.2. Đánh giá mô hình")
            y_pred = rf_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.markdown(f"<h3 style='color:#00FF00;'>Accuracy của mô hình trên tập kiểm tra: <span style='font-weight:bold;'>{acc:.4f}</span></h3>", unsafe_allow_html=True) # Màu xanh lá

            # Lấy confusion matrix và classification report
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=grade_encoder.classes_)

            with st.expander("Xem Ma trận nhầm lẫn (Confusion Matrix)"):
                st.markdown("#### Ma trận nhầm lẫn (Confusion Matrix)")
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                            xticklabels=grade_encoder.classes_,
                            yticklabels=grade_encoder.classes_)
                ax_cm.set_xlabel("Dự đoán")
                ax_cm.set_ylabel("Thực tế")
                ax_cm.set_title("Confusion Matrix")
                st.pyplot(fig_cm)
                plt.close(fig_cm)

            with st.expander("Xem Báo cáo phân loại (Classification Report)"):
                st.markdown("#### Báo cáo phân loại (Classification Report)")
                st.code(report)

            with st.expander("Xem Phân phối các lớp học lực sau huấn luyện"):
                st.markdown("#### Phân phối các lớp học lực sau huấn luyện")
                fig_class_dist, ax_class_dist = plt.subplots(figsize=(8, 6))
                sns.countplot(x=y_train, ax=ax_class_dist, palette='pastel')
                ax_class_dist.set_title("Phân phối lớp học lực trong tập huấn luyện")
                ax_class_dist.set_xlabel("Mã hóa học lực")
                ax_class_dist.set_ylabel("Số lượng")
                ax_class_dist.set_xticklabels(grade_encoder.classes_)
                st.pyplot(fig_class_dist)
                plt.close(fig_class_dist)
            
            # Sau khi tất cả đánh giá hoàn tất, tạo và lưu báo cáo
            st.session_state['training_report_content'] = create_training_report_content(
                acc, cm, report, grade_encoder.classes_
            )

            # Nút tải báo cáo
            st.markdown("### 2.3. Tải báo cáo huấn luyện")
            st.download_button(
                label="Tải xuống Báo cáo Huấn luyện & Đánh giá (TXT) 📄",
                data=st.session_state['training_report_content'],
                file_name="bao_cao_huan_luyen_mo_hinh_hoc_sinh.txt",
                mime="text/plain",
                key="download_training_report"
            )

        else:
            st.error("❌ Không thể tiền xử lý dữ liệu từ file đã tải lên. Vui lòng kiểm tra lại file của bạn.")
    else:
        st.info("💡 Vui lòng tải lên file CSV để bắt đầu quá trình huấn luyện và đánh giá mô hình. Bạn cũng có thể sử dụng dữ liệu mẫu.")
        if st.button("Huấn luyện với dữ liệu mẫu", key="train_sample_data_button"):
            with st.spinner("Đang tải và tiền xử lý dữ liệu mẫu..."):
                df_processed_sample, grade_encoder_sample = load_and_preprocess_data(None)
            
            if df_processed_sample is not None and grade_encoder_sample is not None:
                st.session_state['grade_encoder'] = grade_encoder_sample
                st.session_state['uploaded_train_file_content'] = "sample_data_loaded"
                st.success("✅ Đã sử dụng dữ liệu mẫu để tiền xử lý. Đang chuyển sang huấn luyện...")
                st.experimental_rerun()
            else:
                st.error("❌ Không có dữ liệu để xử lý. Vui lòng tải lên file hoặc kiểm tra lỗi dữ liệu mẫu.")


with tab_predict:
    st.markdown("## 🔮 Dự đoán Học lực cho Dữ liệu Mới")
    st.write("Sử dụng mô hình đã huấn luyện để dự đoán học lực cho dữ liệu học sinh mới của bạn. Đảm bảo file mới có cấu trúc tương tự file huấn luyện.")

    if st.session_state['rf_model'] is not None and \
       st.session_state['X_train_cols'] is not None and \
       st.session_state['grade_encoder'] is not None:
        
        st.success("✅ Mô hình đã được huấn luyện và sẵn sàng để dự đoán. Vui lòng tải lên file CSV để nhận dự đoán.")
        
        uploaded_prediction_file = st.file_uploader(
            "**Bước 1: Tải lên file CSV dữ liệu học sinh mới để dự đoán**",
            type=["csv"],
            key="predict_file_uploader_main"
        )

        if uploaded_prediction_file is not None:
            st.session_state['uploaded_predict_file_content'] = uploaded_prediction_file.getvalue()

            st.subheader("3.1. Dữ liệu mới đã tải lên:")
            new_df = pd.read_csv(io.BytesIO(st.session_state['uploaded_predict_file_content']))
            st.dataframe(new_df.head())

            st.subheader("3.2. Kết quả dự đoán:")

            new_df_processed = new_df.copy()
            
            has_g_cols_new = all(col in new_df_processed.columns for col in ['G1', 'G2', 'G3'])
            has_final_score_new = 'Final_Score' in new_df_processed.columns

            if has_g_cols_new:
                new_df_processed['total_grade'] = new_df_processed['G1'] + new_df_processed['G2'] + new_df_processed['G3']
            elif has_final_score_new:
                new_df_processed['total_grade'] = new_df_processed['Final_Score'] / 100 * 60
            else:
                st.error("❌ File dự đoán không có các cột điểm cần thiết (G1, G2, G3 hoặc Final_Score). Không thể dự đoán.")
                st.stop()

            # Xử lý các cột object/category
            object_cols_new = new_df_processed.select_dtypes(include=['object', 'category']).columns
            if len(object_cols_new) > 0:
                new_df_processed = pd.get_dummies(new_df_processed, columns=object_cols_new, drop_first=True, dtype='int')

            # Đồng bộ cột: thêm các cột bị thiếu vào dữ liệu mới với giá trị 0
            for col in st.session_state['X_train_cols']:
                if col not in new_df_processed.columns:
                    new_df_processed[col] = 0

            # Loại bỏ các cột thừa trong dữ liệu mới không có trong X_train
            new_df_processed = new_df_processed[st.session_state['X_train_cols']]

            # Dự đoán
            try:
                with st.spinner("Đang dự đoán học lực... 🧠"):
                    predictions_encoded = st.session_state['rf_model'].predict(new_df_processed)
                    predictions_decoded = st.session_state['grade_encoder'].inverse_transform(predictions_encoded)
                
                new_df['Predicted_Grade'] = predictions_decoded
                st.dataframe(new_df[['G1', 'G2', 'G3', 'total_grade', 'Predicted_Grade']].head(10))

                st.markdown("#### Phân phối học lực dự đoán trên dữ liệu mới")
                fig_pred_dist, ax_pred_dist = plt.subplots(figsize=(8, 6))
                sns.countplot(x=new_df['Predicted_Grade'], ax=ax_pred_dist, order=st.session_state['grade_encoder'].classes_, palette='magma')
                ax_pred_dist.set_title("Phân phối học lực dự đoán")
                ax_pred_dist.set_xlabel("Học lực")
                ax_pred_dist.set_ylabel("Số lượng")
                st.pyplot(fig_pred_dist)
                plt.close(fig_pred_dist)

                csv_output = new_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Tải xuống kết quả dự đoán (CSV) 📥",
                    data=csv_output,
                    file_name="predicted_grades.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"❌ Đã xảy ra lỗi khi dự đoán: {e}")
                st.write("Vui lòng kiểm tra lại cấu trúc file CSV của bạn. Đảm bảo các cột đầu vào cần thiết cho mô hình đã có trong file và kiểu dữ liệu phù hợp.")
                st.write(f"Các cột mô hình mong đợi: `{', '.join(st.session_state['X_train_cols'])}`")

        else:
            st.info("💡 Tải lên file CSV để nhận dự đoán.")
    else:
        st.warning("⚠️ Mô hình chưa được huấn luyện. Vui lòng chuyển đến tab 'Huấn luyện & Đánh giá Mô hình' để huấn luyện mô hình trước khi dự đoán.")