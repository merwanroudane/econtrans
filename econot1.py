"""
تطبيق شامل لمعالجة وتحويل البيانات الاقتصادية
Comprehensive Economic Data Processing and Transformation Application
من إعداد الدكتور مروان رودان
By Dr. Marouane Roudan
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="معالجة البيانات الاقتصادية | Economic Data Processing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS for warm colors and Arabic support
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #FFF8F0 0%, #FFF5EB 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #FFF8F0 0%, #FFF5EB 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Cairo', sans-serif !important;
        color: #8B4513 !important;
    }
    
    .big-title {
        font-size: 2.5rem;
        color: #8B4513;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #FFE4C4, #FFDAB9);
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(139, 69, 19, 0.2);
    }
    
    .section-header {
        background: linear-gradient(90deg, #D2691E, #CD853F);
        color: white !important;
        padding: 15px 25px;
        border-radius: 10px;
        margin: 20px 0;
        font-size: 1.3rem;
    }
    
    .info-box {
        background: linear-gradient(135deg, #FFF8DC, #FFEFD5);
        border-left: 5px solid #D2691E;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        line-height: 1.8;
    }
    
    .info-box ul, .info-box ol {
        margin: 10px 0;
        padding-left: 25px;
    }
    
    .info-box li {
        margin-bottom: 8px;
    }
    
    .info-box table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }
    
    .info-box th, .info-box td {
        padding: 10px;
        border: 1px solid #DEB887;
        text-align: center;
    }
    
    .info-box th {
        background-color: #FFEFD5;
        color: #8B4513;
    }
    
    .detail-box {
        background: linear-gradient(135deg, #F5F5DC, #FFFACD);
        border: 1px solid #DAA520;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    
    .reference-box {
        background: linear-gradient(135deg, #E6E6FA, #F0E6FA);
        border-left: 4px solid #9370DB;
        padding: 12px 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 0.9rem;
    }
    
    .formula-box {
        background: linear-gradient(135deg, #FAF0E6, #FDF5E6);
        border: 2px solid #DEB887;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        text-align: center;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #FFE4B5, #FFDAB9);
        border-left: 5px solid #FF8C00;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        line-height: 1.8;
    }
    
    .warning-box ul, .warning-box ol {
        margin: 10px 0;
        padding-left: 25px;
    }
    
    .warning-box li {
        margin-bottom: 8px;
    }
    
    .success-box {
        background: linear-gradient(135deg, #F0FFF0, #E8F5E8);
        border-left: 5px solid #228B22;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        line-height: 1.8;
    }
    
    .success-box ul, .success-box ol {
        margin: 10px 0;
        padding-left: 25px;
    }
    
    .success-box li {
        margin-bottom: 8px;
    }
    
    .stSelectbox > div > div {
        background-color: #FFF8F0;
        border-color: #DEB887;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #FFEFD5;
        border-radius: 10px;
        padding: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #FFE4C4;
        border-radius: 8px;
        color: #8B4513;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #D2691E !important;
        color: white !important;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #FFEFD5, #FFE4C4);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(139, 69, 19, 0.15);
    }
    
    .stExpander {
        background-color: #FFF8F0;
        border: 1px solid #DEB887;
        border-radius: 10px;
    }
    
    div[data-testid="stMetricValue"] {
        color: #8B4513;
    }
    
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Cairo', sans-serif;
    }
    
    .bilingual {
        display: flex;
        justify-content: space-between;
        padding: 10px;
        background: #FFF8DC;
        border-radius: 8px;
        margin: 5px 0;
    }
    
    code {
        background-color: #FFF8DC !important;
        color: #8B4513 !important;
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    .stDataFrame {
        background-color: #FFF8F0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #FFE4C4, #FFDAB9);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFF5EB, #FFE4C4);
    }
    
    [data-testid="stSidebar"] .stSelectbox label {
        color: #8B4513 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Helper Functions
# ============================================

def hp_filter(y, lamb=1600):
    """Hodrick-Prescott Filter"""
    n = len(y)
    I = np.eye(n)
    D = np.zeros((n-2, n))
    for i in range(n-2):
        D[i, i] = 1
        D[i, i+1] = -2
        D[i, i+2] = 1
    trend = np.linalg.solve(I + lamb * D.T @ D, y)
    cycle = y - trend
    return trend, cycle

def hamilton_filter(y, h=8, p=4):
    """Hamilton Filter (2018)"""
    n = len(y)
    if n < h + p:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    trend = np.full(n, np.nan)
    cycle = np.full(n, np.nan)
    
    for t in range(h + p - 1, n):
        X = np.column_stack([np.ones(t - h - p + 2)] + 
                           [y[h:t-p+2-i] for i in range(p)])
        if X.shape[0] > p + 1:
            try:
                beta = np.linalg.lstsq(X, y[h+p-1:t+1], rcond=None)[0]
                trend[t] = beta[0] + sum(beta[i+1] * y[t-h-i] for i in range(p))
                cycle[t] = y[t] - trend[t]
            except:
                pass
    
    return trend, cycle

def boosted_hp_filter(y, lamb=1600, max_iter=100, tol=1e-6):
    """Boosted HP Filter (Phillips & Shi, 2021)"""
    n = len(y)
    trend = np.zeros(n)
    residual = y.copy()
    
    for _ in range(max_iter):
        trend_update, _ = hp_filter(residual, lamb)
        trend += trend_update
        new_residual = y - trend
        
        if np.max(np.abs(new_residual - residual)) < tol:
            break
        residual = new_residual
    
    cycle = y - trend
    return trend, cycle

def baxter_king_filter(y, low=6, high=32, K=12):
    """Baxter-King Band-Pass Filter"""
    n = len(y)
    
    omega_low = 2 * np.pi / high
    omega_high = 2 * np.pi / low
    
    b = np.zeros(K + 1)
    b[0] = (omega_high - omega_low) / np.pi
    
    for j in range(1, K + 1):
        b[j] = (np.sin(omega_high * j) - np.sin(omega_low * j)) / (np.pi * j)
    
    b_full = np.concatenate([b[::-1][:-1], b])
    b_full = b_full - np.mean(b_full)
    
    trend = np.full(n, np.nan)
    cycle = np.full(n, np.nan)
    
    for t in range(K, n - K):
        cycle[t] = np.sum(b_full * y[t-K:t+K+1])
        trend[t] = y[t] - cycle[t]
    
    return trend, cycle

def ihs_transform(x, theta=1):
    """Inverse Hyperbolic Sine Transformation"""
    return np.arcsinh(theta * x) / theta

def ihs_inverse(y, theta=1):
    """Inverse of IHS Transformation"""
    return np.sinh(theta * y) / theta

def detect_outliers_zscore(data, threshold=3):
    """Detect outliers using Z-score"""
    z_scores = np.abs(stats.zscore(data[~np.isnan(data)]))
    return np.where(z_scores > threshold)[0]

def detect_outliers_iqr(data, k=1.5):
    """Detect outliers using IQR method"""
    Q1 = np.nanpercentile(data, 25)
    Q3 = np.nanpercentile(data, 75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    return np.where((data < lower) | (data > upper))[0]

def simple_return(prices):
    """Calculate simple returns"""
    return np.diff(prices) / prices[:-1]

def log_return(prices):
    """Calculate log returns"""
    return np.diff(np.log(prices))

def gross_return(prices):
    """Calculate gross returns"""
    return prices[1:] / prices[:-1]

def chow_lin_disaggregate(low_freq, high_freq_indicator, conversion_factor=4):
    """Simple Chow-Lin temporal disaggregation"""
    n_low = len(low_freq)
    n_high = len(high_freq_indicator)
    
    if n_high != n_low * conversion_factor:
        high_freq_indicator = high_freq_indicator[:n_low * conversion_factor]
    
    # Aggregate indicator to low frequency
    indicator_agg = np.array([
        np.sum(high_freq_indicator[i*conversion_factor:(i+1)*conversion_factor])
        for i in range(n_low)
    ])
    
    # Simple regression
    beta = np.sum(low_freq * indicator_agg) / np.sum(indicator_agg ** 2)
    
    # Disaggregate
    high_freq = beta * high_freq_indicator
    
    # Adjust to match low frequency totals
    for i in range(n_low):
        start_idx = i * conversion_factor
        end_idx = (i + 1) * conversion_factor
        current_sum = np.sum(high_freq[start_idx:end_idx])
        if current_sum != 0:
            adjustment = low_freq[i] / current_sum
            high_freq[start_idx:end_idx] *= adjustment
    
    return high_freq

def denton_disaggregate(low_freq, conversion_factor=4):
    """Simple Denton temporal disaggregation without indicator"""
    n_high = len(low_freq) * conversion_factor
    high_freq = np.zeros(n_high)
    
    for i in range(len(low_freq)):
        start_idx = i * conversion_factor
        end_idx = (i + 1) * conversion_factor
        high_freq[start_idx:end_idx] = low_freq[i] / conversion_factor
    
    return high_freq

# ============================================
# Sidebar
# ============================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h2 style="color: #8B4513;">📊 القائمة الرئيسية</h2>
        <h3 style="color: #CD853F;">Main Menu</h3>
    </div>
    """, unsafe_allow_html=True)
    
    section = st.selectbox(
        "اختر القسم | Select Section",
        [
            "🏠 الصفحة الرئيسية | Home",
            "📅 سنة الأساس | Base Year",
            "💰 الأسعار الجارية والثابتة | Current & Constant Prices",
            "📈 التحويلات اللوغاريتمية | Log Transformations",
            "🔄 تحويل IHS | IHS Transformation",
            "📊 الترشيح | Filtering Methods",
            "🔢 العوائد | Returns Calculation",
            "⚠️ القيم الشاذة | Outliers Detection",
            "❓ القيم المفقودة | Missing Values",
            "📆 تحويل التردد | Frequency Conversion",
            "🛠️ أدوات إضافية | Additional Tools"
        ]
    )
    
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #FFE4C4, #FFDAB9); border-radius: 10px;">
        <p style="color: #8B4513; font-weight: bold;">من إعداد</p>
        <p style="color: #D2691E; font-size: 1.1rem; font-weight: bold;">الدكتور مروان رودان</p>
        <p style="color: #8B4513;">Dr. Marouane Roudan</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# Main Content
# ============================================

# Home Page
if "الصفحة الرئيسية" in section:
    st.markdown("""
    <div class="big-title">
        <h1>🎓 معالجة وتحويل البيانات الاقتصادية</h1>
        <h2>Economic Data Processing & Transformation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📅 سنة الأساس</h3>
            <p>Base Year</p>
            <p style="font-size: 0.9rem; color: #666;">تحويل وربط السلاسل الزمنية</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 الترشيح</h3>
            <p>Filtering</p>
            <p style="font-size: 0.9rem; color: #666;">HP, Hamilton, Boosted HP</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔄 التحويلات</h3>
            <p>Transformations</p>
            <p style="font-size: 0.9rem; color: #666;">Log, IHS, Returns</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box">
        <h3>📚 محتويات التطبيق | Application Contents</h3>
        <p>يوفر هذا التطبيق أدوات شاملة لمعالجة وتحويل البيانات الاقتصادية مع شرح نظري مفصل وتطبيقات عملية تفاعلية:</p>
        <ul style="line-height: 2.2;">
            <li><strong>سنة الأساس (Base Year):</strong> مفهوم سنة الأساس، طرق تغييرها، الربط الأمامي والخلفي (Retropolation)، الاستيفاء الخطي، معايير SNA 2008</li>
            <li><strong>الأسعار الجارية والثابتة:</strong> التمييز بينهما، معامل انكماش GDP، مؤشر أسعار المستهلك CPI، طرق التحويل، Fisher Index</li>
            <li><strong>التحويلات اللوغاريتمية:</strong> استخدامات ln في الاقتصاد، المرونات، نماذج Log-Log وLog-Linear، دالة Cobb-Douglas</li>
            <li><strong>تحويل IHS:</strong> البديل للوغاريتم عند وجود أصفار أو سالب، خصائصه، حساسية وحدة القياس، تفسير المعاملات (Bellemare & Wichman 2020)</li>
            <li><strong>طرق الترشيح:</strong> HP Filter، Hamilton Filter (2018)، Boosted HP (Phillips & Shi 2021)، Baxter-King، Christiano-Fitzgerald</li>
            <li><strong>حساب العوائد:</strong> العوائد البسيطة واللوغاريتمية والإجمالية، العوائد الزائدة، CAGR، Sharpe Ratio</li>
            <li><strong>القيم الشاذة:</strong> طرق Z-Score، IQR، MAD، أنواع الشذوذ (AO، LS، TC، IO)، طرق المعالجة</li>
            <li><strong>القيم المفقودة:</strong> أنواع MCAR، MAR، MNAR، الاستيفاء الخطي، Multiple Imputation، KNN</li>
            <li><strong>تحويل التردد:</strong> Chow-Lin، Denton، Fernandez، Litterman، التجميع والتفكيك الزمني</li>
            <li><strong>أدوات إضافية:</strong> اختبارات الاستقرارية ADF وKPSS، التعديل الموسمي X-13، التطبيع، معدلات النمو</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Base Year Section
elif "سنة الأساس" in section:
    st.markdown('<div class="section-header">📅 سنة الأساس | Base Year Concepts</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📖 المفهوم | Concept", "🔢 الحسابات | Calculations", "🔗 الربط | Splicing", "📊 تطبيق عملي | Practical"])
    
    with tabs[0]:
        st.markdown("""
        <div class="info-box">
            <h3>🎯 ما هي سنة الأساس؟ | What is Base Year?</h3>
            <p><strong>العربية:</strong> سنة الأساس هي السنة المرجعية التي تُستخدم لقياس التغيرات في البيانات الإحصائية والاقتصادية عبر الزمن. تُحدد قيمتها عادةً بـ 100 لتسهيل المقارنات.</p>
            <p><strong>English:</strong> Base year is a reference point used to measure changes in statistical and economic data over time. Its value is typically set at 100 to simplify comparisons.</p>
            <hr style="border-color: #DEB887;">
            <h4>📌 معلومات إضافية | Additional Details</h4>
            <p><strong>معايير الأمم المتحدة (SNA 2008/2025):</strong> يوصي نظام الحسابات القومية بتحديث سنة الأساس كل 5 سنوات على الأقل لضمان أن أوزان الأسعار تعكس أنماط الاستهلاك والإنتاج الحالية.</p>
            <p><strong>طريقة السلسلة المتصلة (Chain-linking):</strong> تستخدم العديد من الدول طريقة الربط المتسلسل التي تُحدّث الأوزان سنوياً بدلاً من الاعتماد على سنة أساس ثابتة، مما يقلل من تحيز الاستبدال (Substitution Bias).</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 أهمية سنة الأساس | Importance of Base Year")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="success-box">
                <h4>✅ المزايا | Advantages</h4>
                <ul>
                    <li>تسهيل المقارنات الزمنية بين فترات مختلفة</li>
                    <li>عزل تأثير التضخم عن التغيرات الحقيقية</li>
                    <li>توحيد المعايير للمقارنات الدولية</li>
                    <li>تحسين دقة التحليل الاقتصادي</li>
                    <li>تمكين حساب معدلات النمو الحقيقية</li>
                    <li>دعم اتخاذ القرارات السياسية والاقتصادية</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-box">
                <h4>⚠️ اعتبارات الاختيار | Selection Criteria</h4>
                <ul>
                    <li>اختيار سنة طبيعية (بدون أزمات أو صدمات اقتصادية)</li>
                    <li>تحديث سنة الأساس دورياً (كل 5-10 سنوات)</li>
                    <li>مراعاة التغيرات الهيكلية في الاقتصاد</li>
                    <li>التوافق مع المعايير الدولية (SNA 2008، IMF GFSM)</li>
                    <li>توفر بيانات كاملة وموثوقة للسنة المختارة</li>
                    <li>تمثيل أنماط الاستهلاك والإنتاج الحالية</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📐 الصيغة الرياضية | Mathematical Formula")
        st.latex(r"\text{Index}_t = \frac{\text{Value}_t}{\text{Value}_{\text{base}}} \times 100")
        
        st.markdown("""
        <div class="formula-box">
            <p><strong>حيث | Where:</strong></p>
            <p>Index_t = قيمة المؤشر في السنة t | Index value in year t</p>
            <p>Value_t = القيمة الفعلية في السنة t | Actual value in year t</p>
            <p>Value_base = القيمة في سنة الأساس | Value in base year</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("### 🔢 تغيير سنة الأساس | Base Year Shifting")
        
        st.latex(r"\text{New Index}_t = \frac{\text{Old Index}_t}{\text{Old Index}_{\text{new base}}} \times 100")
        
        st.markdown("### 💻 مثال تطبيقي | Practical Example")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**إدخال البيانات | Input Data:**")
            years = st.text_input("السنوات (مفصولة بفاصلة)", "2015,2016,2017,2018,2019,2020")
            values = st.text_input("القيم (مفصولة بفاصلة)", "100,105,110,115,120,125")
            old_base = st.selectbox("سنة الأساس القديمة", years.split(","))
            new_base = st.selectbox("سنة الأساس الجديدة", years.split(","))
        
        with col2:
            if st.button("🔄 تحويل سنة الأساس | Convert Base Year"):
                try:
                    years_list = [int(y.strip()) for y in years.split(",")]
                    values_list = [float(v.strip()) for v in values.split(",")]
                    
                    old_base_idx = years_list.index(int(old_base))
                    new_base_idx = years_list.index(int(new_base))
                    
                    # Calculate new index
                    new_index = [(v / values_list[new_base_idx]) * 100 for v in values_list]
                    
                    results_df = pd.DataFrame({
                        'السنة | Year': years_list,
                        'القيمة الأصلية | Original': values_list,
                        f'المؤشر الجديد (أساس {new_base})': [round(x, 2) for x in new_index]
                    })
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Visualization
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=years_list, y=values_list, name='Original', 
                                            line=dict(color='#D2691E', width=3)))
                    fig.add_trace(go.Scatter(x=years_list, y=new_index, name='New Index',
                                            line=dict(color='#228B22', width=3)))
                    fig.update_layout(
                        title=f'تحويل سنة الأساس من {old_base} إلى {new_base}',
                        xaxis_title='السنة',
                        yaxis_title='القيمة',
                        template='plotly_white',
                        plot_bgcolor='rgba(255,248,240,0.8)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"خطأ: {e}")
    
    with tabs[2]:
        st.markdown("### 🔗 ربط السلاسل الزمنية | Splicing Time Series")
        
        st.markdown("""
        <div class="info-box">
            <h4>أنواع الربط | Types of Splicing</h4>
            <ol>
                <li><strong>الربط الأمامي (Forward Splicing):</strong> ربط السلسلة الجديدة بسنة الأساس القديمة</li>
                <li><strong>الربط الخلفي (Backward Splicing / Retropolation):</strong> ربط السلسلة القديمة بسنة الأساس الجديدة</li>
                <li><strong>الاستيفاء (Interpolation):</strong> توزيع الفجوة بين السلسلتين بمعدل ثابت</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📐 صيغة الربط الخلفي | Retropolation Formula")
        st.latex(r"Y_t^R = X_t \times \frac{Y_T}{X_T}")
        
        st.markdown("#### 📐 صيغة الاستيفاء الخطي | Linear Interpolation Formula")
        st.latex(r"Y_t^I = X_t \times \left(1 + \frac{Y_T/X_T - 1}{T} \times t\right)")
        
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ ملاحظة مهمة | Important Note</h4>
            <p>طريقة الإسقاط الخلفي (Retropolation) تحافظ على معدلات النمو السابقة لكنها قد تبالغ في تقدير المستويات التاريخية.</p>
            <p>طريقة الاستيفاء (Interpolation) قد ترفع معدلات النمو لكنها تعطي تقديرات أكثر معقولية للمستويات التاريخية.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[3]:
        st.markdown("### 📊 تطبيق عملي للربط | Practical Splicing Application")
        
        # Generate sample data
        np.random.seed(42)
        years_old = list(range(2000, 2011))
        years_new = list(range(2008, 2021))
        
        old_series = 100 * (1.03 ** np.arange(len(years_old)))
        new_series = 150 * (1.035 ** np.arange(len(years_new)))
        
        # Find overlap year (2010)
        overlap_idx_old = years_old.index(2010)
        overlap_idx_new = years_new.index(2010)
        
        # Retropolation
        ratio = new_series[overlap_idx_new] / old_series[overlap_idx_old]
        retropolated = old_series * ratio
        
        # Create combined series
        combined_years = list(range(2000, 2021))
        combined_retro = list(retropolated[:overlap_idx_old]) + list(new_series[overlap_idx_new-2:])
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('السلاسل الأصلية | Original Series', 
                                                            'السلسلة المربوطة | Spliced Series'))
        
        fig.add_trace(go.Scatter(x=years_old, y=old_series, name='السلسلة القديمة (Old)',
                                line=dict(color='#D2691E', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=years_new, y=new_series, name='السلسلة الجديدة (New)',
                                line=dict(color='#228B22', width=2)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=combined_years, y=combined_retro[:len(combined_years)], 
                                name='السلسلة المربوطة (Spliced)',
                                line=dict(color='#4169E1', width=3)), row=1, col=2)
        
        fig.update_layout(height=400, template='plotly_white',
                         plot_bgcolor='rgba(255,248,240,0.8)')
        st.plotly_chart(fig, use_container_width=True)

# Current and Constant Prices Section
elif "الأسعار الجارية والثابتة" in section:
    st.markdown('<div class="section-header">💰 الأسعار الجارية والثابتة | Current & Constant Prices</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📖 المفهوم | Concept", "📐 الصيغ | Formulas", "📊 التطبيق | Application"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h3>💵 الأسعار الجارية | Current Prices</h3>
                <p><strong>التعريف:</strong> القيم المحسوبة بأسعار السوق السائدة في كل فترة زمنية.</p>
                <p><strong>الخصائص:</strong></p>
                <ul>
                    <li>تعكس التغيرات في الأسعار والكميات معاً</li>
                    <li>تُسمى أيضاً "الأسعار الاسمية" (Nominal Prices)</li>
                    <li>تتأثر بالتضخم</li>
                </ul>
                <p><strong>الاستخدام:</strong> التقارير المالية، حسابات الضرائب</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h3>📊 الأسعار الثابتة | Constant Prices</h3>
                <p><strong>التعريف:</strong> القيم المعدلة باستخدام أسعار سنة أساس محددة لإزالة تأثير التضخم.</p>
                <p><strong>الخصائص:</strong></p>
                <ul>
                    <li>تعكس التغيرات الحقيقية في الكميات فقط</li>
                    <li>تُسمى أيضاً "الأسعار الحقيقية" (Real Prices)</li>
                    <li>خالية من تأثير التضخم</li>
                </ul>
                <p><strong>الاستخدام:</strong> تحليل النمو الاقتصادي، المقارنات الزمنية</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📊 مقارنة بصرية | Visual Comparison")
        
        # Generate sample data
        years = list(range(2010, 2024))
        inflation_rate = 0.03
        real_growth = 0.02
        
        nominal_gdp = [1000]
        real_gdp = [1000]
        
        for i in range(1, len(years)):
            nominal_gdp.append(nominal_gdp[-1] * (1 + real_growth + inflation_rate))
            real_gdp.append(real_gdp[-1] * (1 + real_growth))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=nominal_gdp, name='GDP بالأسعار الجارية (Nominal)',
                                line=dict(color='#D2691E', width=3)))
        fig.add_trace(go.Scatter(x=years, y=real_gdp, name='GDP بالأسعار الثابتة (Real)',
                                line=dict(color='#228B22', width=3)))
        
        fig.update_layout(
            title='الفرق بين GDP الاسمي والحقيقي | Nominal vs Real GDP',
            xaxis_title='السنة | Year',
            yaxis_title='القيمة | Value',
            template='plotly_white',
            plot_bgcolor='rgba(255,248,240,0.8)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.markdown("### 📐 صيغ التحويل | Conversion Formulas")
        
        st.markdown("#### 1️⃣ من الجاري إلى الثابت | Current to Constant")
        st.latex(r"\text{القيمة الحقيقية} = \frac{\text{القيمة الاسمية}}{\text{مؤشر الأسعار}} \times 100")
        
        st.markdown("#### 2️⃣ معامل انكماش GDP | GDP Deflator")
        st.latex(r"\text{GDP Deflator} = \frac{\text{Nominal GDP}}{\text{Real GDP}} \times 100")
        
        st.markdown("#### 3️⃣ معدل التضخم | Inflation Rate")
        st.latex(r"\pi_t = \frac{\text{Deflator}_t - \text{Deflator}_{t-1}}{\text{Deflator}_{t-1}} \times 100")
        
        st.markdown("### 📊 مقارنة مؤشرات الأسعار | Price Indices Comparison")
        
        indices_df = pd.DataFrame({
            'المؤشر | Index': ['GDP Deflator', 'CPI', 'PPI', 'PCE Deflator'],
            'النطاق | Coverage': ['كل السلع والخدمات المُنتَجة محلياً', 'سلة استهلاك الأسر', 'أسعار المنتجين', 'استهلاك الأفراد'],
            'الأوزان | Weights': ['متغيرة (Current)', 'ثابتة (Laspeyres)', 'ثابتة', 'متغيرة (Chain)'],
            'الاستخدام | Use': ['تحليل الاقتصاد الكلي', 'تعديل الأجور والمعاشات', 'تحليل تكاليف الإنتاج', 'سياسة Fed الأمريكي']
        })
        st.dataframe(indices_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class="detail-box">
            <h4>📌 الفرق بين GDP Deflator و CPI</h4>
            <table style="width:100%; border-collapse: collapse;">
                <tr style="background-color: #FFEFD5;">
                    <th style="padding: 10px; border: 1px solid #DEB887;">الخاصية</th>
                    <th style="padding: 10px; border: 1px solid #DEB887;">GDP Deflator</th>
                    <th style="padding: 10px; border: 1px solid #DEB887;">CPI</th>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #DEB887;">النطاق</td>
                    <td style="padding: 8px; border: 1px solid #DEB887;">كل الإنتاج المحلي</td>
                    <td style="padding: 8px; border: 1px solid #DEB887;">سلة استهلاك ثابتة</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #DEB887;">الواردات</td>
                    <td style="padding: 8px; border: 1px solid #DEB887;">❌ لا يشمل</td>
                    <td style="padding: 8px; border: 1px solid #DEB887;">✅ يشمل</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #DEB887;">الأوزان</td>
                    <td style="padding: 8px; border: 1px solid #DEB887;">متغيرة (Paasche)</td>
                    <td style="padding: 8px; border: 1px solid #DEB887;">ثابتة (Laspeyres)</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #DEB887;">تحيز الاستبدال</td>
                    <td style="padding: 8px; border: 1px solid #DEB887;">يقلل التضخم</td>
                    <td style="padding: 8px; border: 1px solid #DEB887;">يبالغ في التضخم</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="formula-box">
            <h4>📌 متى نستخدم كل منهما؟ | When to Use Each?</h4>
            <table style="width:100%; text-align: center;">
                <tr style="background-color: #FFEFD5;">
                    <th>الحالة | Case</th>
                    <th>النوع المناسب | Appropriate Type</th>
                </tr>
                <tr>
                    <td>مقارنة النمو الاقتصادي عبر الزمن</td>
                    <td>أسعار ثابتة (Constant)</td>
                </tr>
                <tr style="background-color: #FFF8DC;">
                    <td>حساب الإيرادات الضريبية</td>
                    <td>أسعار جارية (Current)</td>
                </tr>
                <tr>
                    <td>تحليل القوة الشرائية</td>
                    <td>أسعار ثابتة (Constant)</td>
                </tr>
                <tr style="background-color: #FFF8DC;">
                    <td>إعداد الميزانيات</td>
                    <td>أسعار جارية (Current)</td>
                </tr>
                <tr>
                    <td>المقارنات الدولية (PPP)</td>
                    <td>أسعار ثابتة مع تعديل PPP</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("### 📊 أداة التحويل التفاعلية | Interactive Conversion Tool")
        
        col1, col2 = st.columns(2)
        
        with col1:
            nominal_value = st.number_input("القيمة الاسمية | Nominal Value", value=1000.0)
            price_index = st.number_input("مؤشر الأسعار | Price Index", value=120.0)
            base_year = st.text_input("سنة الأساس | Base Year", "2015")
        
        with col2:
            real_value = (nominal_value / price_index) * 100
            st.metric("القيمة الحقيقية | Real Value", f"{real_value:,.2f}")
            st.metric("معامل التحويل | Conversion Factor", f"{100/price_index:.4f}")
            
            deflator_implied = (nominal_value / real_value) * 100 if real_value != 0 else 0
            st.metric("معامل الانكماش الضمني | Implied Deflator", f"{deflator_implied:.2f}")

# Log Transformations Section
elif "التحويلات اللوغاريتمية" in section:
    st.markdown('<div class="section-header">📈 التحويلات اللوغاريتمية | Logarithmic Transformations</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📖 المفهوم | Concept", "📐 الخصائص | Properties", "📊 التطبيق | Application", "⚠️ المحاذير | Cautions"])
    
    with tabs[0]:
        st.markdown("""
        <div class="info-box">
            <h3>🎯 لماذا نستخدم اللوغاريتم في الاقتصاد؟ | Why Use Logarithms in Economics?</h3>
            <ol>
                <li><strong>تحويل العلاقات المضاعفة إلى جمعية:</strong> يسهل التحليل الإحصائي والانحدار</li>
                <li><strong>تثبيت التباين (Variance Stabilization):</strong> يقلل من تأثير القيم المتطرفة والتغاير</li>
                <li><strong>تفسير المعاملات كمرونات:</strong> β في نموذج log-log = المرونة مباشرة</li>
                <li><strong>تقريب التوزيع الطبيعي:</strong> للبيانات ذات الالتواء الموجب (Skewness &gt; 0)</li>
                <li><strong>حساب معدلات النمو:</strong> الفرق اللوغاريتمي ≈ معدل النمو للقيم الصغيرة</li>
                <li><strong>تحويل النمو الأسي إلى خطي:</strong> يسهل تقدير الاتجاهات طويلة المدى</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📐 الصيغ الأساسية | Basic Formulas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.latex(r"\ln(Y) = \text{اللوغاريتم الطبيعي (الأساس } e \text{)}")
            st.latex(r"\log_{10}(Y) = \text{اللوغاريتم العشري (الأساس 10)}")
            st.latex(r"e = 2.71828... \text{ (ثابت أويلر)}")
        
        with col2:
            st.latex(r"\Delta \ln(Y) = \ln(Y_t) - \ln(Y_{t-1}) \approx \frac{Y_t - Y_{t-1}}{Y_{t-1}}")
            st.latex(r"g_{\text{exact}} = e^{\Delta \ln(Y)} - 1 = \frac{Y_t}{Y_{t-1}} - 1")
        
        st.markdown("""
        <div class="detail-box">
            <h4>📌 مثال: دالة الإنتاج Cobb-Douglas</h4>
            <p><strong>الصيغة الأصلية:</strong> Y = A · K<sup>α</sup> · L<sup>β</sup></p>
            <p><strong>بعد اللوغاريتم:</strong> ln(Y) = ln(A) + α·ln(K) + β·ln(L)</p>
            <p><strong>التفسير:</strong></p>
            <ul>
                <li>α = مرونة الإنتاج بالنسبة لرأس المال (زيادة K بـ 1% تزيد Y بـ α%)</li>
                <li>β = مرونة الإنتاج بالنسبة للعمل</li>
                <li>α + β = عوائد الحجم (= 1: ثابتة، &gt;1: متزايدة، &lt;1: متناقصة)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("### 📐 خصائص اللوغاريتم | Properties of Logarithm")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="formula-box">
                <h4>قواعد أساسية | Basic Rules</h4>
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"\ln(A \times B) = \ln(A) + \ln(B)")
            st.latex(r"\ln\left(\frac{A}{B}\right) = \ln(A) - \ln(B)")
            st.latex(r"\ln(A^n) = n \times \ln(A)")
            st.latex(r"\ln(e^x) = x, \quad e^{\ln(x)} = x")
        
        with col2:
            st.markdown("""
            <div class="formula-box">
                <h4>تطبيقات اقتصادية | Economic Applications</h4>
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"\ln(Y) = \ln(A) + \alpha\ln(K) + \beta\ln(L)")
            st.latex(r"\text{Growth Rate} \approx \ln(Y_t) - \ln(Y_{t-1})")
            st.latex(r"\text{Semi-elasticity} = \frac{\partial \ln(Y)}{\partial X} = \frac{1}{Y}\frac{\partial Y}{\partial X}")
        
        st.markdown("### 📊 نماذج الانحدار اللوغاريتمي | Log-Linear Regression Models")
        
        models_df = pd.DataFrame({
            'النموذج | Model': ['Log-Log', 'Log-Linear', 'Linear-Log', 'Linear'],
            'الصيغة | Formula': ['ln(Y) = α + β·ln(X)', 'ln(Y) = α + β·X', 'Y = α + β·ln(X)', 'Y = α + β·X'],
            'تفسير β | Interpretation': ['مرونة: Δ%Y/Δ%X', 'شبه مرونة: Δ%Y = 100β·ΔX', 'ΔY = β·Δ%X/100', 'ميل خطي: ΔY/ΔX'],
            'المثال | Example': ['دالة الطلب السعرية', 'أثر التعليم على ln(الأجر)', 'أثر ln(الدخل) على الاستهلاك', 'دالة الاستهلاك الكينزية']
        })
        st.dataframe(models_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class="detail-box">
            <h4>📌 دقة تقريب معدل النمو | Growth Rate Approximation Accuracy</h4>
            <table style="width:100%; border-collapse: collapse;">
                <tr style="background-color: #FFEFD5;">
                    <th style="padding: 8px; border: 1px solid #DEB887;">النمو الفعلي</th>
                    <th style="padding: 8px; border: 1px solid #DEB887;">Δln (التقريب)</th>
                    <th style="padding: 8px; border: 1px solid #DEB887;">الخطأ %</th>
                </tr>
                <tr><td style="padding: 5px; border: 1px solid #DEB887;">1%</td><td style="padding: 5px; border: 1px solid #DEB887;">0.995%</td><td style="padding: 5px; border: 1px solid #DEB887;">0.5%</td></tr>
                <tr><td style="padding: 5px; border: 1px solid #DEB887;">5%</td><td style="padding: 5px; border: 1px solid #DEB887;">4.88%</td><td style="padding: 5px; border: 1px solid #DEB887;">2.4%</td></tr>
                <tr><td style="padding: 5px; border: 1px solid #DEB887;">10%</td><td style="padding: 5px; border: 1px solid #DEB887;">9.53%</td><td style="padding: 5px; border: 1px solid #DEB887;">4.7%</td></tr>
                <tr><td style="padding: 5px; border: 1px solid #DEB887;">20%</td><td style="padding: 5px; border: 1px solid #DEB887;">18.23%</td><td style="padding: 5px; border: 1px solid #DEB887;">8.8%</td></tr>
                <tr><td style="padding: 5px; border: 1px solid #DEB887;">50%</td><td style="padding: 5px; border: 1px solid #DEB887;">40.55%</td><td style="padding: 5px; border: 1px solid #DEB887;">18.9%</td></tr>
            </table>
            <p style="margin-top: 10px;"><em>القاعدة: التقريب دقيق فقط عندما |g| &lt; 10%</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("### 📊 تطبيق عملي | Practical Application")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            n_points = st.slider("عدد النقاط | Number of Points", 10, 100, 50)
            growth_rate = st.slider("معدل النمو % | Growth Rate %", 1, 10, 3)
            volatility = st.slider("التقلب | Volatility", 0.01, 0.2, 0.05)
        
        with col2:
            np.random.seed(42)
            t = np.arange(n_points)
            y = 100 * np.exp((growth_rate/100) * t + volatility * np.cumsum(np.random.randn(n_points)))
            log_y = np.log(y)
            
            fig = make_subplots(rows=2, cols=2,
                               subplot_titles=('البيانات الأصلية | Original Data',
                                             'البيانات اللوغاريتمية | Log Data',
                                             'معدل النمو | Growth Rate',
                                             'التوزيع | Distribution'))
            
            fig.add_trace(go.Scatter(y=y, mode='lines', name='Y', 
                                    line=dict(color='#D2691E')), row=1, col=1)
            fig.add_trace(go.Scatter(y=log_y, mode='lines', name='ln(Y)',
                                    line=dict(color='#228B22')), row=1, col=2)
            fig.add_trace(go.Scatter(y=np.diff(log_y)*100, mode='lines', name='Growth %',
                                    line=dict(color='#4169E1')), row=2, col=1)
            fig.add_trace(go.Histogram(x=np.diff(log_y)*100, name='Distribution',
                                      marker_color='#CD853F'), row=2, col=2)
            
            fig.update_layout(height=500, showlegend=False, template='plotly_white',
                            plot_bgcolor='rgba(255,248,240,0.8)')
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ محاذير استخدام اللوغاريتم | Cautions When Using Logarithm</h3>
            <ol>
                <li><strong>القيم الصفرية:</strong> ln(0) غير معرف - استخدم IHS كبديل</li>
                <li><strong>القيم السالبة:</strong> ln(x) غير معرف لـ x ≤ 0</li>
                <li><strong>القيم القريبة من الصفر:</strong> قد تعطي نتائج متطرفة</li>
                <li><strong>إعادة التحويل:</strong> E[Y] ≠ exp(E[ln(Y)]) بسبب عدم المساواة</li>
                <li><strong>حجم العينة:</strong> يحتاج عينات كبيرة للتقديرات الدقيقة</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔧 بدائل اللوغاريتم | Alternatives to Logarithm")
        
        alternatives_df = pd.DataFrame({
            'البديل | Alternative': ['ln(x+1)', 'ln(x+c)', 'IHS(x)', 'Box-Cox', 'Cube Root'],
            'الصيغة | Formula': ['ln(x+1)', 'ln(x+c) where c>0', 'sinh⁻¹(θx)/θ', '(xᵟ-1)/λ', 'x^(1/3)'],
            'المميزات | Advantages': ['بسيط', 'مرن', 'يقبل الصفر والسالب', 'تحويل مثالي', 'متماثل'],
            'العيوب | Disadvantages': ['تحيز للقيم الصغيرة', 'اختيار c تعسفي', 'تفسير معقد', 'يحتاج تقدير λ', 'أقل شيوعاً']
        })
        st.dataframe(alternatives_df, use_container_width=True, hide_index=True)

# IHS Transformation Section
elif "تحويل IHS" in section:
    st.markdown('<div class="section-header">🔄 تحويل الجيب الزائدي العكسي | Inverse Hyperbolic Sine (IHS)</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📖 المفهوم | Concept", "📐 الصيغ | Formulas", "📊 المقارنة | Comparison", "⚠️ المحاذير | Cautions"])
    
    with tabs[0]:
        st.markdown("""
        <div class="info-box">
            <h3>🎯 ما هو تحويل IHS؟ | What is IHS Transformation?</h3>
            <p>تحويل الجيب الزائدي العكسي (Inverse Hyperbolic Sine - IHS أو arcsinh) هو بديل للوغاريتم الطبيعي يمكن تطبيقه على جميع القيم الحقيقية بما فيها الصفر والقيم السالبة.</p>
            <hr style="border-color: #DEB887;">
            <h4>📜 التاريخ | History</h4>
            <ul>
                <li>قدمه <strong>Johnson (1949)</strong> في سياق تحويلات التوزيعات الإحصائية</li>
                <li>طوره <strong>Burbidge, Magee & Robb (1988)</strong> للتطبيقات الاقتصادية</li>
                <li>وضح <strong>Bellemare & Wichman (2020)</strong> صيغ حساب المرونات</li>
                <li>حذر <strong>Aihounton & Henningsen (2021)</strong> من حساسية وحدة القياس</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="success-box">
                <h4>✅ مميزات IHS | Advantages</h4>
                <ul>
                    <li>يقبل القيم الصفرية: IHS(0) = 0</li>
                    <li>يقبل القيم السالبة: معرَّف لكل ℝ</li>
                    <li>متماثل حول الصفر: IHS(-x) = -IHS(x)</li>
                    <li>يقارب ln(2x) للقيم الكبيرة الموجبة</li>
                    <li>لا يحتاج تعديلات تعسفية مثل ln(x+1)</li>
                    <li>يحافظ على إشارة البيانات الأصلية</li>
                    <li>تحويل قابل للعكس بشكل دقيق</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-box">
                <h4>⚠️ عيوب وتحديات IHS | Disadvantages</h4>
                <ul>
                    <li><strong>حساس جداً لوحدة القياس:</strong> النتائج تختلف بين $ و $1000</li>
                    <li>تفسير المعاملات أكثر تعقيداً من ln</li>
                    <li>معامل θ يؤثر بشكل كبير على النتائج</li>
                    <li>أقل شيوعاً في الأدبيات</li>
                    <li>المرونات ليست ثابتة (تعتمد على القيم)</li>
                    <li>يحتاج حذراً عند إعادة التحويل</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="detail-box">
            <h4>📌 متى نستخدم IHS بدلاً من ln؟ | When to Use IHS?</h4>
            <ul>
                <li><strong>بيانات تحتوي أصفار:</strong> الدخل، الأرباح، المدخرات، التبرعات</li>
                <li><strong>بيانات تحتوي قيم سالبة:</strong> صافي الثروة، الميزان التجاري، المكاسب/الخسائر</li>
                <li><strong>تجنب الاستبعاد:</strong> ln(x+1) أو حذف الأصفار يسبب تحيز في التقديرات</li>
                <li><strong>نريد تفسيراً قريباً من المرونة:</strong> للقيم الكبيرة بما يكفي (|x| > 10)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("### 📐 الصيغة الرياضية | Mathematical Formula")
        
        st.latex(r"\text{IHS}(x, \theta) = \frac{\sinh^{-1}(\theta x)}{\theta} = \frac{\ln(\theta x + \sqrt{(\theta x)^2 + 1})}{\theta}")
        
        st.markdown("### 📐 التحويل العكسي | Inverse Transformation")
        
        st.latex(r"\text{IHS}^{-1}(y, \theta) = \frac{\sinh(\theta y)}{\theta}")
        
        st.markdown("### 📐 خصائص مهمة | Important Properties")
        
        st.latex(r"\lim_{x \to \infty} \text{IHS}(x, 1) = \ln(2x)")
        st.latex(r"\text{IHS}(0, \theta) = 0")
        st.latex(r"\text{IHS}(-x, \theta) = -\text{IHS}(x, \theta)")
    
    with tabs[2]:
        st.markdown("### 📊 مقارنة بين ln(x) و IHS(x)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            theta = st.slider("معامل θ | Theta Parameter", 0.1, 2.0, 1.0, 0.1)
            x_range = st.slider("نطاق x | X Range", 1, 100, 50)
        
        with col2:
            x = np.linspace(-x_range, x_range, 500)
            x_pos = x[x > 0]
            
            fig = go.Figure()
            
            # IHS
            fig.add_trace(go.Scatter(x=x, y=ihs_transform(x, theta), name='IHS(x)',
                                    line=dict(color='#D2691E', width=3)))
            
            # ln(x) for positive values
            fig.add_trace(go.Scatter(x=x_pos, y=np.log(x_pos), name='ln(x)',
                                    line=dict(color='#228B22', width=3, dash='dash')))
            
            # ln(x+1)
            fig.add_trace(go.Scatter(x=x[x>-1], y=np.log(x[x>-1]+1), name='ln(x+1)',
                                    line=dict(color='#4169E1', width=2, dash='dot')))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title=f'مقارنة التحويلات (θ = {theta})',
                xaxis_title='x',
                yaxis_title='f(x)',
                template='plotly_white',
                plot_bgcolor='rgba(255,248,240,0.8)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ محاذير مهمة عند استخدام IHS | Important Cautions</h3>
            <h4>1. حساسية وحدة القياس | Unit Sensitivity</h4>
            <p>نتائج IHS تتغير بشكل كبير مع تغيير وحدة القياس (دولار vs ألف دولار)</p>
            
            <h4>2. اختيار θ | Choosing θ</h4>
            <p>Aihounton & Henningsen (2021) يقترحون:</p>
            <ul>
                <li>تجربة وحدات قياس مختلفة</li>
                <li>استخدام R² للمقارنة</li>
                <li>اختبار الحساسية</li>
            </ul>
            
            <h4>3. تفسير المعاملات | Coefficient Interpretation</h4>
            <p>معامل β في نموذج IHS ≠ مرونة مباشرة</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📚 مراجع مهمة | Key References")
        st.markdown("""
        - Bellemare & Wichman (2020): "Elasticities and the Inverse Hyperbolic Sine Transformation"
        - Aihounton & Henningsen (2021): "Units of Measurement and the IHS Transformation"
        - Norton (2022): "The IHS Transformation and Retransformed Marginal Effects"
        """)

# Filtering Methods Section
elif "الترشيح" in section:
    st.markdown('<div class="section-header">📊 طرق الترشيح | Filtering Methods</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["HP Filter", "Hamilton Filter", "Boosted HP", "Baxter-King", "📊 المقارنة | Comparison"])
    
    with tabs[0]:
        st.markdown("### 🔷 مرشح هودريك-بريسكوت | Hodrick-Prescott Filter")
        
        st.markdown("""
        <div class="info-box">
            <h4>🎯 الهدف والتاريخ | Purpose and History</h4>
            <p>يفصل السلسلة الزمنية إلى مكونين: <strong>الاتجاه العام (Trend)</strong> و<strong>الدورة (Cycle)</strong></p>
            <p>طوره Robert Hodrick و Edward Prescott (الحائز على نوبل) عام 1997، رغم استخدامه منذ 1981. في الأصل اقترحه E.T. Whittaker عام 1923.</p>
            <p>يُعد أكثر طرق الترشيح استخداماً في البنوك المركزية وصندوق النقد الدولي والبنك الدولي.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r"\min_{\tau} \left\{ \sum_{t=1}^{T}(y_t - \tau_t)^2 + \lambda \sum_{t=2}^{T-1}[(\tau_{t+1} - \tau_t) - (\tau_t - \tau_{t-1})]^2 \right\}")
        
        st.markdown("""
        <div class="formula-box">
            <p><strong>حيث:</strong></p>
            <p>y<sub>t</sub> = القيمة الملاحظة | τ<sub>t</sub> = الاتجاه المُقدَّر | λ = معامل التنعيم (Smoothing Parameter)</p>
            <p>الحد الأول: يقيس قرب الاتجاه من البيانات (Goodness of Fit)</p>
            <p>الحد الثاني: يقيس نعومة الاتجاه - التغير في الميل (Smoothness Penalty)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ⚙️ معامل التنعيم λ | Smoothing Parameter")
        
        lambda_df = pd.DataFrame({
            'تردد البيانات | Frequency': ['سنوي | Annual', 'ربع سنوي | Quarterly', 'شهري | Monthly'],
            'قيمة λ المقترحة | Suggested λ': ['6.25 (Ravn-Uhlig) أو 100 (تقليدي)', '1600 (Hodrick-Prescott)', '129,600 (مشتق)'],
            'الصيغة | Formula': ['1600/4⁴ = 6.25', '1600', '1600 × 3⁴ = 129,600'],
            'فترة الدورة | Cycle Period': ['حوالي 10 سنوات', '32 ربع (8 سنوات)', '96 شهر (8 سنوات)']
        })
        st.dataframe(lambda_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ انتقادات HP Filter | Criticisms (Hamilton, 2018; Moura, 2024)</h4>
            <ol>
                <li><strong>الدورات الوهمية (Spurious Cycles):</strong> ينتج دورات منتظمة حتى من بيانات Random Walk عشوائية تماماً</li>
                <li><strong>تحيز نهاية العينة (End-point Bias):</strong> التقديرات في نهاية السلسلة أقل موثوقية وتتغير مع إضافة بيانات جديدة</li>
                <li><strong>اختيار λ تعسفي:</strong> قيمة 1600 ليس لها أساس نظري قوي، والتقديرات من البيانات تعطي قيم قريبة من 1</li>
                <li><strong>يفترض I(2):</strong> يفترض أن الاتجاه متكامل من الدرجة الثانية، وهو افتراض قد لا يناسب البيانات الفعلية</li>
                <li><strong>تنبؤ مُضلِّل:</strong> تغيير الماضي مع كل ملاحظة جديدة يجعله غير مناسب للتحليل في الوقت الحقيقي</li>
            </ol>
            <p style="margin-top: 15px;"><strong>ومع ذلك:</strong> يبقى HP Filter معياراً صناعياً ويستخدمه البنك الدولي وBIS لحساب فجوة الائتمان (Credit Gap) للتحذير من الأزمات المالية.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("### 🔶 مرشح هاميلتون | Hamilton Filter (2018)")
        
        st.markdown("""
        <div class="info-box">
            <h4>🎯 الفكرة الأساسية | Core Concept</h4>
            <p>اقترحه James Hamilton في ورقته الشهيرة "Why You Should Never Use the Hodrick-Prescott Filter" (Review of Economics and Statistics, 2018)</p>
            <p>يعتمد على انحدار ذاتي بدلاً من التنعيم، مما يتجنب العديد من مشاكل HP Filter.</p>
            <p><strong>الفكرة:</strong> بدلاً من تنعيم البيانات، نتنبأ بالقيمة بعد h فترات باستخدام p قيم سابقة، والبواقي تمثل الدورة.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r"y_{t+h} = \beta_0 + \beta_1 y_t + \beta_2 y_{t-1} + \beta_3 y_{t-2} + \beta_4 y_{t-3} + \epsilon_{t+h}")
        
        st.markdown("""
        <div class="formula-box">
            <p><strong>المعاملات الافتراضية | Default Parameters:</strong></p>
            <p><strong>h = 8</strong> للبيانات الربع سنوية (النظر سنتين للأمام)</p>
            <p><strong>p = 4</strong> عدد الفجوات الزمنية (4 أرباع = سنة)</p>
            <p><strong>الاتجاه (Trend):</strong> القيم المتوقعة من الانحدار: ŷ<sub>t</sub></p>
            <p><strong>الدورة (Cycle):</strong> البواقي: c<sub>t</sub> = y<sub>t</sub> - ŷ<sub>t</sub></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ⚙️ المعاملات حسب تردد البيانات | Parameters by Frequency")
        
        hamilton_params_df = pd.DataFrame({
            'التردد | Frequency': ['ربع سنوي | Quarterly', 'شهري | Monthly', 'سنوي | Annual'],
            'h (أفق التنبؤ)': ['8 أرباع (سنتان)', '24 شهر (سنتان)', '2 سنوات'],
            'p (عدد الفجوات)': ['4', '12', '1-2'],
            'الملاحظات المفقودة': ['h + p - 1 = 11', '35', '3-4']
        })
        st.dataframe(hamilton_params_df, use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="success-box">
                <h4>✅ مميزات Hamilton | Advantages</h4>
                <ul>
                    <li>لا ينتج دورات وهمية من Random Walk</li>
                    <li>لا يوجد تحيز في نهاية العينة</li>
                    <li>مبني على أسس إحصائية متينة (OLS)</li>
                    <li>تفسير اقتصادي واضح (التنبؤ)</li>
                    <li>لا يحتاج اختيار تعسفي لـ λ</li>
                    <li>الماضي لا يتغير مع بيانات جديدة</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-box">
                <h4>⚠️ عيوب Hamilton | Disadvantages (Moura, 2024)</h4>
                <ul>
                    <li>يفقد h+p-1 ملاحظات من بداية السلسلة</li>
                    <li>الاتجاه يتأخر h فترات عن البيانات</li>
                    <li>قد ينتج تقلبات أكبر في الدورة</li>
                    <li>اختيار h و p ليس موضوعياً تماماً</li>
                    <li>قد يعطي نتائج غريبة عند التحولات الحادة</li>
                    <li>ينتج أيضاً دورات من Random Walk (Moura)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="reference-box">
            <h4>📚 المراجع الأساسية | Key References</h4>
            <ul>
                <li>Hamilton, J.D. (2018). "Why You Should Never Use the Hodrick-Prescott Filter." <em>Review of Economics and Statistics</em>, 100(5), 831-843.</li>
                <li>Moura, A. (2024). "Why You Should Never Use the Hodrick-Prescott Filter: Comment." <em>Journal of Comments and Replications in Economics</em>.</li>
                <li>Hall, V.B. & Thomson, P. (2021). "Does Hamilton's OLS regression provide a better alternative?" <em>Journal of Business Cycle Research</em>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("### 🚀 مرشح HP المعزز | Boosted HP Filter")
        
        st.markdown("""
        <div class="info-box">
            <p>طوره Phillips & Shi (2021) كتحسين لـ HP Filter</p>
            <p>يطبق HP Filter بشكل متكرر (Iterative)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r"\text{Boosted HP: } \tau^{(m+1)} = \tau^{(m)} + \text{HP}(y - \tau^{(m)})")
        
        st.markdown("""
        <div class="success-box">
            <h4>✅ مميزات Boosted HP</h4>
            <ul>
                <li>يتعامل مع سلاسل I(1) و I(2)</li>
                <li>أكثر تكيفاً مع أنماط الاتجاه المختلفة</li>
                <li>يلتقط الدورات بشكل أفضل عند الأزمات</li>
                <li>قاعدة توقف تلقائية (Automatic stopping rule)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[3]:
        st.markdown("### 🔷 مرشح باكستر-كينغ | Baxter-King Band-Pass Filter")
        
        st.latex(r"c_t = \sum_{j=-K}^{K} b_j y_{t-j}")
        
        st.markdown("""
        <div class="info-box">
            <p>يستخرج المكونات ضمن نطاق ترددي محدد</p>
            <p>مثالي لاستخراج الدورات الاقتصادية (6-32 ربع سنوي)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[4]:
        st.markdown("### 📊 مقارنة تطبيقية | Practical Comparison")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            n_obs = st.slider("عدد الملاحظات", 50, 200, 100)
            lambda_hp = st.slider("λ (HP)", 100, 3200, 1600)
            trend_strength = st.slider("قوة الاتجاه", 0.01, 0.1, 0.03)
            cycle_amplitude = st.slider("سعة الدورة", 0.5, 3.0, 1.5)
        
        with col2:
            np.random.seed(42)
            t = np.arange(n_obs)
            trend_true = 100 + trend_strength * t
            cycle_true = cycle_amplitude * np.sin(2 * np.pi * t / 32)
            noise = 0.3 * np.random.randn(n_obs)
            y = trend_true + cycle_true + noise
            
            # Apply filters
            hp_trend, hp_cycle = hp_filter(y, lambda_hp)
            ham_trend, ham_cycle = hamilton_filter(y)
            bhp_trend, bhp_cycle = boosted_hp_filter(y, lambda_hp)
            bk_trend, bk_cycle = baxter_king_filter(y)
            
            fig = make_subplots(rows=2, cols=2,
                               subplot_titles=('البيانات والاتجاهات | Data & Trends',
                                             'الدورات | Cycles',
                                             'مقارنة الاتجاهات | Trend Comparison',
                                             'مقارنة الدورات | Cycle Comparison'))
            
            # Original data
            fig.add_trace(go.Scatter(y=y, name='البيانات', line=dict(color='gray', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(y=hp_trend, name='HP Trend', line=dict(color='#D2691E', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(y=bhp_trend, name='bHP Trend', line=dict(color='#228B22', width=2)), row=1, col=1)
            
            # Cycles
            fig.add_trace(go.Scatter(y=hp_cycle, name='HP Cycle', line=dict(color='#D2691E')), row=1, col=2)
            fig.add_trace(go.Scatter(y=ham_cycle, name='Hamilton Cycle', line=dict(color='#4169E1')), row=1, col=2)
            
            # Trend comparison
            fig.add_trace(go.Scatter(y=trend_true, name='True Trend', line=dict(color='black', dash='dash')), row=2, col=1)
            fig.add_trace(go.Scatter(y=hp_trend, name='HP', line=dict(color='#D2691E')), row=2, col=1)
            fig.add_trace(go.Scatter(y=bhp_trend, name='bHP', line=dict(color='#228B22')), row=2, col=1)
            
            # Cycle comparison
            fig.add_trace(go.Scatter(y=cycle_true, name='True Cycle', line=dict(color='black', dash='dash')), row=2, col=2)
            fig.add_trace(go.Scatter(y=hp_cycle, name='HP', line=dict(color='#D2691E')), row=2, col=2)
            fig.add_trace(go.Scatter(y=bk_cycle, name='BK', line=dict(color='#CD853F')), row=2, col=2)
            
            fig.update_layout(height=600, template='plotly_white',
                            plot_bgcolor='rgba(255,248,240,0.8)', showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

# Returns Calculation Section
elif "العوائد" in section:
    st.markdown('<div class="section-header">🔢 حساب العوائد | Returns Calculation</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📖 الأنواع | Types", "📐 الصيغ | Formulas", "📊 المقارنة | Comparison", "💻 التطبيق | Application"])
    
    with tabs[0]:
        st.markdown("### 📊 أنواع العوائد | Types of Returns")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>1️⃣ العائد البسيط</h4>
                <h5>Simple Return</h5>
                <p>التغير النسبي في السعر من فترة لأخرى</p>
                <p style="color: #228B22;">✅ سهل التفسير للمستثمرين</p>
                <p style="color: #228B22;">✅ صحيح لعوائد المحفظة</p>
                <p style="color: #CD5C5C;">❌ غير قابل للجمع عبر الزمن</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>2️⃣ العائد اللوغاريتمي</h4>
                <h5>Log Return (Continuously Compounded)</h5>
                <p>العائد المركب المستمر - يُستخدم في النمذجة المالية</p>
                <p style="color: #228B22;">✅ قابل للجمع عبر الزمن</p>
                <p style="color: #228B22;">✅ متماثل حول الصفر</p>
                <p style="color: #CD5C5C;">❌ غير صحيح لعوائد المحفظة</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-box">
                <h4>3️⃣ العائد الإجمالي</h4>
                <h5>Gross Return</h5>
                <p>نسبة السعر الحالي للسعر السابق (1 + r)</p>
                <p style="color: #228B22;">✅ قابل للضرب عبر الزمن</p>
                <p style="color: #228B22;">✅ مناسب للتراكم</p>
                <p style="color: #CD5C5C;">❌ يحتاج تحويل للتفسير</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="detail-box">
            <h4>📌 أنواع إضافية | Additional Types</h4>
            <table style="width:100%; border-collapse: collapse;">
                <tr style="background-color: #FFEFD5;">
                    <th style="padding: 10px; border: 1px solid #DEB887;">النوع | Type</th>
                    <th style="padding: 10px; border: 1px solid #DEB887;">الوصف | Description</th>
                    <th style="padding: 10px; border: 1px solid #DEB887;">الاستخدام | Use</th>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #DEB887;"><strong>العائد الزائد (Excess Return)</strong></td>
                    <td style="padding: 8px; border: 1px solid #DEB887;">العائد فوق المعدل الخالي من المخاطر: r - r<sub>f</sub></td>
                    <td style="padding: 8px; border: 1px solid #DEB887;">CAPM، تقييم الأداء</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #DEB887;"><strong>العائد المُعدَّل للمخاطر (Risk-Adjusted)</strong></td>
                    <td style="padding: 8px; border: 1px solid #DEB887;">Sharpe Ratio = (r - r<sub>f</sub>) / σ</td>
                    <td style="padding: 8px; border: 1px solid #DEB887;">مقارنة الاستثمارات</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #DEB887;"><strong>معدل النمو السنوي المركب (CAGR)</strong></td>
                    <td style="padding: 8px; border: 1px solid #DEB887;">(V<sub>end</sub>/V<sub>start</sub>)<sup>1/n</sup> - 1</td>
                    <td style="padding: 8px; border: 1px solid #DEB887;">أداء طويل المدى</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("### 📐 صيغ العوائد | Return Formulas")
        
        st.markdown("#### العائد البسيط | Simple Return")
        st.latex(r"r_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1")
        
        st.markdown("#### العائد اللوغاريتمي | Log Return")
        st.latex(r"R_t = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(P_t) - \ln(P_{t-1})")
        
        st.markdown("#### العائد الإجمالي | Gross Return")
        st.latex(r"G_t = \frac{P_t}{P_{t-1}} = 1 + r_t")
        
        st.markdown("#### العلاقة بين العوائد | Relationship")
        st.latex(r"R_t = \ln(1 + r_t) \approx r_t \text{ (للقيم الصغيرة)}")
        
        st.markdown("""
        <div class="detail-box">
            <h4>📌 دقة التقريب | Approximation Accuracy</h4>
            <table style="width:100%; border-collapse: collapse;">
                <tr style="background-color: #FFEFD5;">
                    <th style="padding: 8px; border: 1px solid #DEB887;">r (بسيط)</th>
                    <th style="padding: 8px; border: 1px solid #DEB887;">R = ln(1+r)</th>
                    <th style="padding: 8px; border: 1px solid #DEB887;">الفرق %</th>
                </tr>
                <tr><td style="padding: 5px; border: 1px solid #DEB887;">1%</td><td style="padding: 5px; border: 1px solid #DEB887;">0.995%</td><td style="padding: 5px; border: 1px solid #DEB887;">0.5%</td></tr>
                <tr><td style="padding: 5px; border: 1px solid #DEB887;">5%</td><td style="padding: 5px; border: 1px solid #DEB887;">4.88%</td><td style="padding: 5px; border: 1px solid #DEB887;">2.5%</td></tr>
                <tr><td style="padding: 5px; border: 1px solid #DEB887;">10%</td><td style="padding: 5px; border: 1px solid #DEB887;">9.53%</td><td style="padding: 5px; border: 1px solid #DEB887;">4.9%</td></tr>
                <tr><td style="padding: 5px; border: 1px solid #DEB887;">20%</td><td style="padding: 5px; border: 1px solid #DEB887;">18.23%</td><td style="padding: 5px; border: 1px solid #DEB887;">9.7%</td></tr>
                <tr><td style="padding: 5px; border: 1px solid #DEB887;">50%</td><td style="padding: 5px; border: 1px solid #DEB887;">40.55%</td><td style="padding: 5px; border: 1px solid #DEB887;">23.3%</td></tr>
            </table>
            <p style="margin-top: 10px;"><em>التقريب دقيق فقط للعوائد الصغيرة (&lt; 10%)</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### العائد التراكمي | Cumulative Return")
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r"r_{0:T} = \prod_{t=1}^{T}(1 + r_t) - 1")
            st.markdown("(العائد البسيط - بالضرب)")
        with col2:
            st.latex(r"R_{0:T} = \sum_{t=1}^{T} R_t = \ln(P_T) - \ln(P_0)")
            st.markdown("(العائد اللوغاريتمي - بالجمع)")
        
        st.markdown("#### معدل النمو السنوي المركب | CAGR")
        st.latex(r"CAGR = \left(\frac{V_{end}}{V_{start}}\right)^{\frac{1}{n}} - 1 = \exp\left(\frac{1}{n}\sum_{t=1}^{n} R_t\right) - 1")
        
        st.markdown("#### نسبة شارب | Sharpe Ratio")
        st.latex(r"SR = \frac{E[r] - r_f}{\sigma_r} = \frac{\text{العائد الزائد المتوقع}}{\text{الانحراف المعياري}}")
    
    with tabs[2]:
        st.markdown("### 📊 مقارنة بين أنواع العوائد")
        
        comparison_df = pd.DataFrame({
            'الخاصية | Property': [
                'قابلية الجمع عبر الزمن (Time Additivity)',
                'قابلية الجمع عبر الأصول (Cross-sectional)',
                'التماثل (Symmetry)',
                'سهولة التفسير الاقتصادي',
                'التوزيع الطبيعي (تقريباً)',
                'الاستقرار العددي',
                'مناسب لـ GARCH',
                'مناسب لعوائد المحفظة'
            ],
            'Simple Return': ['❌', '✅', '❌', '✅', '❌', '❌', '⚠️', '✅'],
            'Log Return': ['✅', '❌', '✅', '⚠️', '✅', '✅', '✅', '❌'],
            'Gross Return': ['✅ (ضرب)', '❌', '❌', '⚠️', '❌', '⚠️', '❌', '❌']
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>📌 متى نستخدم كل نوع؟ | When to Use Each Type?</h4>
            <ul>
                <li><strong>العائد البسيط:</strong> التقارير للمستثمرين، حساب عوائد المحفظة المرجحة، الأداء قصير المدى</li>
                <li><strong>العائد اللوغاريتمي:</strong> النمذجة الإحصائية، تحليل السلاسل الزمنية، نماذج GARCH وVAR، تقدير التقلب</li>
                <li><strong>العائد الإجمالي:</strong> حساب العوائد التراكمية، التركيب المضاعف، المقارنات طويلة المدى</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ تنبيهات مهمة | Important Notes</h4>
            <ul>
                <li><strong>عدم التماثل (Asymmetry):</strong> العائد البسيط +50% ثم -50% ≠ 0% (النتيجة = -25%)</li>
                <li><strong>تجميع المحفظة:</strong> r<sub>portfolio</sub> = Σw<sub>i</sub>r<sub>i</sub> للعوائد البسيطة فقط، لا يعمل مع اللوغاريتمية</li>
                <li><strong>العوائد الكبيرة:</strong> عند |r| &gt; 10%، الفرق بين البسيط واللوغاريتمي يصبح كبيراً</li>
                <li><strong>العوائد السالبة الكبيرة:</strong> r = -100% ممكن، لكن R = ln(0) غير معرَّف</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[3]:
        st.markdown("### 💻 حاسبة العوائد | Returns Calculator")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            prices_input = st.text_area("أدخل الأسعار (مفصولة بفاصلة)", "100, 102, 99, 105, 110, 108, 115")
            
            if st.button("🔢 احسب العوائد | Calculate Returns"):
                try:
                    prices = np.array([float(p.strip()) for p in prices_input.split(",")])
                    
                    simple_returns = simple_return(prices) * 100
                    log_returns = log_return(prices) * 100
                    gross_returns = gross_return(prices)
                    
                    results_df = pd.DataFrame({
                        'الفترة | Period': range(1, len(prices)),
                        'السعر | Price': prices[1:],
                        'العائد البسيط % | Simple': np.round(simple_returns, 4),
                        'العائد اللوغاريتمي % | Log': np.round(log_returns, 4),
                        'العائد الإجمالي | Gross': np.round(gross_returns, 4)
                    })
                    
                    st.session_state['returns_df'] = results_df
                    st.session_state['prices'] = prices
                    st.session_state['simple_returns'] = simple_returns
                    st.session_state['log_returns'] = log_returns
                    
                except Exception as e:
                    st.error(f"خطأ: {e}")
        
        with col2:
            if 'returns_df' in st.session_state:
                st.dataframe(st.session_state['returns_df'], use_container_width=True)
                
                fig = make_subplots(rows=1, cols=2, 
                                   subplot_titles=('الأسعار | Prices', 'العوائد | Returns'))
                
                fig.add_trace(go.Scatter(y=st.session_state['prices'], mode='lines+markers',
                                        name='Price', line=dict(color='#D2691E', width=2)), row=1, col=1)
                fig.add_trace(go.Bar(y=st.session_state['simple_returns'], name='Simple %',
                                    marker_color='#228B22'), row=1, col=2)
                fig.add_trace(go.Scatter(y=st.session_state['log_returns'], name='Log %',
                                        mode='lines', line=dict(color='#4169E1', width=2)), row=1, col=2)
                
                fig.update_layout(height=350, template='plotly_white',
                                plot_bgcolor='rgba(255,248,240,0.8)')
                st.plotly_chart(fig, use_container_width=True)

# Outliers Detection Section
elif "القيم الشاذة" in section:
    st.markdown('<div class="section-header">⚠️ اكتشاف القيم الشاذة | Outliers Detection</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📖 المفهوم | Concept", "🔍 الطرق | Methods", "📊 التطبيق | Application"])
    
    with tabs[0]:
        st.markdown("""
        <div class="info-box">
            <h3>🎯 ما هي القيم الشاذة؟ | What are Outliers?</h3>
            <p>القيم الشاذة هي نقاط بيانات تختلف بشكل كبير عن باقي الملاحظات. قد تنتج عن:</p>
            <ul>
                <li><strong>أخطاء في القياس أو الإدخال:</strong> أخطاء بشرية، أعطال أجهزة القياس</li>
                <li><strong>أحداث استثنائية:</strong> أزمات مالية (2008)، جائحة كورونا (2020)، كوارث طبيعية</li>
                <li><strong>تغيرات هيكلية:</strong> تغير السياسات الاقتصادية، انضمام لاتحادات</li>
                <li><strong>تنوع طبيعي:</strong> بعض القيم المتطرفة حقيقية وتحمل معلومات مهمة</li>
            </ul>
            <hr style="border-color: #DEB887;">
            <p><strong>⚠️ تنبيه:</strong> ليس كل قيمة شاذة خاطئة! يجب التحقق من سبب الشذوذ قبل حذفه أو تعديله.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 أنواع القيم الشاذة في السلاسل الزمنية | Types of Time Series Outliers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="warning-box">
                <h4>🔴 Additive Outlier (AO)</h4>
                <p><strong>الوصف:</strong> قفزة مؤقتة في نقطة واحدة فقط</p>
                <p><strong>المثال:</strong> خطأ إدخال بيانات، توقف مؤقت للإنتاج</p>
                <p><strong>التأثير:</strong> يؤثر على ملاحظة واحدة فقط</p>
                <p><strong>الصيغة:</strong> y<sub>t</sub> = y<sub>t</sub>* + ω·I(t=T)</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="warning-box">
                <h4>🟠 Level Shift (LS)</h4>
                <p><strong>الوصف:</strong> تغير دائم في مستوى السلسلة</p>
                <p><strong>المثال:</strong> تغير سياسة ضريبية، انهيار عملة</p>
                <p><strong>التأثير:</strong> يؤثر على جميع الملاحظات بعد T</p>
                <p><strong>الصيغة:</strong> y<sub>t</sub> = y<sub>t</sub>* + ω·I(t≥T)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-box">
                <h4>🟡 Temporary Change (TC)</h4>
                <p><strong>الوصف:</strong> تغير مؤقت يتلاشى تدريجياً</p>
                <p><strong>المثال:</strong> إضراب عمال، كارثة طبيعية</p>
                <p><strong>التأثير:</strong> يتناقص بمعامل δ عبر الزمن</p>
                <p><strong>الصيغة:</strong> y<sub>t</sub> = y<sub>t</sub>* + ω·δ<sup>t-T</sup>·I(t≥T)</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="warning-box">
                <h4>🟢 Innovational Outlier (IO)</h4>
                <p><strong>الوصف:</strong> صدمة تؤثر عبر آلية السلسلة</p>
                <p><strong>المثال:</strong> صدمة نفطية، أزمة مالية</p>
                <p><strong>التأثير:</strong> ينتشر عبر بنية ARIMA</p>
                <p><strong>الصيغة:</strong> ε<sub>t</sub> = ε<sub>t</sub>* + ω·I(t=T)</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("### 🔍 طرق الكشف | Detection Methods")
        
        st.markdown("#### 1️⃣ طريقة Z-Score (الدرجة المعيارية)")
        st.latex(r"z_i = \frac{x_i - \bar{x}}{\sigma}")
        st.markdown("""
        <div class="detail-box">
            <p><strong>القاعدة:</strong> القيمة شاذة إذا |z| &gt; 3 (أو 2.5 في بعض الحالات)</p>
            <p><strong>المزايا:</strong> بسيطة، سهلة التطبيق</p>
            <p><strong>العيوب:</strong> حساسة للقيم الشاذة نفسها (المتوسط والانحراف المعياري يتأثران)</p>
            <p><strong>الافتراض:</strong> البيانات تتبع توزيعاً طبيعياً تقريباً</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 2️⃣ طريقة IQR (المدى الربيعي)")
        st.latex(r"\text{الحد الأدنى} = Q_1 - k \times IQR")
        st.latex(r"\text{الحد الأعلى} = Q_3 + k \times IQR")
        st.markdown("""
        <div class="detail-box">
            <p><strong>حيث:</strong> IQR = Q₃ - Q₁ (المدى بين الربيع الأول والثالث)</p>
            <p><strong>k = 1.5:</strong> قيم شاذة معتدلة (Mild Outliers)</p>
            <p><strong>k = 3.0:</strong> قيم شاذة متطرفة (Extreme Outliers)</p>
            <p><strong>المزايا:</strong> مقاومة (Robust) - لا تتأثر كثيراً بالقيم الشاذة</p>
            <p><strong>العيوب:</strong> قد تفوت قيم شاذة في التوزيعات غير المتماثلة</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 3️⃣ طريقة MAD (الانحراف المطلق عن الوسيط)")
        st.latex(r"MAD = \text{median}(|x_i - \text{median}(x)|)")
        st.latex(r"M_i = \frac{0.6745 \times (x_i - \text{median}(x))}{MAD}")
        st.markdown("""
        <div class="detail-box">
            <p><strong>القاعدة:</strong> القيمة شاذة إذا |M<sub>i</sub>| &gt; 3.5</p>
            <p><strong>الثابت 0.6745:</strong> يجعل MAD مكافئاً للانحراف المعياري في التوزيع الطبيعي</p>
            <p><strong>المزايا:</strong> أكثر مقاومة من Z-Score، تعمل جيداً مع التوزيعات غير الطبيعية</p>
            <p><strong>العيوب:</strong> قد تكون محافظة جداً في بعض الحالات</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 مقارنة الطرق | Methods Comparison")
        
        methods_df = pd.DataFrame({
            'الطريقة | Method': ['Z-Score', 'IQR', 'MAD', 'DBSCAN', 'Isolation Forest', 'LOF'],
            'المقاومة | Robustness': ['❌ ضعيفة', '✅ جيدة', '✅✅ ممتازة', '✅ جيدة', '✅✅ ممتازة', '✅ جيدة'],
            'السرعة | Speed': ['✅✅ سريعة جداً', '✅✅ سريعة جداً', '✅✅ سريعة جداً', '⚠️ متوسطة', '✅ سريعة', '⚠️ متوسطة'],
            'متعدد الأبعاد | Multivariate': ['❌', '❌', '❌', '✅', '✅', '✅'],
            'الاستخدام الأمثل | Best For': [
                'بيانات طبيعية',
                'بيانات عامة',
                'بيانات ذات توزيع غير معروف',
                'تجمعات غير منتظمة',
                'بيانات عالية الأبعاد',
                'كثافة محلية'
            ]
        })
        st.dataframe(methods_df, use_container_width=True, hide_index=True)
    
    with tabs[2]:
        st.markdown("### 📊 أداة تفاعلية للكشف عن القيم الشاذة")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            n_points = st.slider("عدد النقاط", 50, 200, 100)
            n_outliers = st.slider("عدد القيم الشاذة", 0, 10, 3)
            outlier_magnitude = st.slider("حجم الشذوذ", 2, 10, 5)
            method = st.selectbox("طريقة الكشف", ["Z-Score", "IQR"])
            threshold = st.slider("عتبة الكشف", 1.5, 4.0, 3.0, 0.1)
        
        with col2:
            np.random.seed(42)
            data = np.random.randn(n_points) * 10 + 100
            
            # Add outliers
            outlier_idx = np.random.choice(n_points, n_outliers, replace=False)
            data[outlier_idx] += np.random.choice([-1, 1], n_outliers) * outlier_magnitude * 10
            
            # Detect outliers
            if method == "Z-Score":
                detected = detect_outliers_zscore(data, threshold)
            else:
                detected = detect_outliers_iqr(data, threshold)
            
            # Create colors
            colors = ['#D2691E' if i in detected else '#228B22' for i in range(len(data))]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=data, mode='markers',
                                    marker=dict(color=colors, size=10)))
            
            # Add threshold lines for Z-score
            if method == "Z-Score":
                mean_val = np.mean(data)
                std_val = np.std(data)
                fig.add_hline(y=mean_val + threshold * std_val, line_dash="dash", 
                             line_color="red", annotation_text="Upper Threshold")
                fig.add_hline(y=mean_val - threshold * std_val, line_dash="dash",
                             line_color="red", annotation_text="Lower Threshold")
            
            fig.update_layout(
                title=f'القيم الشاذة المكتشفة: {len(detected)} | Detected Outliers',
                xaxis_title='Index',
                yaxis_title='Value',
                template='plotly_white',
                plot_bgcolor='rgba(255,248,240,0.8)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"**القيم الشاذة في المواضع:** {list(detected)}")

# Missing Values Section
elif "القيم المفقودة" in section:
    st.markdown('<div class="section-header">❓ معالجة القيم المفقودة | Missing Values Treatment</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📖 الأنواع | Types", "🔧 الطرق | Methods", "📊 التطبيق | Application"])
    
    with tabs[0]:
        st.markdown("### 📊 أنواع القيم المفقودة | Types of Missing Data (Rubin, 1976)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>MCAR</h4>
                <h5>Missing Completely at Random</h5>
                <p>الفقدان عشوائي تماماً ولا يرتبط بأي متغير ملاحظ أو غير ملاحظ</p>
                <hr style="border-color: #DEB887;">
                <p><strong>مثال:</strong> فقدان بيانات بسبب عطل تقني عشوائي</p>
                <p><strong>الاختبار:</strong> Little's MCAR Test</p>
                <p style="color: #228B22;">✅ جميع طرق المعالجة صالحة</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>MAR</h4>
                <h5>Missing at Random</h5>
                <p>الفقدان يرتبط بمتغيرات ملاحظة أخرى لكن ليس بالقيمة المفقودة</p>
                <hr style="border-color: #DEB887;">
                <p><strong>مثال:</strong> الذكور أقل احتمالاً للإبلاغ عن الدخل</p>
                <p><strong>الاختبار:</strong> مقارنة أنماط الفقدان عبر المجموعات</p>
                <p style="color: #DAA520;">⚠️ يحتاج Multiple Imputation أو ML</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-box">
                <h4>MNAR</h4>
                <h5>Missing Not at Random</h5>
                <p>الفقدان يرتبط بالقيمة المفقودة نفسها (غير قابل للتجاهل)</p>
                <hr style="border-color: #DEB887;">
                <p><strong>مثال:</strong> ذوو الدخل العالي لا يفصحون عن دخلهم</p>
                <p><strong>الاختبار:</strong> لا يوجد اختبار قاطع - يحتاج معرفة مسبقة</p>
                <p style="color: #CD5C5C;">❌ يحتاج نماذج selection معقدة</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="detail-box">
            <h4>📌 كيف نحدد نوع الفقدان؟ | How to Identify Missing Type?</h4>
            <ol>
                <li><strong>فحص أنماط الفقدان:</strong> هل الفقدان عشوائي أم مرتبط بمتغيرات معينة؟</li>
                <li><strong>اختبار Little's MCAR:</strong> إذا p-value &gt; 0.05 فالبيانات MCAR</li>
                <li><strong>المقارنة بين المجموعات:</strong> هل خصائص الملاحظات الكاملة تختلف عن غير الكاملة؟</li>
                <li><strong>المعرفة المسبقة:</strong> ما آلية جمع البيانات؟ ما أسباب الفقدان المحتملة؟</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("### 🔧 طرق المعالجة | Treatment Methods")
        
        methods_df = pd.DataFrame({
            'الطريقة | Method': [
                'الحذف الكامل (Listwise)',
                'الحذف الجزئي (Pairwise)',
                'المتوسط/الوسيط',
                'الاستيفاء الخطي (Linear)',
                'الاستيفاء التكعيبي (Spline)',
                'LOCF/NOCB',
                'Multiple Imputation (MICE)',
                'KNN Imputation',
                'الاستيفاء بالتوقع (EM)'
            ],
            'الوصف | Description': [
                'حذف جميع الصفوف ذات القيم المفقودة',
                'استخدام كل البيانات المتاحة لكل تحليل',
                'استبدال بالمتوسط أو الوسيط العام أو المجموعة',
                'رسم خط مستقيم بين النقاط المتاحة',
                'منحنى ناعم يمر بالنقاط',
                'القيمة السابقة (LOCF) أو التالية (NOCB)',
                'توليد m مجموعات بيانات مكتملة وتجميع النتائج',
                'التقدير بناءً على k أقرب ملاحظات مشابهة',
                'تقدير القيم باستخدام خوارزمية Expectation-Maximization'
            ],
            'الأفضل لـ | Best For': [
                'MCAR فقط، عينات كبيرة',
                'MCAR، تحليلات متعددة',
                'MCAR، نسبة فقدان قليلة',
                'سلاسل زمنية، اتجاه واضح',
                'سلاسل زمنية، أنماط معقدة',
                'سلاسل زمنية، panel data',
                'MAR، تحليل إحصائي دقيق',
                'MAR، بيانات متعددة المتغيرات',
                'MAR، توزيعات طبيعية'
            ]
        })
        st.dataframe(methods_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ تنبيهات مهمة | Important Warnings</h4>
            <ul>
                <li><strong>لا تستخدم الحذف مع MNAR:</strong> سيؤدي لتحيز شديد في النتائج</li>
                <li><strong>الاستبدال بالمتوسط يقلل التباين:</strong> قد يؤدي لتقدير خاطئ للعلاقات</li>
                <li><strong>LOCF قد ينتج قيم غير واقعية:</strong> خاصة مع سلاسل متقلبة</li>
                <li><strong>Multiple Imputation يحتاج m ≥ 20:</strong> للحصول على تقديرات موثوقة لعدم اليقين</li>
                <li><strong>KNN حساس للتطبيع:</strong> طبّع البيانات قبل الاستخدام</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
            <h4>✅ دليل اختيار الطريقة | Method Selection Guide</h4>
            <ul>
                <li><strong>نسبة فقدان &lt; 5% و MCAR:</strong> الحذف أو المتوسط مقبول</li>
                <li><strong>سلسلة زمنية:</strong> الاستيفاء الخطي أو Spline</li>
                <li><strong>بيانات panel:</strong> LOCF أو الاستيفاء داخل المجموعة</li>
                <li><strong>تحليل إحصائي دقيق:</strong> Multiple Imputation (MICE)</li>
                <li><strong>Machine Learning:</strong> KNN أو MissForest</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("### 📊 أداة تفاعلية للمعالجة")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            n_points = st.slider("عدد النقاط", 30, 100, 50, key="missing_n")
            missing_pct = st.slider("نسبة الفقدان %", 5, 30, 15)
            method = st.selectbox("طريقة المعالجة", 
                                 ["المتوسط | Mean", "الوسيط | Median", 
                                  "الاستيفاء الخطي | Linear", "القيمة السابقة | LOCF"])
        
        with col2:
            np.random.seed(42)
            t = np.arange(n_points)
            y_true = 100 + 0.5 * t + 5 * np.sin(2 * np.pi * t / 20) + np.random.randn(n_points) * 2
            
            # Create missing values
            y_missing = y_true.copy()
            n_missing = int(n_points * missing_pct / 100)
            missing_idx = np.random.choice(range(1, n_points-1), n_missing, replace=False)
            y_missing[missing_idx] = np.nan
            
            # Impute
            y_imputed = y_missing.copy()
            if "المتوسط" in method:
                y_imputed[np.isnan(y_imputed)] = np.nanmean(y_missing)
            elif "الوسيط" in method:
                y_imputed[np.isnan(y_imputed)] = np.nanmedian(y_missing)
            elif "الاستيفاء" in method:
                # Linear interpolation
                mask = ~np.isnan(y_missing)
                y_imputed = np.interp(t, t[mask], y_missing[mask])
            else:  # LOCF
                for i in range(1, len(y_imputed)):
                    if np.isnan(y_imputed[i]):
                        y_imputed[i] = y_imputed[i-1]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=y_true, mode='lines', name='القيم الحقيقية',
                                    line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=t[~np.isnan(y_missing)], y=y_missing[~np.isnan(y_missing)],
                                    mode='markers', name='القيم الملاحظة',
                                    marker=dict(color='#228B22', size=8)))
            fig.add_trace(go.Scatter(x=t, y=y_imputed, mode='lines', name='بعد المعالجة',
                                    line=dict(color='#D2691E', width=2)))
            fig.add_trace(go.Scatter(x=t[missing_idx], y=y_imputed[missing_idx],
                                    mode='markers', name='القيم المُقدَّرة',
                                    marker=dict(color='#FF6347', size=10, symbol='x')))
            
            fig.update_layout(
                title='معالجة القيم المفقودة',
                xaxis_title='الفترة',
                yaxis_title='القيمة',
                template='plotly_white',
                plot_bgcolor='rgba(255,248,240,0.8)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate error
            rmse = np.sqrt(np.mean((y_true[missing_idx] - y_imputed[missing_idx])**2))
            st.metric("RMSE للقيم المُقدَّرة", f"{rmse:.3f}")

# Frequency Conversion Section
elif "تحويل التردد" in section:
    st.markdown('<div class="section-header">📆 تحويل تردد البيانات | Frequency Conversion</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📖 المفهوم | Concept", "📐 الطرق | Methods", "📊 التطبيق | Application"])
    
    with tabs[0]:
        st.markdown("""
        <div class="info-box">
            <h3>🎯 ما هو تحويل التردد؟ | What is Frequency Conversion?</h3>
            <p><strong>التفكيك الزمني (Temporal Disaggregation):</strong> تحويل من تردد منخفض إلى مرتفع (مثل: سنوي → ربع سنوي → شهري)</p>
            <p><strong>التجميع الزمني (Temporal Aggregation):</strong> تحويل من تردد مرتفع إلى منخفض (مثل: شهري → ربع سنوي → سنوي)</p>
            <hr style="border-color: #DEB887;">
            <h4>📌 الأهمية | Importance</h4>
            <ul>
                <li>توفير بيانات ربع سنوية أو شهرية للناتج المحلي الإجمالي عندما لا تتوفر إلا سنوية</li>
                <li>توحيد تردد المتغيرات لنماذج VAR والتكامل المشترك</li>
                <li>إنتاج تقديرات مبكرة (Flash Estimates) للمؤشرات الاقتصادية</li>
                <li>ملء الفجوات في السلاسل الزمنية التاريخية</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 حالات الاستخدام | Use Cases")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="success-box">
                <h4>⬆️ التفكيك (Disaggregation)</h4>
                <ul>
                    <li>توفير بيانات ربع سنوية للناتج المحلي من بيانات سنوية</li>
                    <li>تقدير مؤشرات شهرية من بيانات ربع سنوية</li>
                    <li>توحيد تردد المتغيرات لنموذج VAR</li>
                    <li>بناء سلاسل زمنية تاريخية طويلة</li>
                    <li>إنتاج تقديرات مبكرة (Nowcasting)</li>
                </ul>
                <p><em>الطرق: Chow-Lin, Denton, Fernandez, Litterman</em></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="success-box">
                <h4>⬇️ التجميع (Aggregation)</h4>
                <ul>
                    <li>حساب المتوسطات أو المجاميع السنوية</li>
                    <li>تقليل الضوضاء والتقلبات قصيرة المدى</li>
                    <li>توحيد الفترات للمقارنة الدولية</li>
                    <li>تبسيط التحليل للعرض</li>
                    <li>حساب مؤشرات طويلة المدى</li>
                </ul>
                <p><em>الطرق: Sum (للتدفقات), Average (للمخزونات), Last (للأرصدة)</em></p>
            </div>
            """, unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("### 🔧 طرق التفكيك الزمني | Disaggregation Methods")
        
        st.markdown("#### 1️⃣ طريقة Denton (1971)")
        st.latex(r"\min_y \sum_{t=2}^{n} \left(\frac{y_t}{p_t} - \frac{y_{t-1}}{p_{t-1}}\right)^2")
        st.markdown("""
        <div class="detail-box">
            <p><strong>الفكرة:</strong> تقليل التغيرات في النسبة بين السلسلة المُقدَّرة والمؤشر (Movement Preservation)</p>
            <p><strong>الميزة:</strong> يمكن العمل بدون مؤشر عالي التردد</p>
            <p><strong>المتغيرات:</strong> Denton-Cholette (نسخة محسنة تزيل التحيز في نهاية السلسلة)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 2️⃣ طريقة Chow-Lin (1971)")
        st.latex(r"y = X\beta + u, \quad u \sim AR(1): u_t = \rho u_{t-1} + \epsilon_t")
        st.markdown("""
        <div class="detail-box">
            <p><strong>الفكرة:</strong> انحدار GLS يربط البيانات منخفضة التردد بمؤشرات عالية التردد</p>
            <p><strong>الافتراض:</strong> أخطاء AR(1) مع تقدير ρ من البيانات</p>
            <p><strong>الأفضل لـ:</strong> سلاسل مستقرة أو متكاملة مشتركة مع المؤشر</p>
            <p><strong>المتطلب:</strong> يحتاج مؤشر عالي التردد مرتبط بالسلسلة الهدف</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 3️⃣ طريقة Fernandez (1981)")
        st.latex(r"y = X\beta + u, \quad u \sim I(1): u_t = u_{t-1} + \epsilon_t")
        st.markdown("""
        <div class="detail-box">
            <p><strong>الفكرة:</strong> حالة خاصة من Chow-Lin مع ρ = 1 (Random Walk)</p>
            <p><strong>الأفضل لـ:</strong> متغيرات التدفق (Flow Variables) مثل GDP، الاستهلاك، الاستثمار</p>
            <p><strong>الميزة:</strong> لا يحتاج تقدير ρ - أبسط حسابياً</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 4️⃣ طريقة Litterman (1983)")
        st.latex(r"\Delta y = X\Delta\beta + \epsilon, \quad \epsilon \sim AR(1)")
        st.markdown("""
        <div class="detail-box">
            <p><strong>الفكرة:</strong> Random Walk للمستويات + Markov للتغيرات</p>
            <p><strong>الأفضل لـ:</strong> متغيرات المخزون (Stock Variables) مثل العمالة، رأس المال</p>
            <p><strong>الميزة:</strong> أكثر مرونة في التعامل مع السلاسل غير المستقرة</p>
        </div>
        """, unsafe_allow_html=True)
        
        methods_comparison = pd.DataFrame({
            'الطريقة | Method': ['Denton', 'Denton-Cholette', 'Chow-Lin', 'Fernandez', 'Litterman'],
            'يحتاج مؤشر؟': ['اختياري', 'اختياري', 'نعم', 'نعم', 'نعم'],
            'افتراض الأخطاء': ['- (تحسين فقط)', '- (تحسين فقط)', 'AR(1), |ρ|<1', 'I(1), ρ=1', 'RW + AR(1)'],
            'نوع المتغير': ['أي نوع', 'أي نوع', 'I(0) أو CI', 'تدفقات (Flow)', 'مخزونات (Stock)'],
            'الاستخدام الأمثل': ['الحفاظ على الحركة', 'التنعيم', 'سلاسل مستقرة', 'GDP, استهلاك', 'عمالة, رأس مال']
        })
        st.dataframe(methods_comparison, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class="reference-box">
            <h4>📚 المراجع الأساسية | Key References</h4>
            <ul>
                <li>Chow, G.C. & Lin, A. (1971). "Best Linear Unbiased Interpolation, Distribution, and Extrapolation of Time Series by Related Series." <em>Review of Economics and Statistics</em>.</li>
                <li>Denton, F.T. (1971). "Adjustment of Monthly or Quarterly Series to Annual Totals." <em>JASA</em>.</li>
                <li>Fernandez, R.B. (1981). "A Methodological Note on the Estimation of Time Series." <em>Review of Economics and Statistics</em>.</li>
                <li>Litterman, R.B. (1983). "A Random Walk, Markov Model for the Distribution of Time Series." <em>JBES</em>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("### 📊 تطبيق عملي")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            n_years = st.slider("عدد السنوات", 5, 20, 10)
            method = st.selectbox("طريقة التفكيك", ["Denton (بدون مؤشر)", "Chow-Lin (مع مؤشر)"])
            
        with col2:
            np.random.seed(42)
            
            # Generate annual data
            years = list(range(2010, 2010 + n_years))
            annual_gdp = 1000 * (1.03 ** np.arange(n_years)) + np.random.randn(n_years) * 20
            
            # Generate quarterly indicator (if needed)
            n_quarters = n_years * 4
            quarters = [f"{y}Q{q}" for y in years for q in range(1, 5)]
            quarterly_indicator = np.repeat(annual_gdp / 4, 4) * (1 + 0.1 * np.random.randn(n_quarters))
            
            if "Denton" in method:
                quarterly_gdp = denton_disaggregate(annual_gdp, 4)
            else:
                quarterly_gdp = chow_lin_disaggregate(annual_gdp, quarterly_indicator, 4)
            
            fig = make_subplots(rows=2, cols=1,
                               subplot_titles=('البيانات السنوية | Annual Data',
                                             'البيانات الربع سنوية المُقدَّرة | Estimated Quarterly Data'))
            
            fig.add_trace(go.Bar(x=years, y=annual_gdp, name='GDP السنوي',
                                marker_color='#D2691E'), row=1, col=1)
            fig.add_trace(go.Scatter(x=quarters, y=quarterly_gdp, name='GDP الربع سنوي',
                                    line=dict(color='#228B22', width=2)), row=2, col=1)
            
            fig.update_layout(height=500, template='plotly_white',
                            plot_bgcolor='rgba(255,248,240,0.8)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Verify consistency
            annual_from_quarterly = [np.sum(quarterly_gdp[i*4:(i+1)*4]) for i in range(n_years)]
            consistency = np.allclose(annual_gdp, annual_from_quarterly, rtol=0.01)
            
            if consistency:
                st.success("✅ التحقق من الاتساق: مجموع الأرباع = السنوي")
            else:
                st.warning("⚠️ هناك فرق بين المجموع الربع سنوي والقيمة السنوية")

# Additional Tools Section
elif "أدوات إضافية" in section:
    st.markdown('<div class="section-header">🛠️ أدوات إضافية | Additional Tools</div>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📊 Stationarity", "📈 Seasonality", "🔢 Normalization", "📐 Growth Rates"])
    
    with tabs[0]:
        st.markdown("### 📊 اختبارات الاستقرارية | Stationarity Tests")
        
        st.markdown("""
        <div class="info-box">
            <h4>اختبار ADF (Augmented Dickey-Fuller)</h4>
            <p>يختبر وجود جذر وحدة (Unit Root)</p>
            <ul>
                <li>H₀: السلسلة غير مستقرة (Unit Root)</li>
                <li>H₁: السلسلة مستقرة (Stationary)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r"\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \epsilon_t")
        
        st.markdown("""
        <div class="info-box">
            <h4>اختبار KPSS</h4>
            <p>يختبر الاستقرارية حول اتجاه أو متوسط</p>
            <ul>
                <li>H₀: السلسلة مستقرة</li>
                <li>H₁: السلسلة غير مستقرة</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("### 📈 التعديل الموسمي | Seasonal Adjustment")
        
        st.markdown("""
        <div class="formula-box">
            <h4>نموذج التحليل الموسمي</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**النموذج الجمعي | Additive:**")
            st.latex(r"Y_t = T_t + S_t + C_t + I_t")
        
        with col2:
            st.markdown("**النموذج الضربي | Multiplicative:**")
            st.latex(r"Y_t = T_t \times S_t \times C_t \times I_t")
        
        st.markdown("""
        <div class="info-box">
            <h4>المكونات | Components</h4>
            <ul>
                <li><strong>T:</strong> الاتجاه العام (Trend)</li>
                <li><strong>S:</strong> المكون الموسمي (Seasonal)</li>
                <li><strong>C:</strong> الدورة الاقتصادية (Cycle)</li>
                <li><strong>I:</strong> المكون غير المنتظم (Irregular)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("### 🔢 طرق التطبيع | Normalization Methods")
        
        normalization_df = pd.DataFrame({
            'الطريقة | Method': ['Min-Max', 'Z-Score', 'Robust Scaling', 'Log Normalization'],
            'الصيغة | Formula': [
                '(x - min) / (max - min)',
                '(x - μ) / σ',
                '(x - median) / IQR',
                'log(x) / log(max)'
            ],
            'النطاق | Range': ['[0, 1]', '(-∞, +∞)', 'Variable', '[0, 1]'],
            'الاستخدام | Use Case': [
                'البيانات ذات الحدود المعروفة',
                'البيانات الطبيعية تقريباً',
                'وجود قيم شاذة',
                'البيانات ذات الالتواء الموجب'
            ]
        })
        st.dataframe(normalization_df, use_container_width=True, hide_index=True)
    
    with tabs[3]:
        st.markdown("### 📐 أنواع معدلات النمو | Growth Rate Types")
        
        st.markdown("#### 1️⃣ معدل النمو السنوي البسيط")
        st.latex(r"g = \frac{Y_t - Y_{t-1}}{Y_{t-1}} \times 100")
        
        st.markdown("#### 2️⃣ معدل النمو السنوي المركب (CAGR)")
        st.latex(r"CAGR = \left(\frac{Y_T}{Y_0}\right)^{\frac{1}{T}} - 1")
        
        st.markdown("#### 3️⃣ معدل النمو اللوغاريتمي")
        st.latex(r"g = \ln(Y_t) - \ln(Y_{t-1}) \approx \frac{Y_t - Y_{t-1}}{Y_{t-1}}")
        
        st.markdown("#### 4️⃣ معدل النمو على أساس سنوي (YoY)")
        st.latex(r"g_{YoY} = \frac{Y_t - Y_{t-4}}{Y_{t-4}} \times 100 \quad \text{(للبيانات الربع سنوية)}")
        
        st.markdown("#### 5️⃣ معدل النمو المعدل موسمياً (QoQ)")
        st.latex(r"g_{QoQ} = \frac{Y_t^{SA} - Y_{t-1}^{SA}}{Y_{t-1}^{SA}} \times 100")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #FFE4C4, #FFDAB9); border-radius: 15px; margin-top: 30px;">
    <h3 style="color: #8B4513;">📊 تطبيق معالجة وتحويل البيانات الاقتصادية</h3>
    <p style="color: #D2691E;">Economic Data Processing & Transformation Application</p>
    <p style="color: #8B4513;">من إعداد الدكتور مروان رودان | By Dr. Marouane Roudan</p>
    <p style="color: #CD853F; font-size: 0.9rem;">جميع الحقوق محفوظة © 2025</p>
</div>
""", unsafe_allow_html=True)
