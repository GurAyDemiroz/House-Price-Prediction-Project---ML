import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_lottie import st_lottie
import json
import os
import time
from logger import get_logger
from loading_page import show_loading_animation
import matplotlib.pyplot as plt
import io
import base64
from pipeline import DataPreprocessingPipeline
import altair as alt

# Sayfa yapılandırması; ilk komut olmalı
st.set_page_config(
    page_title="Adana Ev Fiyat Tahmini",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Uygulama başlığı ve tanıtımını güncelle - kutu arka planı kaldırıldı
st.markdown("""
    <div style="text-align: center; padding: 2rem 1rem; margin-bottom: 1.5rem;">
        <h1 style="color: #202124; font-size: 2.5rem; margin-bottom: 1rem;">🏠 Adana Akıllı Ev Fiyat Tahmini</h1>
        <p style="color: #202124; font-size: 1.1rem; margin-bottom: 0.5rem;">
            Yapay Zeka Destekli Anlık Fiyat Tahmin Sistemi
        </p>
    </div>
""", unsafe_allow_html=True)

# -------------------- CONFIG --------------------
logger = get_logger("logs/app.log")

# -------------------- LOTTIE --------------------
def load_lottie(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Lottie loading error: {str(e)}")
        return None

# Modeli yüklerken @st.cache_resource kullanarak önbelleğe alıyoruz
@st.cache_resource
def load_model():
    try:
        model_dirs = [d for d in os.listdir("models") if d.startswith("model_")]
        if not model_dirs:
            st.error("Hiçbir model dosyası bulunamadı. Lütfen önce modeli eğitin.")
            logger.error("No model directories found in 'models/' folder")
            raise FileNotFoundError("No model directories found")
        
        latest_model_dir = sorted(model_dirs, reverse=True)[0]
        model_base_path = os.path.join("models", latest_model_dir)
        logger.info(f"Loading model from {model_base_path}")
        
        model_path = os.path.join(model_base_path, "model.joblib")
        scaler_path = os.path.join(model_base_path, "scaler.joblib")
        feature_info_path = os.path.join(model_base_path, "feature_info.joblib")
        encoder_path = os.path.join(model_base_path, "encoder.joblib")
        
        for path in [model_path, scaler_path, feature_info_path, encoder_path]:
            if not os.path.exists(path):
                st.error(f"Model dosyası bulunamadı: {path}")
                logger.error(f"Required model file not found: {path}")
                raise FileNotFoundError(f"Required model file not found: {path}")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_info = joblib.load(feature_info_path)
        encoder = joblib.load(encoder_path)
        
        if 'district_neighborhoods' not in feature_info or len(feature_info['district_neighborhoods']) == 0:
            logger.info("İlçe-mahalle haritası oluşturuluyor...")
            district_neighborhoods, district_neighborhood_prices = load_district_neighborhood_map()
            
            if not district_neighborhoods:
                logger.warning("Veri dosyasından ilçe-mahalle haritası yüklenemedi!")
                
            feature_info['district_neighborhoods'] = district_neighborhoods
            feature_info['district_neighborhood_prices'] = district_neighborhood_prices
            logger.info(f"Created district-neighborhood mappings for {len(district_neighborhoods)} districts")
        
        if 'categorical_values' not in feature_info:
            logger.warning("feature_info does not contain 'categorical_values'. Initializing empty dictionary.")
            feature_info['categorical_values'] = {}
        
        cat_features = feature_info.get('categorical_features', [])
        categories = getattr(encoder, 'categories_', [])
        
        if len(categories) > 0:
            for i, feature in enumerate(cat_features):
                if i < len(categories):
                    feature_info['categorical_values'][feature] = [str(v).lower() for v in categories[i]]
                    logger.info(f"Loaded {len(categories[i])} known values for '{feature}'")
        else:
            logger.warning("Could not retrieve categories from encoder. Using empty lists.")
            for feature in cat_features:
                if feature not in feature_info['categorical_values']:
                    feature_info['categorical_values'][feature] = []
        
        province_values = feature_info['categorical_values'].get('province', [])
        if not province_values or all(p != 'adana' for p in province_values):
            feature_info['categorical_values']['province'] = ['adana'] + province_values
            logger.info("Added 'adana' to province values")
        
        critical_features = ['province', 'district', 'neighborhood', 'heating']
        for feature in critical_features:
            if feature not in feature_info['categorical_values'] or len(feature_info['categorical_values'][feature]) == 0:
                logger.warning(f"Critical feature '{feature}' has no values from encoder. Using fallback value.")
                if feature == 'province':
                    feature_info['categorical_values'][feature] = ['Adana']
                elif feature == 'district':
                    feature_info['categorical_values'][feature] = ['Seyhan']
                elif feature == 'neighborhood':
                    feature_info['categorical_values'][feature] = ['Merkez']
                elif feature == 'heating':
                    feature_info['categorical_values'][feature] = ['Kombi']
        
        if 'furnished' not in feature_info['categorical_values']:
            logger.info("Adding 'furnished' feature with default values.")
            feature_info['categorical_values']['furnished'] = ['Evet', 'Hayır']
        
        building_age_options = feature_info.get("building_age_options", ['0', '1-4', '5-10', '11-15', '16-20', '21-25', '26-30', '31+'])
        
        cat_features = feature_info.get('categorical_features', [])
        categories = getattr(encoder, 'categories_', [])
        for i, feature in enumerate(cat_features):
            if i < len(categories):
                feature_info['categorical_values'][feature] = list(categories[i])
                logger.info(f"Loaded {len(categories[i])} known values for '{feature}'")
        
        if 'numeric_features' in feature_info and 'gross_area' in feature_info['numeric_features']:
            logger.info("Removing gross_area from numeric_features")
            feature_info['numeric_features'] = [f for f in feature_info['numeric_features'] if f != 'gross_area']
        if 'column_order' in feature_info and 'gross_area' in feature_info['column_order']:
            logger.info("Removing gross_area from column_order")
            feature_info['column_order'] = [c for c in feature_info['column_order'] if c != 'gross_area']
        
        if 'feature_effects' not in feature_info:
            logger.warning("No feature effects found in the model. Neighborhood comparison will be limited.")
            feature_info['feature_effects'] = {}
        else:
            grouped_effects = feature_info['feature_effects'].get('_grouped', {})
            neighborhood_effects = grouped_effects.get('neighborhood', {})
            district_effects = grouped_effects.get('district', {})
            logger.info(f"Loaded feature effects with {len(neighborhood_effects)} neighborhoods and {len(district_effects)} districts")
            if neighborhood_effects:
                top_neighborhoods = sorted(neighborhood_effects.items(), key=lambda x: x[1], reverse=True)[:3]
                logger.info(f"Top neighborhoods by effect: {top_neighborhoods}")
        
        logger.info(f"Model successfully loaded from {model_base_path}")
        return model, scaler, feature_info, encoder, building_age_options
    except Exception as e:
        logger.error(f"Model loading error: {str(e)}")
        st.error(f"Model yüklenirken bir hata oluştu: {str(e)}. Lütfen önce modeli eğittiğinizden emin olun.")
        st.stop()

def load_district_neighborhood_map():
    try:
        logger.info("İlçe-mahalle eşleştirmesi ve fiyatları yükleniyor...")
        processed_df = pd.read_csv("data/processed_data.csv")
        
        district_neighborhoods = {}
        district_neighborhood_prices = {}
        
        if 'district' in processed_df.columns and 'neighborhood' in processed_df.columns:
            # İlçe-mahalle haritası oluşturma
            for district in processed_df['district'].unique():
                if pd.isna(district):
                    continue
                
                # District isimlerini normalize et (tüm boşlukları temizle, küçük harfe çevir)
                district_str = str(district).strip().lower()
                district_data = processed_df[processed_df['district'] == district]
                neighborhoods = district_data['neighborhood'].unique()
                
                clean_neighborhoods = sorted([str(n).strip().lower() for n in neighborhoods if pd.notna(n)])
                
                if clean_neighborhoods:
                    district_neighborhoods[district_str] = clean_neighborhoods
            
            # İlçe bazında mahalle fiyatlarını hesaplama
            if 'house_price' in processed_df.columns:
                # İlçe ve mahalle bazında ortalama house_priceları hesapla
                neighborhood_prices = processed_df.groupby(['district', 'neighborhood'])['house_price'].mean().reset_index()
                
                # İlçe bazında gruplama (case-insensitive olarak çalışacak şekilde)
                for _, row in neighborhood_prices.iterrows():
                    district = str(row['district']).strip().lower()
                    neighborhood = str(row['neighborhood']).strip().lower()
                    price = row['house_price']
                    
                    if district not in district_neighborhood_prices:
                        district_neighborhood_prices[district] = {}
                    
                    district_neighborhood_prices[district][neighborhood] = price
                
                logger.info(f"{len(district_neighborhood_prices)} ilçe için ortalama mahalle fiyatları hesaplandı")
            else:
                # 'fiyat' sütunu yoksa uyarı ver
                logger.warning("Veri setinde 'house_price' sütunu bulunamadı.")
        
        logger.info(f"Pipeline ile {len(district_neighborhoods)} ilçe yüklendi")
                
        return district_neighborhoods, district_neighborhood_prices
        
    except Exception as e:
        logger.error(f"İlçe-mahalle haritası oluşturulurken hata: {e}", exc_info=True)
        return {}, {}

def set_custom_style():
    st.markdown("""
        <style>
        /* Temel stil değişkenleri */
        :root {
            --primary: #1a73e8;
            --secondary: #4285f4;
            --background: #f8f9fa;
            --surface: #ffffff;
            --text: #202124;
            --error: #d93025;
        }

        /* DOĞRUDAN HATA MESAJLARI İÇİN ZORLAYICI CSS RESET */
        /* Tüm Streamlit hata mesajlarının arka planı ve yazı rengi */
        div[data-testid="stAlert"] {
            background-color: #ffebee !important;
            border-color: #b71c1c !important;
        }

        /* Tüm hata mesajlarındaki tüm metin elementleri */
        div[data-testid="stAlert"] * {
            color: #000000 !important;
            font-weight: bold !important;
        }

        /* En spesifik CSS selector */
        div.element-container div[data-testid="stAlert"] p {
            color: #000000 !important;
            font-size: 16px !important;
            font-weight: 600 !important;
        }

        /* Hata mesajı ikon rengi */
        div[data-testid="stAlert"] svg {
            fill: #000000 !important;
            color: #000000 !important;
        }

        /* Genel uyarı mesajı stili */
        .stAlert {
            background-color: #ffebee !important;
            border-color: #b71c1c !important;
            color: #000000 !important;
        }

        /* Inline stil eklemesi */
        div[data-testid="stAlert"] {
            background-color: #ffebee !important; 
            border-color: #b71c1c !important;
            color: #000000 !important;
        }

        /* Tüm elemanları kapsayıcı stil */
        div[data-testid="stAlert"] div, 
        div[data-testid="stAlert"] p, 
        div[data-testid="stAlert"] span, 
        div[data-testid="stAlert"] a, 
        div[data-testid="stAlert"] label {
            color: #000000 !important;
            text-shadow: none !important;
        }

        /* Diğer mevcut CSS kodları */
        // ...existing code...
        </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_cached_css():
    try:
        with open('static/styles/main.css') as f:
            return f.read()
    except Exception as e:
        logger.error(f"CSS loading error: {str(e)}")
        return ""

def load_custom_css():
    css_content = load_cached_css()
    st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)

def create_feature_input_form(feature_info):
    # Form değişikliklerini izlemek için callback fonksiyonu
    def reset_prediction():
        if "show_prediction" in st.session_state:
            st.session_state["show_prediction"] = False
            
    screen_width = st.session_state.get("screen_width", 1200)  # Varsayılan değer
    
    if screen_width > 768:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<p class="css-10trblm">📐 Alan ve Oda Bilgileri</p>', unsafe_allow_html=True)
            net_area = st.number_input("Net Alan (m²)", min_value=40, max_value=500, value=100, on_change=reset_prediction)
            room_count = st.number_input("Oda Sayısı", min_value=1, max_value=10, value=3, on_change=reset_prediction)
            living_room_count = st.number_input("Salon Sayısı", min_value=0, max_value=4, value=1, on_change=reset_prediction)
            bathroom_count = st.number_input("Banyo Sayısı", min_value=1, max_value=5, value=1, on_change=reset_prediction)
            furnished_options = feature_info.get('categorical_values', {}).get('furnished', ['Evet', 'Hayır'])
            if not furnished_options:
                furnished_options = ['Evet', 'Hayır']
            furnished = st.selectbox("Eşyalı mı?", options=furnished_options, on_change=reset_prediction)
        
        with col2:
            st.markdown('<p class="css-10trblm">🏙️ Konum ve Bina Bilgileri</p>', unsafe_allow_html=True)
            subcol1, subcol2 = st.columns(2)
            
            district_neighborhoods = feature_info.get('district_neighborhoods', {})
            
            formatted_district_neighborhoods = {}
            for district, neighborhoods in district_neighborhoods.items():
                formatted_district = district.title()
                formatted_neighborhoods = [n.title() for n in neighborhoods]
                formatted_district_neighborhoods[formatted_district] = formatted_neighborhoods
            
            district_neighborhoods = formatted_district_neighborhoods
            
            logger.info(f"Form oluşturulurken ilçeler: {list(district_neighborhoods.keys())}")
            
            with subcol1:
                district_options = ["Seçiniz"]
                if district_neighborhoods:
                    district_options += sorted(list(district_neighborhoods.keys()))
                
                if 'current_district' not in st.session_state:
                    st.session_state['current_district'] = "Seçiniz"
                
                def on_district_change():
                    selected = st.session_state.district_selector
                    logger.info(f"İlçe değişti: '{selected}'")
                    st.session_state['current_district'] = selected
                    st.session_state['current_neighborhood'] = "Seçiniz"
                    # Sadece tahmin sonucunu sıfırla, animasyon gösterme
                    if "show_prediction" in st.session_state:
                        st.session_state["show_prediction"] = False
                
                district_index = district_options.index(st.session_state['current_district']) if st.session_state['current_district'] in district_options else 0
                selected_district = st.selectbox(
                    "İlçe", 
                    options=district_options,
                    index=district_index,
                    key="district_selector",
                    on_change=on_district_change
                )
            
            with subcol2:
                current_district = st.session_state['current_district']
                
                neighborhood_options = ["Seçiniz"]
                if current_district != "Seçiniz" and current_district in district_neighborhoods:
                    mahalles = district_neighborhoods[current_district]
                    logger.info(f"'{current_district}' ilçesi için {len(mahalles)} mahalle bulundu: {mahalles[:3]}...")
                    neighborhood_options = ["Seçiniz"] + sorted(mahalles)
                else:
                    logger.warning(f"'{current_district}' ilçesi için mahalle bulunamadı veya 'Seçiniz' seçeneği seçildi.")
                
                if 'current_neighborhood' not in st.session_state:
                    st.session_state['current_neighborhood'] = "Seçiniz"
                elif st.session_state['current_neighborhood'] not in neighborhood_options:
                    st.session_state['current_neighborhood'] = "Seçiniz"
                
                neighborhood_index = neighborhood_options.index(st.session_state['current_neighborhood']) if st.session_state['current_neighborhood'] in neighborhood_options else 0
                selected_neighborhood = st.selectbox(
                    "Mahalle", 
                    options=neighborhood_options,
                    index=neighborhood_index,
                    key="neighborhood_selector",
                    on_change=reset_prediction
                )
                
                st.session_state['current_neighborhood'] = selected_neighborhood
            
            building_age = st.number_input("Bina Yaşı (Yıl)", min_value=0, max_value=50, value=5, on_change=reset_prediction)
            floor_number = st.number_input("Bulunduğu Kat", min_value=-2, max_value=30, value=2, on_change=reset_prediction)
            total_floors = st.number_input("Toplam Kat Sayısı", min_value=1, max_value=30, value=5, on_change=reset_prediction)
            
            heating_options = feature_info['categorical_values'].get('heating', ['Kombi', 'Merkezi', 'Soba', 'Yok'])
            if not heating_options:
                heating_options = ['Kombi', 'Merkezi', 'Soba', 'Yok']
            heating = st.selectbox("Isıtma Tipi", options=heating_options, on_change=reset_prediction)
    else:
        st.markdown('<p class="css-10trblm">📐 Alan ve Oda Bilgileri</p>', unsafe_allow_html=True)
        net_area = st.number_input("Net Alan (m²)", min_value=40, max_value=500, value=100, on_change=reset_prediction)
        room_count = st.number_input("Oda Sayısı", min_value=1, max_value=10, value=3, on_change=reset_prediction)
        living_room_count = st.number_input("Salon Sayısı", min_value=0, max_value=4, value=1, on_change=reset_prediction)
        bathroom_count = st.number_input("Banyo Sayısı", min_value=1, max_value=5, value=1, on_change=reset_prediction)
        
        furnished_options = feature_info.get('categorical_values', {}).get('furnished', ['Evet', 'Hayır'])
        if not furnished_options:
            furnished_options = ['Evet', 'Hayır']
        furnished = st.selectbox("Eşyalı mı?", options=furnished_options, on_change=reset_prediction)
        
        district_neighborhoods = feature_info.get('district_neighborhoods', {})
        
        formatted_district_neighborhoods = {}
        for district, neighborhoods in district_neighborhoods.items():
            formatted_district = district.title()
            formatted_neighborhoods = [n.title() for n in neighborhoods]
            formatted_district_neighborhoods[formatted_district] = formatted_neighborhoods
        
        district_neighborhoods = formatted_district_neighborhoods
        
        district_options = ["Seçiniz"]
        if district_neighborhoods:
            district_options += sorted(list(district_neighborhoods.keys()))
        
        if 'current_district' not in st.session_state:
            st.session_state['current_district'] = "Seçiniz"
        
        def on_district_change():
            selected = st.session_state.district_selector
            logger.info(f"İlçe değişti: '{selected}'")
            st.session_state['current_district'] = selected
            st.session_state['current_neighborhood'] = "Seçiniz"
            # Sadece tahmin sonucunu sıfırla, animasyon gösterme
            if "show_prediction" in st.session_state:
                st.session_state["show_prediction"] = False
        
        district_index = district_options.index(st.session_state['current_district']) if st.session_state['current_district'] in district_options else 0
        selected_district = st.selectbox(
            "İlçe", 
            options=district_options,
            index=district_index,
            key="district_selector",
            on_change=on_district_change
        )
        
        current_district = st.session_state['current_district']
        
        neighborhood_options = ["Seçiniz"]
        if current_district != "Seçiniz" and current_district in district_neighborhoods:
            mahalles = district_neighborhoods[current_district]
            neighborhood_options = ["Seçiniz"] + sorted(mahalles)
        
        if 'current_neighborhood' not in st.session_state:
            st.session_state['current_neighborhood'] = "Seçiniz"
        elif st.session_state['current_neighborhood'] not in neighborhood_options:
            st.session_state['current_neighborhood'] = "Seçiniz"
        
        neighborhood_index = neighborhood_options.index(st.session_state['current_neighborhood']) if st.session_state['current_neighborhood'] in neighborhood_options else 0
        selected_neighborhood = st.selectbox(
            "Mahalle", 
            options=neighborhood_options,
            index=neighborhood_index,
            key="neighborhood_selector",
            on_change=reset_prediction
        )
        
        st.session_state['current_neighborhood'] = selected_neighborhood
        
        building_age = st.number_input("Bina Yaşı (Yıl)", min_value=0, max_value=50, value=5, on_change=reset_prediction)
        floor_number = st.number_input("Bulunduğu Kat", min_value=-2, max_value=30, value=2, on_change=reset_prediction)
        total_floors = st.number_input("Toplam Kat Sayısı", min_value=1, max_value=30, value=5, on_change=reset_prediction)
        
        heating_options = feature_info['categorical_values'].get('heating', ['Kombi', 'Merkezi', 'Soba', 'Yok'])
        if not heating_options:
            heating_options = ['Kombi', 'Merkezi', 'Soba', 'Yok']
        heating = st.selectbox("Isıtma Tipi", options=heating_options, on_change=reset_prediction)
    
    submit_button = st.button("💰 Fiyat Tahmini Yap", use_container_width=True)
    
    valid_selection = True
    if submit_button:
        if selected_district == "Seçiniz" or selected_neighborhood == "Seçiniz":
            # Özel stil ekleyerek hata mesajı
            st.markdown("""
            <div style="background-color:#ffebee; color:#000000; padding:10px; 
                 border-radius:5px; border-left:5px solid #b71c1c; margin:10px 0; 
                 font-weight:600; font-size:16px;">
                ⚠️ Lütfen ilçe ve mahalle seçiniz!
            </div>
            """, unsafe_allow_html=True)
            valid_selection = False
            logger.warning("Kullanıcı ilçe veya mahalle seçmeden tahmin yapmaya çalıştı")
        else:
            logger.info(f"Fiyat tahmini yapılıyor: İlçe={selected_district}, Mahalle={selected_neighborhood}")
    
    return submit_button and valid_selection, {
        'net_area': net_area,
        'room_count': room_count,
        'living_room_count': living_room_count,
        'bathroom_count': bathroom_count,
        'building_age': building_age,
        'floor_number': floor_number,
        'total_floors': total_floors,
        'heating': heating,
        'neighborhood': selected_neighborhood if selected_neighborhood != "Seçiniz" else "Merkez",
        'province': "Adana",
        'district': selected_district if selected_district != "Seçiniz" else "Seyhan",
        'furnished': furnished
    }

def show_prediction_result(prediction, features, building_age_options):
    lower_bound = prediction * 0.9
    upper_bound = prediction * 1.1
    
    formatted_prediction = "{:,.0f}".format(prediction).replace(",", ".")
    formatted_lower = "{:,.0f}".format(lower_bound).replace(",", ".")
    formatted_upper = "{:,.0f}".format(upper_bound).replace(",", ".")
    
    st.markdown("""
    <div style="width:100%; margin:0 auto; padding:min(20px, 5vw); 
        background: linear-gradient(135deg, #ffffff, #e1f5fe); 
        border-radius:12px; 
        box-shadow: 0 3px 10px rgba(0,0,0,0.1); 
        text-align:center;">
        <h2 style="color:#1a73e8; margin-bottom:10px; font-weight:600; font-size:clamp(1.2rem, 3vw, 1.5rem);">Tahmin Edilen Fiyat</h2>
        <h1 style="color:#1a73e8; font-size:clamp(1.8rem, 5vw, 2.2rem); margin:10px 0; font-weight:700;">
            {} TL
        </h1>
        <p style="color:#5f6368; margin-top:8px; font-size:clamp(0.7rem, 1.5vw, 0.85rem);">
            Bu tahmin, girdiğiniz ev özelliklerine göre hesaplanmıştır.<br>
            Tahmin ±%10 aralığında değişebilir ({} - {} TL)
        </p>
    </div>
    """.format(formatted_prediction, formatted_lower, formatted_upper), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

def predict_price(features, model, scaler, feature_info, encoder):
    try:
        logger.info(f"Fiyat tahmini yapılıyor. Mahalle: {features['neighborhood']}, İlçe: {features['district']}")
        input_df = pd.DataFrame([features])
        numeric_features = feature_info.get('numeric_features', [])
        categorical_features = feature_info.get('categorical_features', [])
        
        # Kategorik özelliklerin loglanması
        logger.info(f"Kategorik özellikler: {categorical_features}")
        if 'district' in categorical_features and 'neighborhood' in categorical_features:
            logger.info("İlçe ve mahalle kategorik özellikler arasında bulunuyor.")
        else:
            logger.warning("DİKKAT: İlçe ve/veya mahalle kategorik özellikler arasında bulunmuyor!")
        
        for feature in numeric_features:
            if feature not in input_df.columns:
                logger.warning(f"Sayısal özellik '{feature}' input_df'de bulunamadı. Varsayılan değer 0 atandı.")
                input_df[feature] = 0
        
        for feature in categorical_features:
            if feature not in input_df.columns:
                logger.warning(f"Kategorik özellik '{feature}' input_df'de bulunamadı. Varsayılan değer atandı.")
                if feature == 'province':
                    input_df[feature] = 'Adana'
                elif feature == 'district':
                    input_df[feature] = 'Seyhan'
                elif feature == 'neighborhood':
                    input_df[feature] = 'Merkez'
                elif feature == 'heating':
                    input_df[feature] = 'Kombi'
                elif feature == 'furnished':
                    input_df[feature] = 'Hayır'
                else:
                    input_df[feature] = 'Bilinmiyor'
        
        X_numeric = input_df[numeric_features]
        X_numeric_scaled = scaler.transform(X_numeric)
        
        X_categorical = input_df[categorical_features]
        # Encoding öncesi kategorik değerleri logla
        logger.info(f"Encoding öncesi kategorik değerler: {X_categorical.iloc[0].to_dict()}")
        X_categorical_encoded = encoder.transform(X_categorical)
        
        X_numeric_scaled_df = pd.DataFrame(
            X_numeric_scaled,
            columns=numeric_features
        )
        feature_names = encoder.get_feature_names_out(categorical_features)
        X_categorical_encoded_df = pd.DataFrame(
            X_categorical_encoded,
            columns=feature_names
        )
        
        X_combined = pd.concat([X_numeric_scaled_df, X_categorical_encoded_df], axis=1)
        
        # Giriş verilerinin son kontrolü
        logger.info(f"Model giriş boyutu: {X_combined.shape}")
        
        if 'feature_order' in feature_info:
            missing_features = set(feature_info['feature_order']) - set(X_combined.columns)
            if missing_features:
                logger.warning(f"Eksik özellikler: {missing_features}")
        
        predicted_price = model.predict(X_combined)[0]
        if predicted_price < 0:
            logger.warning(f"Negatif fiyat tahmini düzeltiliyor: {predicted_price} -> 0")
            predicted_price = 0
        
        logger.info(f"Tahmin edilen fiyat: {predicted_price:,.2f} TL (Mahalle: {features['neighborhood']})")
        return predicted_price
    except Exception as e:
        logger.error(f"Tahmin hatası: {str(e)}", exc_info=True)
        st.error(f"Fiyat tahmini yapılırken bir hata oluştu: {str(e)}")
        return 2_000_000

def main():
    # CSS'i en başa al
    load_custom_css()
    set_custom_style()
    
    # Animasyonu sadece ilk yüklemede göstermek için session state kullan
    if "first_load" not in st.session_state:
        st.session_state["first_load"] = True
        loading_placeholder = show_loading_animation("Uygulama başlatılıyor...", duration=2)
    else:
        loading_placeholder = None
    
    try:
        screen_width_detection = """
        <script>
            var width = window.innerWidth;
            if (width !== undefined) {
                sessionStorage.setItem('screen_width', width);
            }
        </script>
        """
        st.markdown(screen_width_detection, unsafe_allow_html=True)
        
        width_getter = """
        <script>
            var screenWidth = sessionStorage.getItem('screen_width');
            if (screenWidth !== null) {
                window.parent.postMessage({
                    type: "streamlit:setSessionState",
                    data: { 
                        "screen_width": parseInt(screenWidth)
                    }
                }, "*");
            }
        </script>
        """
        st.markdown(width_getter, unsafe_allow_html=True)
        
        # Animasyonu kapat eğer gösterilmişse
        if loading_placeholder is not None:
            loading_placeholder.empty()
        
        if "model" not in st.session_state:
            # Model yükleme durumunu bildir ama animasyon gösterme
            with st.spinner("Model yükleniyor, lütfen bekleyin..."):
                model, scaler, feature_info, encoder, building_age_options = load_model()
                st.session_state["model"] = model
                st.session_state["scaler"] = scaler
                st.session_state["encoder"] = encoder
                st.session_state["building_age_options"] = building_age_options
                st.session_state["feature_info"] = feature_info
                st.session_state["model_loaded"] = True

                logger.info("İlçe-mahalle haritası yükleniyor...")
                district_neighborhoods, district_neighborhood_prices = load_district_neighborhood_map()
                
                retry_count = 0
                while (not district_neighborhoods or not district_neighborhood_prices) and retry_count < 3:
                    logger.warning(f"İlçe verileri yüklenemedi. Yeniden deneniyor... ({retry_count+1}/3)")
                    time.sleep(1)  
                    district_neighborhoods, district_neighborhood_prices = load_district_neighborhood_map()
                    retry_count += 1
                
                feature_info['district_neighborhoods'] = district_neighborhoods
                feature_info['district_neighborhood_prices'] = district_neighborhood_prices
                st.session_state["feature_info"] = feature_info
                logger.info(f"İlçe-mahalle haritası yüklendi: {len(district_neighborhoods)} ilçe, {len(district_neighborhood_prices)} fiyat")
        else:
            model = st.session_state["model"]
            scaler = st.session_state["scaler"]
            encoder = st.session_state["encoder"]
            building_age_options = st.session_state["building_age_options"]
            feature_info = st.session_state["feature_info"]

            if not feature_info.get('district_neighborhoods') or len(feature_info.get('district_neighborhoods', {})) == 0:
                logger.info("İlçe-mahalle haritası boş, pipeline'dan yükleniyor...")
                district_neighborhoods, district_neighborhood_prices = load_district_neighborhood_map()
                feature_info['district_neighborhoods'] = district_neighborhoods
                feature_info['district_neighborhood_prices'] = district_neighborhood_prices
                st.session_state["feature_info"] = feature_info
                logger.info(f"İlçe-mahalle haritası yüklendi: {len(district_neighborhoods)} ilçe")

        submit_button, features = create_feature_input_form(feature_info)
        
        if submit_button:
            prediction = predict_price(features, model, scaler, feature_info, encoder)
            
            st.session_state["last_features"] = features
            st.session_state["last_prediction"] = prediction
            st.session_state["show_prediction"] = True
            
            st.rerun()
        
        if st.session_state.get("show_prediction", False):
            st.container().markdown('<div id="prediction-top"></div>', unsafe_allow_html=True)
            show_prediction_result(
                st.session_state["last_prediction"],
                st.session_state["last_features"],
                st.session_state.get("building_age_options", [])
            )

    except Exception as e:
        logger.error(f"Uygulama başlatılırken hata: {e}")
        st.error(f"Uygulama başlatılırken bir hata oluştu: {e}")
        st.stop()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"Uygulama başlatılırken bir hata oluştu: {e}")
        logger.error(f"Application error: {e}")
