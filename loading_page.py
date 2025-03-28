import streamlit as st
import time

def show_loading_animation(message="Yükleniyor...", duration=None, autoplay=True):
    """
    Gelişmiş bir yükleme animasyonu gösterir.
    
    Args:
        message: Gösterilecek mesaj
        duration: Animasyon süresi (saniye), None ise otomatik kapanmaz
        autoplay: True ise otomatik başlar, False ise kontrol edilebilir
    
    Returns:
        placeholder: Animasyon placeholder'ı (opsiyonel)
    """
    # Daha renkli ve çekici bir animasyon CSS'i
    st.markdown("""
    <style>
    @keyframes rotation {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes colorChange {
        0% { border-top-color: #4285f4; }
        25% { border-top-color: #ea4335; }
        50% { border-top-color: #fbbc05; }
        75% { border-top-color: #34a853; }
        100% { border-top-color: #4285f4; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .loading-animation {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        margin: 0 auto;
        max-width: 400px;
        background: linear-gradient(145deg, #ffffff, #f5f8ff);
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        z-index: 1000;
        animation: fadeIn 0.5s ease-out forwards, pulse 2s infinite;
        border-left: 4px solid #4285f4;
    }
    
    .loading-spinner {
        width: 70px;
        height: 70px;
        border-radius: 50%;
        border: 5px solid #e1e4e8;
        border-top-color: #4285f4;
        animation: rotation 1s linear infinite, colorChange 3s infinite;
        margin-bottom: 1.5rem;
    }
    
    .loading-text {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a73e8;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .loading-message {
        font-size: 1.2rem;
        color: #5f6368;
        text-align: center;
        margin-top: 0.5rem;
        background: linear-gradient(90deg, #4285f4, #34a853, #fbbc05, #ea4335);
        -webkit-background-clip: text;
        color: transparent;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Geliştirilen HTML yapısı
    loading_html = f"""
    <div class="loading-animation">
        <div class="loading-spinner"></div>
        <div class="loading-text">🏠 Ev Fiyat Tahmini</div>
        <div class="loading-message">{message}</div>
    </div>
    """
    
    # Animasyon placeholder'ı
    placeholder = st.empty()
    placeholder.markdown(loading_html, unsafe_allow_html=True)
    
    # Sabit minimum süre
    if duration is not None:
        time.sleep(duration)
        placeholder.empty()
    
    return placeholder

def example_usage():
    """Loading sayfasının örnek kullanımını gösterir"""
    st.title("Yükleme Animasyonu Örneği")
    
    if st.button("Yükleme Animasyonu Göster"):
        # Uzun süreli animasyon
        loading_placeholder = show_loading_animation("Veriler hazırlanıyor...", duration=None, autoplay=False)
        
        # İşlem simülasyonu
        progress = st.progress(0)
        for i in range(100):
            # Daha yavaş ilerleyen bir ilerleme çubuğu - her adım 0.1 saniye
            time.sleep(0.1)
            progress.progress(i + 1)
        
        # Minimum 3 saniye görünür kalsın
        time.sleep(3)
        
        # Elle kapatma
        loading_placeholder.empty()
        progress.empty()
        
        st.success("Yükleme tamamlandı!")

if __name__ == "__main__":
    example_usage()
