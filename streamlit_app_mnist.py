import os
import glob
import joblib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="MNIST",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

# Minimera vertikala avstånd för att få mer kompakt layout och ta bort "länk-ikoner" i rubrikerna
st.markdown("""
<style>
div.block-container { padding-top: 2.2rem !important; padding-bottom: 1rem !important; }
h1 { margin-top: 0rem !important; margin-bottom: 0.35rem !important; padding-top: 0rem !important; }
a.header-anchor, a[aria-label="Link to this section"], div[data-testid="stMarkdownContainer"] h1 a { display: none !important; }
div[data-testid="stCaptionContainer"] { margin-top: 0rem !important; padding-top: 0rem !important; }
</style>
""", unsafe_allow_html=True)

st.title('MNIST - Rita en siffra och prediktera sannolikheten')

# -----------------------------
# Ladda modellen
# -----------------------------
models_dir = 'models'
model_paths = glob.glob(os.path.join(models_dir, '*.joblib'))
if not model_paths:
    st.error('Hittar ingen .joblib-fil i models/.')
    st.stop()

model_path = max(model_paths, key=os.path.getmtime)
# st.caption(f'Modellfil: {os.path.basename(model_path)}')

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(model_path)

# -----------------------------
# Session state för att kunna "rensa" prediktion
# -----------------------------
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0

if 'last_pred' not in st.session_state:
    st.session_state.last_pred = None

if 'last_top3' not in st.session_state:
    st.session_state.last_top3 = None

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# -----------------------------
# Steg 5: Auto-crop & center
# -----------------------------
def crop_and_center_to_28x28(gray_0_255: np.ndarray) -> np.ndarray:
    """
    Tar en gråskalebild (0..255), hittar ritad siffra, croppar, gör kvadrat,
    skalar till 20x20 och paddar till 28x28 samt centrerar med tyngdpunkt.
    Returnerar en (28,28) i 0..255.
    """
    x = gray_0_255.copy()

    # Tröskel för att hitta "ritad" pixel
    thresh = 10
    ys, xs = np.where(x > thresh)

    # Om inget ritats: returnera original (tom bild)
    if len(xs) == 0:
        return x

    # Bounding box
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Lägg på lite marginal (10%)
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    pad = int(0.10 * max(w, h))

    x_min = max(0, x_min - pad)
    x_max = min(x.shape[1] - 1, x_max + pad)
    y_min = max(0, y_min - pad)
    y_max = min(x.shape[0] - 1, y_max + pad)

    cropped = x[y_min:y_max + 1, x_min:x_max + 1]

    # Gör kvadrat med svart padding
    ch, cw = cropped.shape
    size = max(ch, cw)
    square = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - ch) // 2
    x_off = (size - cw) // 2
    square[y_off:y_off + ch, x_off:x_off + cw] = cropped

    # Skala till 20x20 och pad till 28x28
    img20 = Image.fromarray(square).resize((20, 20), resample=Image.Resampling.LANCZOS)
    img20 = np.array(img20, dtype=np.uint8)

    img28 = np.zeros((28, 28), dtype=np.uint8)
    img28[4:24, 4:24] = img20

    # Centering med tyngdpunkt
    yy, xx = np.indices(img28.shape)
    mass = img28.astype(np.float32)
    total = mass.sum()

    if total > 0:
        cy = (yy * mass).sum() / total
        cx = (xx * mass).sum() / total

        shift_y = int(round(14 - cy))
        shift_x = int(round(14 - cx))

        img28 = np.roll(img28, shift=shift_y, axis=0)
        img28 = np.roll(img28, shift=shift_x, axis=1)

    return img28

def uploaded_image_to_gray_array(uploaded_file) -> np.ndarray:
    """
    Tar en uppladdad bildfil och returnerar en gråskalebild (H, W) i 0..255.
    Antagande: vit bakgrund, svart/grå siffra.
    """
    img = Image.open(uploaded_file).convert('L')
    gray = np.array(img, dtype=np.uint8)

    # Eftersom bakgrunden är vit och siffran mörk vill vi invert:
    # efter invert blir siffran ljus på svart bakgrund (som vår pipeline förväntar sig).
    gray = 255 - gray
    return gray

# -----------------------------
# Sidebar-inställningar
# -----------------------------
with st.sidebar:
    st.header('Inställningar')
    stroke_width = st.slider('Penselstorlek', 5, 40, 20)
    invert = st.checkbox('Invertera färger', value=False)
    use_autocrop = st.checkbox('Auto-crop & center av inmatning', value=True)
    uploaded_file = st.file_uploader(
        'Ladda upp en bild (png/jpg/jpeg)',
        type=['png', 'jpg', 'jpeg'],
        key=f'uploader_{st.session_state.uploader_key}',
)
    if st.button('Ta bort uppladdad bild'):
        st.session_state.uploader_key += 1
        st.rerun()
# -----------------------------
# Layout: Canvas vänster, resultat höger
# -----------------------------
left, right = st.columns([1.1, 0.9])

with left:
    st.subheader('Rita en siffra (0 - 9)')
    canvas_size = 280
    canvas_result = st_canvas(
        fill_color='rgba(0, 0, 0, 0)',
        stroke_width=stroke_width,
        stroke_color='#FFFFFF',
        background_color='#000000',
        height=canvas_size,
        width=canvas_size,
        drawing_mode='freedraw',
        display_toolbar=False,
        key=f'canvas_{st.session_state.canvas_key}',
    )
    # Rensa-knapp under canvas
    if st.button('Rensa'):
        st.session_state.canvas_key += 1
        st.session_state.last_pred = None
        st.session_state.last_top3 = None
        st.rerun()

with right:
    st.subheader('Preview + prediktion')

    # Välj källa: uppladdad bild om den finns, annars canvas
    img_gray = None
    source = None

    if uploaded_file is not None:
        img_gray = uploaded_image_to_gray_array(uploaded_file)
        source = 'uppladdad bild'
    elif canvas_result.image_data is not None:
        img_rgba = canvas_result.image_data.astype('uint8')
        img_gray = Image.fromarray(img_rgba, mode='RGBA').convert('L')
        img_gray = np.array(img_gray, dtype=np.uint8)
        source = 'canvas'

    if img_gray is None:
        st.info('Rita en siffra till vänster eller ladda upp en bild i sidomenyn.')
    else:
        st.caption(f'Källa: {source}')
    
        if source == 'uppladdad bild':
            st.info("Uppladdad bild används. Klicka 'Ta bort uppladdad bild' i sidomenyn för att rita istället.")
        else:
            st.info("Canvas används. Ladda upp en bild i sidomenyn om du vill prediktera en fil istället.")

        # Invert-checkbox gäller bara canvas (uppladdad bild inverteras redan i funktionen)
        if invert and source == 'canvas':
            img_gray = 255 - img_gray

        # Gör 28x28:
        if use_autocrop:
            img28 = crop_and_center_to_28x28(img_gray)
        else:
            img28 = Image.fromarray(img_gray).resize((28, 28), resample=Image.Resampling.LANCZOS)
            img28 = np.array(img28, dtype=np.uint8)

        # Kolla om bilden är "tom" (inget ritats)
        if img28.max() < 10:
            st.session_state.last_pred = None
            st.session_state.last_top3 = None
            st.image(img28, width=180, caption='28x28 preview')
        else:
            # Skala till 0..1 och reshape till (1, 784)
            x = img28.astype(np.float32).reshape(1, -1) / 255.0

            # Preview
            st.image(img28, width=180, caption='Det modellen ser (28x28 preview)')

            # Prediktion
            pred = int(model.predict(x)[0])
            st.session_state.last_pred = pred

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(x)[0]
                top3 = np.argsort(proba)[::-1][:3]
                st.session_state.last_top3 = [(int(i), float(proba[i])) for i in top3]
            else:
                st.session_state.last_top3 = None

    # Visa resultat endast om de finns (och de rensas när canvas rensas)
    if st.session_state.last_pred is not None:
        st.success(f'Prediktion: {st.session_state.last_pred}')

        if st.session_state.last_top3 is not None:
            st.write('Topp-3 sannolikheter:')
            for cls, p in st.session_state.last_top3:
                st.write(f'{cls}: {p * 100:.1f}%')