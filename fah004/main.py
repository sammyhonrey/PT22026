import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import random
import base64
from PIL import Image
import io
from inference.visual_engine import VisualSearchEngine

st.set_page_config(page_title="VisualCommerce", page_icon="🛍️", layout="wide")

# --- PREMIUM E-COMMERCE CSS ---
st.markdown("""
<style>
.stApp { background-color: #FFFFFF; }
.block-container { padding-top: 20px !important; padding-bottom: 50px !important; max-width: 95% !important; margin: 0 auto !important; }

/* Top Banner */
.top-banner { background-color: #000; color: #FFF; text-align: center; padding: 10px; font-size: 0.85em; font-weight: 600; letter-spacing: 1px; width: 100%; display: block; }

/* Header Layout */
.header-container { text-align: center; padding: 30px 0 20px 0; border-bottom: 1px solid #EAEAEA; margin-bottom: 20px; }
.store-logo { font-size: 2.8em; font-weight: 900; letter-spacing: -1.5px; color: #000; text-decoration: none; margin-bottom: 15px; }

/* Unified Product Card */
.product-card { background-color: #FFFFFF; border: 1px solid #F0F0F0; padding: 0; text-align: left; margin-bottom: 20px; transition: transform 0.2s, box-shadow 0.2s; display: flex; flex-direction: column; height: 100%; }
.product-card:hover { box-shadow: 0 12px 25px rgba(0,0,0,0.08); transform: translateY(-3px); border-color: #E0E0E0; }

.card-image { width: 100%; height: 220px; background-color: #FAFAFA; display: flex; align-items: center; justify-content: center; overflow: hidden; border-bottom: 1px solid #F0F0F0; }
.card-image img { max-height: 85%; max-width: 85%; object-fit: contain; transition: transform 0.3s; }
.product-card:hover .card-image img { transform: scale(1.05); }

.card-details { padding: 15px; display: flex; flex-direction: column; flex-grow: 1; }
.brand-name { font-size: 0.75em; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
.product-title { font-size: 1.05em; font-weight: 700; color: #111; margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.short-desc { font-size: 0.8em; color: #666; line-height: 1.4; margin-bottom: 15px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }

.price-row { display: flex; justify-content: space-between; align-items: center; margin-top: auto; }
.price-tag { font-size: 1.2em; font-weight: 800; color: #000; }
.match-score { font-size: 0.75em; color: #008A00; font-weight: bold; background: #E6F4EA; padding: 4px 8px; border-radius: 6px; }

/* Section Titles */
.section-title { color: #000 !important; font-weight: 800; margin-bottom: 25px; margin-top: 10px; font-size: 1.5em; text-align: center; display: block; }

/* Hide Streamlit clutter */
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

/* Fix Expander Text Visibility */
[data-testid="stExpander"] summary { color: #000000 !important; font-weight: 600 !important; }
[data-testid="stExpander"] p { color: #000000 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    engine = VisualSearchEngine()
    num = engine.index_images()
    return engine, num

def get_dummy_price(filename):
    hash_val = sum(ord(c) for c in filename)
    return f"${(hash_val % 80) + 20}.99"

def image_to_base64(image_path):
    img = Image.open(image_path)
    img = img.resize((300, 300), Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def generate_short_description(brand, item_name):
    templates = [
        f"Premium {item_name} designed for everyday comfort and style.",
        f"Elevate your look with this classic {item_name} from {brand}.",
        f"A modern take on the traditional {item_name}, crafted for durability.",
        f"Must-have {item_name} featuring a versatile and timeless design."
    ]
    return random.choice(templates)

def display_products(results, title, show_score=True):
    if title:
        st.markdown(f'<p class="section-title">{title}</p>', unsafe_allow_html=True)
    
    cols = st.columns(4, gap="medium")
    for i, res in enumerate(results):
        with cols[i % 4]:
            raw_filename = res['filename']
            brand = raw_filename.split('_')[0].split('-')[0].title() if '_' in raw_filename else "VisualCommerce"
            title_name = raw_filename.split('_')[1].replace('.jpg','').replace('.png','').title() if len(raw_filename.split('_')) > 1 else "Classic Item"
            price = get_dummy_price(raw_filename)
            desc = generate_short_description(brand, title_name)
            
            img_base64 = image_to_base64(res['image_path'])
            score_html = f'<div class="match-score">{res["score"]:.1f}% Match</div>' if show_score else ''
            
            card_html = f'''
<div class="product-card">
<div class="card-image">
<img src="data:image/jpeg;base64,{img_base64}">
</div>
<div class="card-details">
<div class="brand-name">{brand}</div>
<div class="product-title">{title_name}</div>
<div class="short-desc">{desc}</div>
<div class="price-row">
<div class="price-tag">{price}</div>
{score_html}
</div>
</div>
</div>
'''
            st.markdown(card_html, unsafe_allow_html=True)

# ==========================================
#  MAIN APPLICATION LOGIC
# ==========================================
def main():
     # --- Render Header ---
    st.markdown("""
<div class="top-banner">FREE EXPRESS SHIPPING ON ALL ORDERS OVER $50</div>
<div class="header-container">
<div class="store-logo">VISUAL<span style="color:#E50914;">COMMERCE</span></div>
</div>
""", unsafe_allow_html=True)
     # --- Load AI Engine ---
    with st.spinner("Loading Catalog..."):
        engine, num_images = load_engine()
        
    if num_images == 0:
        st.error("⚠️ No images found.")
        st.stop()

    # Initialize State for Form
    if 'page' not in st.session_state:
        st.session_state.page = 0
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    if 'selected_cat' not in st.session_state:
        st.session_state.selected_cat = "All Categories"

    # --- TOP SEARCH BAR (USING FORM) ---
    with st.form(key='search_form', clear_on_submit=False):
        search_col, category_col, btn_col = st.columns([5, 2, 1])
        
        with search_col:
            query_input = st.text_input("Search", value=st.session_state.search_query, placeholder="🔍 Search our collection...", label_visibility="collapsed")
            
        with category_col:
            categories = set()
            for path in engine.image_paths:
                filename = os.path.basename(path)
                cat = filename.split('_')[0].split('-')[0].title()
                categories.add(cat)
            
            # Get the index of the currently selected category so the dropdown stays on it
            cat_list = ["All Categories"] + sorted(list(categories))
            try:
                cat_index = cat_list.index(st.session_state.selected_cat)
            except:
                cat_index = 0
                
            cat_input = st.selectbox("Category", cat_list, index=cat_index, label_visibility="collapsed")
            
        with btn_col:
            submit_button = st.form_submit_button("Search", use_container_width=True)

    # --- CAMERA UPLOAD (OUTSIDE FORM) ---
    with st.expander("📸 Upload an image for Visual Search"):
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    # --- LOGIC ON SUBMIT ---
    if submit_button:
        st.session_state.search_query = query_input
        st.session_state.selected_cat = cat_input
        st.session_state.page = 0 # Reset page on new search

    # Use stored state values
    query = st.session_state.search_query
    selected_category = st.session_state.selected_cat

    all_filtered_results = []
    title_text = ""
    
     # --- Determine What to Show ---
    # 1. If there is a text search
    if query:
        with st.spinner("Searching..."):
            # Get top 200 results from vector search
            raw_results = engine.search(query, top_k=200)
            
            # If a category is selected, filter the semantic results
            if selected_category != "All Categories":
                all_filtered_results = [r for r in raw_results if selected_category.lower() in r['filename'].lower()]
            else:
                all_filtered_results = raw_results
                
            title_text = f"Results for '{query}'"
            if selected_category != "All Categories":
                title_text += f" in {selected_category}"

    # 2. If there is an image search
    elif uploaded_file is not None:
        with st.spinner("Finding similar items..."):
            raw_results = engine.search_by_image(uploaded_file, top_k=200)
            if selected_category != "All Categories":
                all_filtered_results = [r for r in raw_results if selected_category.lower() in r['filename'].lower()]
            else:
                all_filtered_results = raw_results
            title_text = "Visually Similar Items"
            
    # 3. If NO search query, just show the selected category
    else:
        all_images = engine.image_paths
        if selected_category != "All Categories":
            filtered_images = [p for p in all_images if selected_category.lower() in os.path.basename(p).lower()]
            title_text = f"{selected_category} Collection"
        else:
            filtered_images = all_images
            title_text = "Featured Collection"
            
        all_filtered_results = [{"image_path": p, "filename": os.path.basename(p), "score": 0} for p in filtered_images]

    # --- PAGINATION LOGIC ---
    items_per_page = 8
    total_pages = (len(all_filtered_results) - 1) // items_per_page + 1 if all_filtered_results else 1
    
    if st.session_state.page >= total_pages:
        st.session_state.page = max(0, total_pages - 1)

    start_idx = st.session_state.page * items_per_page
    end_idx = start_idx + items_per_page
    
    paginated_results = all_filtered_results[start_idx:end_idx]

    # --- DISPLAY ---
    if paginated_results:
        show_score = True if (query or uploaded_file) else False
        display_products(paginated_results, title_text, show_score=show_score)
        
        st.markdown("<br>", unsafe_allow_html=True)
        btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
        
        with btn_col1:
            if st.session_state.page > 0:
                if st.button("⬅️ Previous Page", use_container_width=True):
                    st.session_state.page -= 1
                    st.rerun()
                    
        with btn_col3:
            if st.session_state.page < total_pages - 1:
                if st.button("Next Page ➡️", use_container_width=True):
                    st.session_state.page += 1
                    st.rerun()
                    
        with btn_col2:
            st.markdown(f"<div style='text-align: center; color: #666; margin-top: 10px;'>Page {st.session_state.page + 1} of {total_pages}</div>", unsafe_allow_html=True)
            
    else:
        st.warning(f"No products found for '{query}' in the '{selected_category}' category. Try removing the filter or searching for something else.")

if __name__ == "__main__":
    main()