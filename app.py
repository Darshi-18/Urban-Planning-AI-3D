import streamlit as st
import numpy as np
import pyvista as pv
from scipy.spatial import Voronoi
import cv2
import os
import torch
import torchvision.transforms as transforms
from models import Generator  # Import your U-Net architecture

# Page Optimization Setup
st.set_page_config(page_title="AI 3D Urban Architect Engine", layout="wide")
st.title("🗺️ Generative AI 3D Urban Planning System")
st.write("Upload a square satellite screenshot of any unused land to dynamically synthesize an individualized 3D architectural master plan grid.")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------
# 1. CORE GENERATIVE AI LAYER: CACHED AI INITIALIZATION
# -----------------------------------------------------------------
@st.cache_resource
def load_generative_ai_model():
    model_path = "generator_urban.pth"
    if os.path.exists(model_path):
        ai_model = Generator().to(DEVICE)
        try:
            ai_model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        except TypeError:
            ai_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        ai_model.eval()
        return ai_model
    return None

ai_model = load_generative_ai_model()

# -----------------------------------------------------------------
# 2. RUNTIME GRAPHICS AND ANALYSIS ENGINE
# -----------------------------------------------------------------
def generate_3d_twin_mesh(image_path):
    orig_img = cv2.imread(image_path)
    h, w, _ = orig_img.shape

    # Extract topographical features
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blurred, 30, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Compute Adaptive Seeding Fingerprint
    np.random.seed(int(np.sum(orig_img) % 100000))
    num_hubs = 50
    points = np.random.rand(num_hubs, 2) * [w, h]

    for cnt in contours[::2]:
        if cv2.contourArea(cnt) > 200:
            for pt in cnt[::12]:
                points = np.vstack([points, pt.reshape(2)])

    vor = Voronoi(points)

    # Executing Generative AI Layout Inference
    if ai_model is not None:
        pil_img = transforms.ToPILImage()(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        transform_pipeline = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        input_tensor = transform_pipeline(pil_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            generated_tensor = ai_model(input_tensor)
            
        ai_output = (generated_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) + 1) / 2
        ai_output = np.clip(ai_output * 255, 0, 255).astype(np.uint8)
        ai_output = cv2.resize(ai_output, (w, h))
        ai_gray = cv2.cvtColor(ai_output, cv2.COLOR_RGB2GRAY)
        _, ai_dev_mask = cv2.threshold(ai_gray, 180, 255, cv2.THRESH_BINARY)
    else:
        _, ai_dev_mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Initialize PyVista Scene Plotter for Browser Integration
    pv.start_xvfb() # Soft framebuffer for background server compilation
    pl = pv.Plotter(notebook=True)
    pl.background_color = '#0e1013'
    pl.enable_eye_dome_lighting()

    # Apply 3D Satellite Texture to Floor Plane
    texture = pv.read_texture(image_path)
    floor = pv.Plane(center=(w/2, h/2, -0.2), direction=(0, 0, 1), i_size=w, j_size=h)
    floor.active_t_coords = np.zeros((floor.n_points, 2))
    floor.active_t_coords[:, 0] = floor.points[:, 0] / w
    floor.active_t_coords[:, 1] = 1.0 - (floor.points[:, 1] / h)
    pl.add_mesh(floor, texture=texture, opacity=0.9, lighting=True)

    road_lines_computed = []

    # Building Sector Meshes
    for region in vor.regions:
        if not -1 in region and len(region) > 3:
            poly_2d = np.array([vor.vertices[i] for i in region])
            x_c, y_c = np.mean(poly_2d, axis=0).astype(int)
            
            if 0 <= x_c < w and 0 <= y_c < h:
                if edges[y_c, x_c] > 0: # Environmental Zones 🟩
                    center = np.mean(poly_2d, axis=0)
                    shrunk_poly = np.array([center + 0.90 * (p - center) for p in poly_2d])
                    num_pts = len(shrunk_poly)
                    faces = np.hstack([num_pts, list(range(num_pts))])
                    points_3d = np.zeros((num_pts, 3))
                    points_3d[:, :2] = shrunk_poly
                    try:
                        park = pv.PolyData(points_3d, faces=faces).extrude((0, 0, 1.5), capping=True)
                        pl.add_mesh(park, color='#3b945e', opacity=0.75, show_edges=True, edge_color='#225937')
                    except: pass
                    continue

                # Query AI Zoning Prediction Matrix to determine heights
                is_commercial_zone = ai_dev_mask[y_c, x_c] > 0 and (x_c + y_c) % 3 == 0
                center = np.mean(poly_2d, axis=0)
                
                if is_commercial_zone: # Commercial Sky-rises 🟥
                    shrunk_poly = np.array([center + 0.80 * (p - center) for p in poly_2d])
                    num_pts = len(shrunk_poly)
                    faces = np.hstack([num_pts, list(range(num_pts))])
                    points_3d = np.zeros((num_pts, 3))
                    points_3d[:, :2] = shrunk_poly
                    try:
                        comm = pv.PolyData(points_3d, faces=faces).extrude((0, 0, np.random.uniform(55, 100)), capping=True)
                        pl.add_mesh(comm, color='#d94141', opacity=0.85, show_edges=True, edge_color='#401010')
                    except: pass
                else: # Residential Parcels 🟦
                    for i in range(2):
                        for j in range(2):
                            sx = 0.4 if i == 0 else -0.4
                            sy = 0.4 if j == 0 else -0.4
                            sub_center = center + [sx * (w/20), sy * (h/20)]
                            sub_poly = np.array([sub_center + 0.35 * (p - center) for p in poly_2d])
                            if np.all(sub_poly >= 0):
                                num_pts = len(sub_poly)
                                faces = np.hstack([num_pts, list(range(num_pts))])
                                points_3d = np.zeros((num_pts, 3))
                                points_3d[:, :2] = sub_poly
                                try:
                                    house = pv.PolyData(points_3d, faces=faces).extrude((0, 0, np.random.uniform(15, 30)), capping=True)
                                    pl.add_mesh(house, color='#3a86c8', opacity=0.85, show_edges=True, edge_color='#103050')
                                    road_lines_computed.append((center, sub_center))
                                except: pass

    # Render Infrastructure Highways & Access Roads
    for ridge_points in vor.ridge_vertices:
        if -1 not in ridge_points:
            p1, p2 = vor.vertices[ridge_points[0]], vor.vertices[ridge_points[1]]
            if 0 <= p1[0] < w and 0 <= p2[0] < w:
                try:
                    highway = pv.MultipleLines(points=np.array([[p1[0], p1[1], 0.8], [p2[0], p2[1], 0.8]]))
                    pl.add_mesh(highway, color='#ffffff', line_width=3)
                except: pass

    for connection in road_lines_computed:
        try:
            street = pv.MultipleLines(points=np.array([[connection[0][0], connection[0][1], 0.6], [connection[1][0], connection[1][1], 0.6]]))
            pl.add_mesh(street, color='#545b66', line_width=2)
        except: pass

    return pl

# -----------------------------------------------------------------
# 3. INTERACTIVE WEB USER INTERFACE FRONTEND
# -----------------------------------------------------------------
if ai_model is None:
    st.warning("⚠️ Neural weights file 'generator_urban.pth' not detected. Running system on default algorithmic configuration mode.")

uploaded_file = st.file_uploader("Select square land coordinate image map...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Save the file temporarily to read dimensions natively
    temp_path = "temp_uploaded_land.png"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1. Target Land Footprint")
        st.image(temp_path, use_container_width=True)
        
    with col2:
        st.subheader("2. Fused 3D Architectural Digital Twin Viewport")
        with st.spinner("Generative deep learning logic computing urban layout matrices..."):
            # Compute and extract the 3D pyvista engine viewer configuration
            plotter_instance = generate_3d_twin_mesh(temp_path)
            
            # Embed the full, rotatable, dynamic WebGL viewport block on screen
            st.pyvista_canvas(plotter_instance, use_container_width=True, height=600)
            st.success("Urban master plan generated successfully!")
