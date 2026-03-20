import streamlit as st
import pandas as pd
import numpy as np
import folium
import joblib
from streamlit_folium import folium_static
import requests
import datetime
import plotly.express as px

# Load dataset and models
@st.cache_data
def load_data():
    data = pd.read_csv('jharkhand_route_safety_dataset_4000_with_cities.csv')
    data.fillna({'is_lit': 'No', 'crowd_density': 'Medium'}, inplace=True)
    data['is_lit'] = data['is_lit'].map({'Yes': 1, 'No': 0})
    data['crowd_density'] = data['crowd_density'].map({'Low': 0, 'Medium': 1, 'High': 2})
    data['date_time'] = pd.to_datetime(data['date_time'])
    data['hour'] = data['date_time'].dt.hour
    data['is_night'] = data['hour'].apply(lambda x: 1 if 20 <= x <= 5 else 0)
    
    # Calculate risk_score and risk_label
    data['risk_score'] = 0.6 * data['crime_rate'] + 0.4 * data['accident_rate']
    data['risk_label'] = (data['risk_score'] > data['risk_score'].quantile(0.70)).astype(int)
    
    # Clean location_name to handle variations
    data['location_name'] = data['location_name'].str.strip().str.title()
    
    return data

@st.cache_resource
def load_models():
    xgb_reg = joblib.load('xgb_risk_regressor_model.pkl')
    rf_clf = joblib.load('rf_risk_classifier_model.pkl')
    return xgb_reg, rf_clf

data = load_data()
xgb_reg, rf_clf = load_models()

# Create a mapping of city names to average coordinates
city_coords = data.groupby('location_name')[['latitude', 'longitude']].mean().to_dict('index')
available_cities = list(city_coords.keys())

# Function to get coordinates from city name (from dataset)
def get_coords(city):
    city = city.strip().title()
    if city in city_coords:
        return city_coords[city]['latitude'], city_coords[city]['longitude']
    else:
        st.error(f"City '{city}' not found in dataset. Please select a different city.")
        return None, None

# Function to approximate a route using dataset points
def approximate_route(start_coords, end_coords, data, num_points=5):
    start_lat, start_lon = start_coords
    end_lat, end_lon = end_coords
    
    route_points = data[
        (data['latitude'].between(min(start_lat, end_lat), max(start_lat, end_lat))) &
        (data['longitude'].between(min(start_lon, end_lon), max(start_lon, end_lon)))
    ]
    
    if len(route_points) < 3:
        lat_range = max(start_lat, end_lat) - min(start_lat, end_lat)
        lon_range = max(start_lon, end_lon) - min(start_lon, end_lon)
        expanded_lat_min = min(start_lat, end_lat) - 0.2 * lat_range
        expanded_lat_max = max(start_lat, end_lat) + 0.2 * lat_range
        expanded_lon_min = min(start_lon, end_lon) - 0.2 * lon_range
        expanded_lon_max = max(start_lon, end_lon) + 0.2 * lon_range
        
        route_points = data[
            (data['latitude'].between(expanded_lat_min, expanded_lat_max)) &
            (data['longitude'].between(expanded_lon_min, expanded_lon_max))
        ]
    
    route_points = route_points.sample(min(num_points, len(route_points)), random_state=42) if len(route_points) > 0 else pd.DataFrame()
    
    if len(route_points) == 0:
        start_point = data.iloc[((data['latitude'] - start_lat)**2 + (data['longitude'] - start_lon)**2).idxmin()]
        end_point = data.iloc[((data['latitude'] - end_lat)**2 + (data['longitude'] - end_lon)**2).idxmin()]
        route_points = pd.concat([pd.DataFrame([start_point]), pd.DataFrame([end_point])], ignore_index=True)
    else:
        start_point = data.iloc[((data['latitude'] - start_lat)**2 + (data['longitude'] - start_lon)**2).idxmin()]
        end_point = data.iloc[((data['latitude'] - end_lat)**2 + (data['longitude'] - end_lon)**2).idxmin()]
        route_points = pd.concat([pd.DataFrame([start_point]), route_points, pd.DataFrame([end_point])], ignore_index=True)
    
    route_points['distance_from_start'] = ((route_points['latitude'] - start_lat)**2 + (route_points['longitude'] - start_lon)**2)**0.5
    route_points = route_points.sort_values('distance_from_start').drop(columns=['distance_from_start'])
    
    return route_points

# Function to calculate route risk score and safety label
def calculate_route_risk(start_city, end_city, selected_hour, is_night, data, xgb_reg, rf_clf):
    start_coords = get_coords(start_city)
    end_coords = get_coords(end_city)
    
    if start_coords[0] is None or end_coords[0] is None:
        return None, None, None, None, None, None, None
    
    route_points = approximate_route(start_coords, end_coords, data)
    
    # Filter route points based on selected hour and is_night
    route_points = route_points[
        (route_points['hour'] == selected_hour) &
        (route_points['is_night'] == is_night)
    ]
    
    # If no points match the exact hour and is_night, use the closest hour
    if len(route_points) == 0:
        route_points = approximate_route(start_coords, end_coords, data)
        route_points['hour_diff'] = abs(route_points['hour'] - selected_hour)
        route_points = route_points.sort_values('hour_diff')
        route_points = route_points[
            (route_points['is_night'] == is_night)
        ].head(5)  # Take the top 5 closest points
        
        if len(route_points) == 0:  # If still no points, relax the is_night condition
            route_points = approximate_route(start_coords, end_coords, data)
            route_points['hour_diff'] = abs(route_points['hour'] - selected_hour)
            route_points = route_points.sort_values('hour_diff').head(5)
    
    st.write(f"Number of route points selected: {len(route_points)}")
    
    features = ['crime_rate', 'accident_rate', 'hour', 'is_night', 'is_lit', 'crowd_density']
    route_features = route_points[features]
    
    risk_scores = xgb_reg.predict(route_features)
    avg_risk_score = np.mean(risk_scores) if len(risk_scores) > 0 else 0
    
    safety_preds = rf_clf.predict(route_features)
    avg_safety_label = np.mean(safety_preds) if len(safety_preds) > 0 else 0
    model_safety_label = "Unsafe" if avg_safety_label > 0.5 else "Safe"
    
    # Time-based safety rule: Not Safe between 6:00 PM (18:00) and 6:00 AM (06:00)
    time_based_safety_label = "Not Safe" if (selected_hour >= 18 or selected_hour < 6) else "Safe"
    
    route_coords = [(start_coords[0], start_coords[1])] + list(zip(route_points['latitude'], route_points['longitude'])) + [(end_coords[0], end_coords[1])]
    
    return route_points, avg_risk_score, model_safety_label, time_based_safety_label, route_coords, start_coords, end_coords

# Streamlit App
st.title("Route Safety Predictor System")
st.write("This app predicts route safety in Jharkhand using machine learning models and visualizes risk distribution around your selected route.")

st.write("                                                      ")
col1, col2, col3 = st.columns(3)

with col1:
    start_city = st.selectbox("Start City", options=available_cities, index=available_cities.index('Main Road, Ranchi') if 'Main Road, Ranchi' in available_cities else 0, key="start_city")
    start_coords = get_coords(start_city)
    st.write(f"Selected Start City: {start_city}, Coordinates: {start_coords}")

with col2:
    end_city = st.selectbox("End City", options=available_cities, index=available_cities.index('Bazaar, Chaibasa') if 'Bazaar, Chaibasa' in available_cities else 1, key="end_city")
    end_coords = get_coords(end_city)
    st.write(f"Selected End City: {end_city}, Coordinates: {end_coords}")

with col3:
    # Default to current time: 06:26 IST on May 07, 2025
    selected_time = st.time_input("Select Time of Travel", value=datetime.time(6, 26), step=900)  # Step of 15 minutes
    selected_hour = selected_time.hour
    is_night = 1 if (selected_hour >= 20 or selected_hour <= 5) else 0
    st.write(f"Selected Time: {selected_time}, Hour: {selected_hour}, {'Night' if is_night else 'Day'}")

# Colored Dotted Plot of Risk Scores (Filtered by Selected Cities)
if start_coords[0] is not None and end_coords[0] is not None:
    st.subheader("Risk Distribution Around Your Route")
   
    # Calculate bounding box with a margin of 0.5 degrees
    margin = 0.5
    min_lat = min(start_coords[0], end_coords[0]) - margin
    max_lat = max(start_coords[0], end_coords[0]) + margin
    min_lon = min(start_coords[1], end_coords[1]) - margin
    max_lon = max(start_coords[1], end_coords[1]) + margin

    # Filter data to the bounding box
    filtered_data = data[
        (data['latitude'].between(min_lat, max_lat)) &
        (data['longitude'].between(min_lon, max_lon))
    ]

    if len(filtered_data) > 0:
        fig_dots = px.scatter(
            filtered_data, 
            x='longitude', 
            y='latitude', 
            color='risk_score',
            title=f"Risk Distribution Between {start_city} and {end_city}",
            labels={'longitude': 'Longitude', 'latitude': 'Latitude', 'risk_score': 'Risk Score'},
            color_continuous_scale='Reds',
            height=500,
            width=700
        )
        fig_dots.update_traces(marker=dict(size=8, opacity=0.7))
        st.plotly_chart(fig_dots, use_container_width=True)
    else:
        st.write("No data points available in the selected region for the risk distribution plot.")

if st.button("Predict Route Risk"):
    route_points, avg_risk_score, model_safety_label, time_based_safety_label, route_coords, start_coords, end_coords = calculate_route_risk(
        start_city, end_city, selected_hour, is_night, data, xgb_reg, rf_clf
    )
    
    if route_points is None or route_coords is None:
        st.error("Failed to calculate route. Please check your city selections.")
    else:
        st.subheader("Route Risk Prediction")
        st.write(f"**Average Risk Score**: {avg_risk_score:.2f}")
        st.write(f"**Model-Predicted Safety**: {model_safety_label}")
        st.write(f"**Time-Based Safety (6:00 PM - 6:00 AM Rule)**: {time_based_safety_label}")
        
        all_lats = [start_coords[0], end_coords[0]] + route_points['latitude'].tolist()
        all_lons = [start_coords[1], end_coords[1]] + route_points['longitude'].tolist()
        center_lat = (min(all_lats) + max(all_lats)) / 2
        center_lon = (min(all_lons) + max(all_lons)) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
        
        folium.Marker(
            [start_coords[0], start_coords[1]],
            popup=f"Start: {start_city}",
            icon=folium.Icon(color='green')
        ).add_to(m)
        
        folium.Marker(
            [end_coords[0], end_coords[1]],
            popup=f"End: {end_city}",
            icon=folium.Icon(color='red')
        ).add_to(m)
        
        for _, point in route_points.iterrows():
            popup_text = f"Risk Score: {point['risk_score']:.2f}" if 'risk_score' in point else "Risk Score: Not Available"
            folium.Marker(
                [point['latitude'], point['longitude']],
                popup=popup_text,
                icon=folium.Icon(color='orange' if point['risk_label'] == 1 else 'blue')
            ).add_to(m)
        
        # Get the OSRM route between the coordinates
        osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}?overview=full&geometries=geojson"
        response = requests.get(osrm_url)
        route_data = response.json()

        # Extract the route coordinates
        osrm_route_coords = [(lat, lon) for lon, lat in route_data['routes'][0]['geometry']['coordinates']]

        # Plot the actual road-following route
        folium.PolyLine(osrm_route_coords, color='blue', weight=5, opacity=0.8).add_to(m)

        m.fit_bounds([[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]])
        
        st.subheader("Route Map")
        folium_static(m, width=700, height=500)