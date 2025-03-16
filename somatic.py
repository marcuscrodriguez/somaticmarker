import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pygame
import time
import pandas as pd
from PIL import Image

# Initialize pygame for audio playback
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# Title of the app
st.header("Synthetic Somatic Markers:")
st.write("Heuristic Cognition Based on Quantum Artificial Emotional Intelligence")
st.write("By: Marcus C. Rodriguez www.marcusc.com")

# Ensure session state keys exist before accessing them
if "current_image_index" not in st.session_state:
    st.session_state["current_image_index"] = 0
if "outcome_displayed" not in st.session_state:
    st.session_state["outcome_displayed"] = False
if "outcome_text" not in st.session_state:
    st.session_state["outcome_text"] = ""
if "audio_played" not in st.session_state:
    st.session_state["audio_played"] = False
if "results" not in st.session_state:
    st.session_state["results"] = []  # Initialize as an empty list

# Stimulus Data
stimulus_data = [
    {
        "id": 1, "image": "stimulus1.png", "audio": "audio1.wav",
        "description": ["A child lost in a forest at night"],
        "emotion": ["Afraid", "Brave"],
        "decision": ["Flee", "Fight"],
        "behavior": ["Run", "Pick up a large stick"],
        "outcomes": ["Wolf chases, catches and attacks", "Wolf runs away"]
    },
    {
        "id": 2, "image": "stimulus2.png", "audio": "audio2.wav",
        "description": ["Confronted by an aggressive stranger"],
        "emotion": ["Fear", "Anger"],
        "decision": ["Flee", "Fight"],
        "behavior": ["Avoid eye contact, walk away", "Make eye contact, walk towards"],
        "outcomes": ["Aggressive stranger ignores you", "Aggressive stranger assaults you"]
    },
    {
        "id": 3, "image": "stimulus3.png", "audio": "audio3.wav",
        "description": ["Witnessing someone drowning"],
        "emotion": ["Helpless", "Empowered"],
        "decision": ["Shutdown", "Engage"],
        "behavior": ["Ignore, walk away", "Attempt to Rescue"],
        "outcomes": ["The person drowns", "The drowning person is saved"]
    },
    {
        "id": 4, "image": "stimulus4.png", "audio": "audio4.wav",
        "description": ["Public speaking in front of a large audience"],
        "emotion": ["Anxiety", "Confidence"],
        "decision": ["Shutdown", "Engage"],
        "behavior": ["Freeze, forgetting words", "Speak with conviction"],
        "outcomes": ["Audience laughs, chides and boos", "Audience applauds"]
    },
    {
        "id": 5, "image": "stimulus5.png", "audio": "audio5.wav",
        "description": ["Seeing a helpless animal suffering"],
        "emotion": ["Innocence", "Guilt"],
        "decision": ["Shutdown", "Engage"],
        "behavior": ["Drive away, (I didn't hit it)", "Pick up dog, take it to the vet"],
        "outcomes": ["Dog dies", "You save the dog's life"]
    },
    {
        "id": 6, "image": "stimulus6.png", "audio": "audio6.wav",
        "description": ["Boy stuck in a burning building"],
        "emotion": ["Panic", "Confront"],
        "decision": ["Flee", "Fight"],
        "behavior": ["Leave him and get yourself out of the burning building", "Attempt to rescue the boy"],
        "outcomes": ["You escape the fire, the boy is rescued by fire dept.", "You and the boy die in the fire"]
    },
    {
        "id": 7, "image": "stimulus7.png", "audio": "audio7.wav",
        "description": ["Betrayed by a close friend"],
        "emotion": ["Disbelief", "Belief"],
        "decision": ["Flee", "Fight"],
        "behavior": ["Disengage, walk away and reflect", "Confront aggressively"],
        "outcomes": ["The relationship is salvaged", "The relationship ends, bitterly"]
    },
    {
        "id": 8, "image": "stimulus8.png", "audio": "audio8.wav",
        "description": ["Witnessing social injustice"],
        "emotion": ["Trust", "Disgust"],
        "decision": ["Shutdown", "Engage"],
        "behavior": ["Ignore, and keep walking", "Take a stand intervene"],
        "outcomes": ["You watch later on the news, and feel guilty", "You get arrested"]
    },
    {
        "id": 9, "image": "stimulus9.png", "audio": "audio9.wav",
        "description": ["Facing financial ruin, lost it all"],
        "emotion": ["Sadness", "Happy"],
        "decision": ["Shutdown", "Engage"],
        "behavior": ["Jump", "Get off the roof, go back inside and start over"],
        "outcomes": ["Suicidal Death", "Experience the freedom of nothing left to lose and rebuilding"]
    },
    {
        "id": 10, "image": "stimulus10.png", "audio": "audio10.wav",
        "description": ["Discovering an intruder in the home"],
        "emotion": ["Fear", "Anger"],
        "decision": ["Flee", "Fight"],
        "behavior": ["Run out the back door", "Fight the intruder"],
        "outcomes": ["You get away, police arrive a few minutes later", "The intruder breaks your nose"]
    },
]

#Plot Block Sphere Function
def plot_bloch_sphere(ax, theta, phi, title):
    """Plots a Bloch sphere representation of a single qubit state."""
    
    # Create the Bloch sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color='c', alpha=0.1, edgecolor='k')

    # Axes labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Convert dx, dy, dz to Bloch sphere coordinates
    x_vec = np.sin(theta) * np.cos(phi)
    y_vec = np.sin(theta) * np.sin(phi)
    z_vec = np.cos(theta)

    # Plot the qubit vector
    ax.quiver(0, 0, 0, x_vec, y_vec, z_vec, color='r', linewidth=2, arrow_length_ratio=0.1)

    # Set limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

# Load Stimulus Data
current_image_index = st.session_state["current_image_index"]
current_stimulus = stimulus_data[current_image_index]

# Display Image
image = Image.open(current_stimulus["image"])
st.image(image, caption=f"Situation {current_image_index + 1}: {current_stimulus['description'][0]}", use_container_width=True)
st.header(current_stimulus["description"][0])

# Play Audio Function
def play_audio(audio_path):
    """Play audio only once per stimulus image."""
    if not st.session_state["audio_played"]:  
        if os.path.exists(audio_path):
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            st.session_state["audio_played"] = True  # Mark audio as played

# Play Audio (Only Once Per Stimulus)
play_audio(current_stimulus["audio"])

# Dynamically Retrieve Labels
emotion_labels = current_stimulus["emotion"]
decision_labels = current_stimulus["decision"]

# Emotion Slider
st.markdown(f"<div style='display: flex; justify-content: space-between; width: 100%;'><span>{emotion_labels[0]}</span><span>{emotion_labels[1]}</span></div>", unsafe_allow_html=True)
dx = st.slider('emotion', -1.0, 1.0, 0.0, 0.1, label_visibility="hidden", key='emotion')

# Decision Slider
st.markdown(f"<div style='display: flex; justify-content: space-between; width: 100%;'><span>{decision_labels[0]}</span><span>{decision_labels[1]}</span></div>", unsafe_allow_html=True)
dy = st.slider('decision', -1.0, 1.0, 0.0, 0.1, label_visibility="hidden", key='decision')

# Compute Behavior (dz) based on E and D vectors
def compute_behavior(E, D, threshold=0.5):
    """Compute behavior (dz) based on E and D interaction"""
    M = np.sqrt(E**2 + D**2)
    if M <= threshold:
        return -1  # Neutral zone defaults to negative action
    return np.sign(E + D)

# Outcome Category Mapping
positive_outcomes = {
    "Wolf runs away", "Aggressive stranger ignores you", "The drowning person is saved", "Audience applauds", "You save the dog's life", 
    "You escape the fire, the boy is rescued by fire dept.", "The relationship is salvaged", "You get arrested", "Experience the freedom of nothing left to lose and rebuilding",
    "You get away, police arrive a few minutes later"
}
    
negative_outcomes = {
    "Wolf chases, catches and attacks", "Aggressive stranger assaults you", "The person drowns",
    "Audience laughs, chides and boos", "Dog dies", "You and the boy die in the fire",
    "The relationship ends, bitterly", "You watch later on the news, and feel guilty",
    "You get arrested", "Suicidal Death", "The intruder breaks your nose"
}

# Submit Response & Show Outcome
if st.button("Submit Response"):
    # Compute Behavior
    dz = compute_behavior(dx, dy)
    behavior_label = current_stimulus["behavior"][int((dz + 1) / 2)]
    
    # Determine Outcome Index
    if dz < 0:
        outcome_index = 0
    else:
        outcome_index = 1

    # Retrieve the corresponding outcome text
    outcome_text = current_stimulus["outcomes"][outcome_index]

    # Assign Outcome Type
    if outcome_text in positive_outcomes:
        outcome_type = "Positive"
    else:
        outcome_type = "Negative"

    # Display Results
    st.write(f"Computed Vectors: Emotion (dx): {dx}, Decision (dy): {dy}, Behavior (dz): {dz}, Outcome: {outcome_type}")
    st.text(f"### Behavior: {behavior_label}")
    st.text(f"### Outcome: {outcome_text}")

    # Play Outcome Audio
    def play_outcome_audio(outcome_type):
        """Play the corresponding audio file based on outcome type."""
        outcome_audio_map = {
            "Positive": "positive_outcome.wav",
            "Negative": "negative_outcome.wav"
        }
        audio_path = outcome_audio_map[outcome_type]

        if not os.path.exists(audio_path):
            st.error(f"Audio file not found: {audio_path}")
            return

        try:
            pygame.mixer.quit()
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
            sound = pygame.mixer.Sound(audio_path)
            sound.play()
        except pygame.error as e:
            st.error(f"Error playing WAV file: {e}")

    play_outcome_audio(outcome_type)
    
    # Initialize results storage if not present
    if "results" not in st.session_state:
    	st.session_state["results"] = []
    
    # Define DSI Mapping for Standard Cases (Default)
    dsi_mapping = {
    	0: {"Emotion": "feel bad about self", "Decision": "bad", "Behavior": "pain"},
    	1: {"Emotion": "feel good about self", "Decision": "good", "Behavior": "pleasure"}
    }

    # Custom DSI Mapping for Outlier Cases (#6, #8, #2/#10)
    dsi_outlier_mapping = {
    	2: {  # Situation #2 (Flipped Mapping)
    		0: {"Emotion": "feel good about self", "Decision": "good", "Behavior": "pleasure"},
    		1: {"Emotion": "feel bad about self", "Decision": "bad", "Behavior": "pain"}
    	},    	  	

    	6: {  # Situation #6
        	0: {"Emotion": "feel bad about self", "Decision": "good", "Behavior": "pleasure"},
        	1: {"Emotion": "feel good about self", "Decision": "bad", "Behavior": "pain"}
    	},
    	
    	8: {  # Situation #8
        	0: {"Emotion": "feel bad about self", "Decision": "bad", "Behavior": "pleasure"},
        	1: {"Emotion": "feel good about self", "Decision": "good", "Behavior": "pain"}
    	},
    	
    	10: {  # Situation #10 (Flipped Mapping)
    		0: {"Emotion": "feel good about self", "Decision": "good", "Behavior": "pleasure"},
    		1: {"Emotion": "feel bad about self", "Decision": "bad", "Behavior": "pain"}
    	}
    }

    # Determine Stimulus Outcome Index
    stimulus_index = current_stimulus["id"]  # Match the Situation ID
    outcome_index = 0 if dz < 0 else 1  # Map `dz` to 0 (negative) or 1 (positive)
    
    # Fetch the Correct DSI Values
    if stimulus_index in dsi_outlier_mapping:
    	dsi_values = dsi_outlier_mapping[stimulus_index][outcome_index]  # Use custom mapping
    else:
    	dsi_values = dsi_mapping[outcome_index]  # Use default mapping

    # Store the response for this question
    st.session_state["results"].append({
    	"Question": current_stimulus["id"],
    	"dx (Emotion)": dx,
    	"dy (Decision)": dy,
    	"dz (Behavior)": dz,
    	"Outcome": outcome_text,
    	"DSI Emotion": dsi_values["Emotion"],
    	"DSI Decision": dsi_values["Decision"],
    	"DSI Behavior": dsi_values["Behavior"],
    	"Memory Encoding": f"({dx}, {dy}, {dz})"
    })
    
    # Map dx, dy, dz to Bloch sphere angles (theta, phi)
    E_theta, E_phi = np.pi * (1 - dx) / 2, np.pi * dx
    D_theta, D_phi = np.pi * (1 - dy) / 2, np.pi * dy
    B_theta, B_phi = np.pi * (1 - dz) / 2, np.pi * dz
    
    # Create figure and subplots for three Bloch spheres
    fig = plt.figure(figsize=(12, 4))
    
    ax1 = fig.add_subplot(131, projection='3d')
    plot_bloch_sphere(ax1, E_theta, E_phi, "Emotion (E)")
    
    ax2 = fig.add_subplot(132, projection='3d')
    plot_bloch_sphere(ax2, D_theta, D_phi, "Decision (D)")
    
    ax3 = fig.add_subplot(133, projection='3d')
    plot_bloch_sphere(ax3, B_theta, B_phi, "Behavior (B)")
    
    # Display updated Bloch spheres in Streamlit    
    st.subheader("Quantum Heuristic Model of Somatic Markers")
    st.write("(SNS-fight/flight vs PNS-shutdown)")
    st.write("Bloch sphere / Emotion (E), Decision (D), and Behavior (B), evolving dynamically based on user responses.")    
    st.pyplot(fig)
    
    # Pause for 20 seconds before transitioning
    time.sleep(6)    

    # Move to the next image
    st.session_state["current_image_index"] = (st.session_state["current_image_index"] + 1) % len(stimulus_data)
    st.session_state["audio_played"] = False
    st.session_state["outcome_displayed"] = False\
    
    # Force UI refresh to load the next image
    st.rerun()
    
# Divider
st.markdown("<hr style='border: 5px solid; border-image-source: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet); border-image-slice: 1;'>", unsafe_allow_html=True)

# Calculate remaining situations
remaining_situations = 10 - st.session_state["current_image_index"]

# Display message to user
st.warning(f"You have **{remaining_situations}** situations left to answer.")

# Display summary table if all ten questions have been answered
if st.session_state["current_image_index"] == 0 and len(st.session_state["results"]) >= 10:
    st.write("### Summary of Responses")

    # Convert results to DataFrame
    df_results = pd.DataFrame(st.session_state["results"])

    # Display results table using Streamlit's built-in method
    st.dataframe(df_results, use_container_width=True)

    # Ensure the session state for cumulative DSI is initialized
    if "cumulative_dsi_list" not in st.session_state:
        st.session_state["cumulative_dsi_list"] = []

    # Compute cumulative DSI directly from recorded results
    cumulative_dsi_values = []
    cumulative_dsi = 0  # Start with zero

    # Iterate through all 10 situations
    for i in range(10):
        # Retrieve the recorded DSI Emotion (text field)
        dsi_text_value = df_results.iloc[i]["DSI Emotion"]  

        # Convert DSI emotion text to numerical value
        if dsi_text_value == "feel bad about self":
            dsi_value = -1.5  # Negative reinforcement weighs more
        elif dsi_text_value == "feel good about self":
            dsi_value = 1.0   # Positive reinforcement
        else:
            dsi_value = 0  # Handle unexpected values

        # Accumulate DSI
        cumulative_dsi += dsi_value
        cumulative_dsi_values.append(cumulative_dsi)

        # Debugging: Print step-by-step calculations
        st.write(f"DEBUG: Situation {i+1} | DSI Emotion: {dsi_text_value} | DSI Value: {dsi_value} | Cumulative DSI: {cumulative_dsi}")

    # Store cumulative DSI in session state
    st.session_state["cumulative_dsi_list"] = cumulative_dsi_values

    # Generate x-axis labels (Situations 1-10)
    situation_steps = list(range(1, 11))

    # Create the figure for cumulative DSI visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(situation_steps, cumulative_dsi_values, label="Cumulative DSI (Feel Good/Bad About Self)", linestyle="-", marker="o", color="purple")

    # Labeling and formatting
    ax.set_xlabel("Situation Number")
    ax.set_ylabel("Cumulative DSI Score")
    ax.set_title("Cumulative Developing Self Image (DSI) Over Situations")
    ax.set_xticks(situation_steps)
    ax.legend()
    ax.grid(True)

    # Display plot in Streamlit (after loop finishes)
    st.pyplot(fig)

    # Stop execution so the user can explore the table and chart
    st.stop()







