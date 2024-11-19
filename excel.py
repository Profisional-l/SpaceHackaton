import pandas as pd
from camera_data import CameraData


def parse_camera_settings(file_path):
    # Load the Excel file and sheet data
    excel_file = pd.ExcelFile(file_path)
    sheet_data = excel_file.parse('Seq1')

    # Identify the starting row for the 'Settings' section
    settings_start_index = sheet_data['Settings'].first_valid_index()

    # Variables to store common settings
    focal_length = matrix_width = matrix_height = sphere_diameter = None

    # Dictionary to store camera-specific settings
    camera_settings = {}

    # Variable to keep track of the current camera
    current_camera = None

    # Iterate through the settings section to collect data
    for index, row in sheet_data.iloc[settings_start_index:].iterrows():
        setting_name = row['Settings']
        setting_value = row['Unnamed: 7']

        if pd.notna(setting_name):
            if "all cameras" in setting_name.lower():
                # Collect common camera parameters
                current_camera = None
            elif "camera" in setting_name.lower() or ":" in setting_name:
                # Start of a new camera section
                current_camera = setting_name.strip(":")
                camera_settings[current_camera] = {}
            elif current_camera:
                # Collect settings under the current camera
                if pd.notna(setting_value):
                    camera_settings[current_camera][setting_name] = setting_value
            else:
                # Collect general settings
                if setting_name == "focal length, mm" and pd.notna(setting_value):
                    focal_length = setting_value
                elif setting_name == "ширина матрицы, mm" and pd.notna(setting_value):
                    matrix_width = setting_value
                elif setting_name == "высота матрицы, mm" and pd.notna(setting_value):
                    matrix_height = setting_value
                elif setting_name == "diameter, m" and pd.notna(setting_value):
                    sphere_diameter = setting_value

    # Convert camera settings into CameraData objects
    camera_data_objects = {}
    for camera, settings in camera_settings.items():
        x = settings.get("x, m", None)
        y = settings.get("y, m", None)
        z = settings.get("z, m", None)
        az = settings.get("azimuth, deg", None)
        sphere_diameter = 3
        # Create CameraData object
        camera_data_objects[camera] = CameraData(
            focal_length=float(focal_length),
            matrix_width=float(matrix_width),
            matrix_height=float(matrix_height),
            x=float(x),
            y=float(y),
            z=float(z),
            az=float(az),
            sphere_diameter=float(sphere_diameter)
        )

    return camera_data_objects
