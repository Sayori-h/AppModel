import pandas as pd

# Load the Excel file
file_path = '../doc/kml.xlsx'  # Update with your actual path
data = pd.read_excel(file_path)

# Format the data as "latitude,longitude,altitude" without quotes
formatted_data = data.apply(lambda row: f"{row['LON']},{row['LAT']},{int(row['ALT'])*0.3048}", axis=1)

# Save the formatted data to a text file
output_path = '../doc/converted_coordinates.txt'  # Update with your actual path
with open(output_path, 'w') as file:
    file.write("\n".join(formatted_data))

print("File saved successfully at:", output_path)
