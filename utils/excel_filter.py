import csv


##############################################################################
#delete part of the zero when continuous zero is more than 10
##############################################################################
# # Load the CSV file
# file_path = 'driving_log.csv'
# with open(file_path, 'r') as f:
#     reader = csv.reader(f)
#     next(reader)  # Skip the header row
#     data = list(reader)

# # Initialize variables to track consecutive zeros and the start index of consecutive zero segments
# consecutive_zeros = 0
# start_index = None

# # A list to store the indices of the rows to keep
# rows_to_keep = []

# # Iterate over the rows in the data
# for index, row in enumerate(data):
#     if float(row[3]) == 0:  # Assume the fourth column is 0-based
#         if consecutive_zeros == 0:
#             start_index = index
#         consecutive_zeros += 1
#     else:
#         if consecutive_zeros > 10:
#             rows_to_keep.extend([start_index, start_index + 1, index - 2 ,index - 1, index])
#         elif consecutive_zeros > 0 and consecutive_zeros <= 10:
#             rows_to_keep.extend(range(start_index, index + 1))
#         else:
#             rows_to_keep.append(index)  # Include non-zero rows
#         consecutive_zeros = 0

# # Check if there are trailing consecutive zero segments that need to be included
# if consecutive_zeros > 10:
#     rows_to_keep.extend([start_index, start_index + 1])

# if consecutive_zeros > 0 and consecutive_zeros <= 10:
#     rows_to_keep.extend(range(start_index, index + 1))

# # Create a new list with the selected rows
# filtered_data = [data[i] for i in rows_to_keep]

# print(f"Original data has {len(data)} rows")
# print(f"Filtered data has {len(filtered_data)} rows")

# # Save the modified data to a new CSV file
# output_file_path = 'modified_csv_file.csv'
# with open(output_file_path, 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(filtered_data)

# print(f"Modified CSV file has been saved to {output_file_path}")

##############################################################################
#delete part of the zero when continuous zero is more than 10  END
##############################################################################


##############################################################################
#delete all the rows if it's fouth column is zero
##############################################################################

# Load the CSV file
input_file_path = 'driving_log.csv'
output_file_path = 'modified_csv_file.csv'


with open(input_file_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row
    data = list(reader)

# A list to store the indices of the rows to keep
rows_to_keep = []

# Iterate over the rows in the data
for index, row in enumerate(data):
    if float(row[3]) != 0:  # Assume the fourth column is 0-based
        rows_to_keep.append(index)  # Include non-zero rows

# Create a new list with the selected rows
filtered_data = [data[i] for i in rows_to_keep]

print(f"Original data has {len(data)} rows")
print(f"Filtered data has {len(filtered_data)} rows")

# Save the modified data to a new CSV file

with open(output_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(filtered_data)

print(f"Modified CSV file has been saved to {output_file_path}")

##############################################################################
#delete all the rows if it's fouth column is zero END
##############################################################################