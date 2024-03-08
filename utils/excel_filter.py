import csv
import random
import os
import sys
sys.path.append("..")
import config 

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
#define the function to delete some of the rows if it's fouth column is zero
##############################################################################

# define delete zero function

def delete_zero(input_file_path, output_file_path):
# #get the path of file
# current_path = os.getcwd()
# parent_path = os.path.dirname(current_path)
# input_file_path = parent_path + '/data/test/driving_log.csv'

# # Load the CSV file
# output_file_path = parent_path + '/data/test/modified_csv_file.csv'
    
    with open(input_file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        data = list(reader)

    # A list to store the indices of the rows to keep
    rows_to_keep = []

    zero_row_index = []

    # Iterate over the rows in the data
    for index, row in enumerate(data):
        if float(row[3]) != 0:  # Assume the fourth column is 0-based
            rows_to_keep.append(index)  # Include non-zero rows
    # count how many rows have zero in the fourth column
        else:
            zero_row_index.append(index)

    # compare the number of rows with zero in the fourth column with 
    #the number of rows without zero in the fourth column
    # if the number of rows with zero in the fourth column is more than the number of rows without zero in the fourth column
    # randomly delete some index in the zero_row_index,so that zero_row_index and rows_to_keep have the same length
    if len(zero_row_index) > len(rows_to_keep):
        random.shuffle(zero_row_index)
        zero_row_index = zero_row_index[:len(rows_to_keep)]

    # add the index in zero_row_index to rows_to_keep
    rows_to_keep.extend(zero_row_index)

    # sort the index in rows_to_keep
    rows_to_keep.sort()


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
#define the function to delete some of the rows if it's fouth column is zero END
##############################################################################
    




##############################################################################
#use the function to delete some of the rows if it's fouth column is zero
##############################################################################
# get the path of file
current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
data_path = parent_path + '/data/{}'.format(config.DATA_TYPE)
# find all directories in the data path
sub_dirs = os.listdir(data_path)
print('having data: ' + str(sub_dirs))

# for each directory, find the driving_log.csv file and delete some of the rows if it's fouth column is zero
for sub_dir in sub_dirs:
    # if data_path + '/{}/driving_log.csv'.format(sub_dir) exists
    if os.path.isfile(data_path + '/{}/driving_log.csv'.format(sub_dir)):
        print('processing csv excel in {}'.format(sub_dir))
        # use the function to delete some of the rows if it's fouth column is zero
        input_file_path = data_path + '/{}/driving_log.csv'.format(sub_dir)
        output_file_path = data_path + '/{}/modified_driving_log.csv'.format(sub_dir)
        delete_zero(input_file_path, output_file_path)
        print(' ')




##############################################################################
#use the function to delete some of the rows if it's fouth column is zero END
##############################################################################