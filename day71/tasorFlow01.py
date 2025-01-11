import tensorflow as tf

print(tf.__version__)

# Define some variables   variable_name=tf.Variable(value,data_type)
string = tf.Variable("this is a string", tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)

# Print the variables
print(string.numpy(), number.numpy(), floating.numpy())

# Create a rank-2 tensor    variable_name=tf.Variable(['value1'],['value2'] ,data_type)
rank_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)
print(rank_tensor.numpy())
# Print the rank of the tensor
print("Rank of rank_tensor:", tf.rank(rank_tensor))
# Print the shape of tensors
print("Shape of rank_tensor:", rank_tensor.shape)
# Create tensors filled with ones
tensor1 = tf.ones([1, 2, 3])
tensor2 = tf.reshape(tensor1, [2, 3, 1])
print("Tensor1:")
print(tensor1.numpy())
print("Tensor2 (reshaped Tensor1):")
print(tensor2.numpy())

# Define a matrix as a variable
matrix = [[1, 2, 3, 4, 5],
          [6, 7, 8, 9, 10],
          [11, 12, 13, 14, 15],
          [16, 17, 18, 19, 20]]
tensor = tf.Variable(matrix, dtype=tf.int32) #data_type is int32
# Select specific elements, rows, and columns from the tensor
three = tensor[0, 2].numpy()  # selects the 3rd element from the 1st row
print("Element at [0, 2]:", three)
row1 = tensor[0].numpy()  # selects the first row
print("First row:", row1)
column1 = tensor[:, 0].numpy()  # selects the first column
print("First column:", column1)
row_2_and_4 = tensor[1::2].numpy()  # selects second and fourth row
print("Rows 2 and 4:")
print(row_2_and_4)
column_1_in_row_2_and_3 = tensor[1:3, 0]
print(column_1_in_row_2_and_3)