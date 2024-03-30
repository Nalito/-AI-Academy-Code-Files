# Import pandas library
import pandas as pd

# Create Pandas Series
my_list = [1, 2, 3, 4, 5]
my_series = pd.Series(my_list)
print("Pandas Series: {}".format(my_series))



# Creating a DataFrame from a dictionary
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Emma'],
        'Age': [25, 30, 35, 40, 45],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']}
df = pd.DataFrame(data)

# Displaying the DataFrame
print("DataFrame:")
print(df)

# Basic DataFrame operations

# Viewing shape of DataFrame
print("\nShape of DataFrame: {}".format(df.shape))

# Viewing first few rows
print("\nFirst few rows:")
print(df.head())

# Viewing last few rows
print("\nLast few rows:")
print(df.tail())

# Accessing columns
print("\nAccessing columns:")
print(df['Name'])

# Accessing rows by index
print("\nAccessing rows by index:")
print(df.iloc[0])  # First row

# Slicing a dataframe
print("\nSlicing a DataFrame:")
print(df.iloc[0:2])  # First two rows
print(df[["Name", "Age"]])

# Filtering rows
print("\nFiltering rows:")
print(df[df['Age'] > 30])  # Selecting rows where Age is greater than 30

# Adding a new column
df['Gender'] = ['Female', 'Male', 'Male', 'Male', 'Female']
print("\nDataFrame with a new column:")
print(df)

# Dropping a column
df.drop('City', axis=1, inplace=True)
print("\nDataFrame after dropping a column:")
print(df)

# Renaming columns
df.rename(columns={'Name': 'Full Name'}, inplace=True)
print("\nDataFrame after renaming column:")
print(df)

# Handling missing values
data_with_missing = {'A': [1, 2, None, 4],
                     'B': [None, 6, 7, 8]}
df_with_missing = pd.DataFrame(data_with_missing)
print("\nDataFrame with missing values:")
print(df_with_missing)

# Filling missing values
df_with_missing.fillna(0, inplace=True)
print("\nDataFrame after filling missing values:")
print(df_with_missing)

# Aggregating data
print("\nAggregating data:")
print(df.groupby('Gender')['Age'].mean())  # Calculating mean age by gender

# Concatenating two datasets
data1 = {'A': [1, 2, 3],
         'B': [4, 5, 6]}
data2 = {'A': [7, 8, 9],
            'B': [10, 11, 12]}
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df_concatenated = pd.concat([df1, df2])
print("\nConcatenated DataFrame:")
print(df_concatenated)

# Reading from and writing to files
df.to_csv('data.csv', index=False)  # Writing to a CSV file
df_from_file = pd.read_csv('data.csv')  # Reading from a CSV file
print("\nDataFrame from file:")
print(df_from_file)

# Importing NumPy library
import numpy as np

# Creating NumPy arrays

# Creating a 1D array
arr_1d = np.array([1, 2, 3, 4, 5])
print("1D Array:")
print(arr_1d)

# Creating a 2D array (matrix)
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\n2D Array:")
print(arr_2d)

# Read image as numpy array


# Basic array attributes

# Shape of the array
print("\nShape of the array:")
print(arr_2d.shape)

# Number of dimensions
print("\nNumber of dimensions:")
print(arr_2d.ndim)

# Data type of the elements
print("\nData type of the elements:")
print(arr_2d.dtype)

# Array indexing and slicing

# Accessing elements
print("\nAccessing elements:")
print(arr_2d[0, 0])  # First element
print(arr_2d[1, 1])  # Middle element

# Slicing
print("\nSlicing:")
print(arr_2d[:, :2])  # First two columns
print(arr_2d[1:, 1:])  # Bottom-right submatrix

# Array operations

# Create a sequence of of values with linspace
arr = np.linspace(1, 10, 10)
print("\nSequence of values:")
print(arr)

# Flatten array 
arr = np.array([[1, 2, 3], [4, 5, 6]]) 
flat_arr = arr.flatten() 

print ("Original array:\n", arr) 
print ("Fattened array:\n", flat_arr)

# Arithmetic operations

# basic operations on single array 
a = np.array([1, 2, 5, 3]) 

# add 1 to every element 
print ("Adding 1 to every element:", a+1) 

# subtract 3 from each element 
print ("Subtracting 3 from each element:", a-3) 

# multiply each element by 10 
print ("Multiplying each element by 10:", a*10) 

# square each element 
print ("Squaring each element:", a**2) 

# modify existing array 
a *= 2
print ("Doubled each element of original array:", a) 

# transpose of array 
a = np.array([[1, 2, 3], [3, 4, 5], [9, 6, 0]]) 

print ("\nOriginal array:\n", a) 
print ("Transpose of array:\n", a.T) 


print("\nArithmetic operations:")
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
print("Addition:")
print(arr1 + arr2)
print("Multiplication:")
print(arr1 * arr2)

# Broadcasting
print("\nBroadcasting:")
arr3 = np.array([10, 20])
print(arr1 + arr3)

# Universal functions (ufuncs)
print("\nUniversal functions (ufuncs):")
print(np.sqrt(arr1))
print(np.exp(arr1))

# Array manipulation

# Reshaping arrays
print("\nReshaping arrays:")
arr = np.arange(1, 10)
print("Original array:")
print(arr)
reshaped_arr = arr.reshape(3, 3)
print("Reshaped array:")
print(reshaped_arr)

# Transposing arrays
print("\nTransposing arrays:")
print("Original array:")
print(arr_2d)
transposed_arr = arr_2d.T
print("Transposed array:")
print(transposed_arr)

# Stacking arrays
print("\nStacking arrays:")
stacked_arr = np.vstack((arr1, arr2))
print("Vertical stack:")
print(stacked_arr)
stacked_arr = np.hstack((arr1, arr2))
print("Horizontal stack:")
print(stacked_arr)

# Random number generation
print("\nRandom number generation:")
rand_arr = np.random.randint(1, 100, size=(3, 3))
print(rand_arr)

# Saving and loading arrays
np.save('saved_array.npy', rand_arr)  # Saving array to file
loaded_arr = np.load('saved_array.npy')  # Loading array from file
print("\nLoaded array:")
print(loaded_arr)

# Creating a matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Displaying the matrix
print("Matrix:")
print(matrix)

# Basic matrix operations

# Matrix shape
print("\nMatrix shape:", matrix.shape)

# Matrix transpose
print("\nMatrix transpose:")
print(matrix.T)

# Matrix addition
print("\nMatrix addition:")
matrix_addition = matrix + matrix
print(matrix_addition)

# Matrix subtraction
print("\nMatrix subtraction:")
matrix_subtraction = matrix - matrix
print(matrix_subtraction)

# Matrix multiplication (element-wise)
print("\nMatrix multiplication (element-wise):")
matrix_elementwise_mul = matrix * matrix
print(matrix_elementwise_mul)

# Matrix multiplication (dot product)
print("\nMatrix multiplication (dot product):")
matrix_dot_product = np.dot(matrix, matrix)
print(matrix_dot_product)

# Scalar multiplication
print("\nScalar multiplication:")
scalar = 2
matrix_scalar_mul = scalar * matrix
print(matrix_scalar_mul)

# Matrix slicing
print("\nMatrix slicing:")
print("First row:", matrix[0])
print("Second column:", matrix[:, 1])
print("Sub-matrix:", matrix[:2, :2])

# Matrix operations and functions

# Sum of all elements
print("\nSum of all elements:", np.sum(matrix))

# Mean of all elements
print("Mean of all elements:", np.mean(matrix))

# Max and min elements
print("Maximum element:", np.max(matrix))
print("Minimum element:", np.min(matrix))

# Reshaping matrix
print("\nReshaping matrix:")
reshaped_matrix = np.reshape(matrix, (1, 9))
print(reshaped_matrix)

# Finding the determinant of a matrix
print("\nDeterminant of matrix:")
determinant = np.linalg.det(matrix)
print(determinant)

# Finding the inverse of a matrix
matrix = np.array([[1, 2],
                    [3, 4]])
print("\nInverse of matrix:")
inverse_matrix = np.linalg.inv(matrix)
print(inverse_matrix)

# Solving linear equations
A = np.array([[2, 1], [1, 1]])
b = np.array([4, 3])
print("\nSolving linear equations:")
x = np.linalg.solve(A, b)
print("Solution:", x)
