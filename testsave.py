import numpy as np

# Example arrays
array1 = np.array([1.1, 2, 3])
array2 = np.array([4, 5.121212, 6])

# Stack arrays vertically (as columns)
combined_array = np.column_stack((array1, array2))

# Save to a CSV file
np.savetxt('combined_arrays.csv', combined_array, delimiter=' ', fmt='%.6f')
