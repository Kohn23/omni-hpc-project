#include <iostream>
#include <string>
#include <vector>
#include <cuda.h>


// test example
void generate_singular_values(
    const std::string& type,
    double max_val,
    double min_val,
    std::vector<double>& s_values,
    size_t rank
);

// timer
cudaEvent_t begin_evt, end_evt;

void startTimer();
float stopTimer();

// matrix output

template<typename T>
void print_device_matrix(const char* filename, size_t rows, size_t cols, const T* d_matrix_ptr, size_t ld)
{
    size_t num_elements_to_copy = rows * cols;
    std::vector<T> h_matrix_buffer(num_elements_to_copy);
    cudaError_t err = cudaMemcpy2D(
        h_matrix_buffer.data(),     
        rows * sizeof(T),           
        d_matrix_ptr,                
        ld * sizeof(T),            
        rows * sizeof(T),           
        cols,                        
        cudaMemcpyDeviceToHost
    );

    if (err != cudaSuccess) {
        printf("Failed to copy matrix from device to host: %s\n", cudaGetErrorString(err));
        return;
    }

    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        printf("Error opening file %s!\n", filename);
        return;
    }

    printf("Printing matrix (%zu x %zu) to %s...\n", rows, cols, filename);

    for(size_t i = 0; i < rows; i++) {
        for(size_t j = 0; j < cols; j++) {
            fprintf(f, "%.6f", (double)h_matrix_buffer[j * rows + i]);
            if (j == cols - 1) {
                fprintf(f, "\n");
            } else {
                fprintf(f, ",");
            }
        }
    }

    fclose(f);
    printf("Done printing to %s.\n", filename);
}