#include<utils.h>

void generate_singular_values(
    const std::string& type,
    double max_val,
    double min_val,
    std::vector<double>& s_values,
    size_t rank)
{
    s_values.resize(rank);
    if (rank == 0) return;

    if (type == "geometric") {
        printf("Generating GEOMETRICALLY distributed singular values...\n");
        if (rank == 1) {
            s_values[0] = max_val;
        } else {
            double log_max = log(max_val);
            double log_min = log(min_val);
            double step = (log_min - log_max) / (double)(rank - 1);
            for (size_t i = 0; i < rank; ++i) {
                s_values[i] = exp(log_max + (double)i * step);
            }
        }
    }
    else if (type == "geometric_zero") {
        printf("Generating GEOMETRICALLY distributed singular values with an internal ZERO block...\n");
        if (rank == 1) {
            s_values[0] = max_val;
        } else {
            size_t zero_count = std::max<size_t>(1, rank / 6);
            size_t zero_start = (rank > zero_count) ? (rank - zero_count) / 2 : 0;
            size_t zero_end = std::min(rank, zero_start + zero_count);
            size_t nonzero_count = rank - (zero_end - zero_start);

            double min_positive = (min_val > 0.0) ? min_val : max_val * 1e-6;
            double log_max = log(max_val);
            double log_min = log(min_positive);
            double step = (nonzero_count > 1) ? (log_min - log_max) / (double)(nonzero_count - 1) : 0.0;

            size_t nz_pos = 0;
            for (size_t i = 0; i < rank; ++i) {
                if (i >= zero_start && i < zero_end) {
                    s_values[i] = 0.0;
                } else {
                    double val = (nonzero_count == 1) ? max_val : exp(log_max + (double)nz_pos * step);
                    s_values[i] = val;
                    ++nz_pos;
                }
            }
        }
    }
    else if (type == "uniform") {
        printf("Generating UNIFORMLY (linearly) distributed singular values...\n");
        if (rank == 1) {
            s_values[0] = max_val;
        } else {
            double step = (max_val - min_val) / (double)(rank - 1);
            for (size_t i = 0; i < rank; ++i) {
                s_values[i] = max_val - (double)i * step;
            }
        }
    }
    else if (type == "cluster0") {
        printf("Generating 'Cluster0' singular values (sharp drop)...\n");
        size_t cutoff_rank = rank / 4;
        if (cutoff_rank == 0 && rank > 0) cutoff_rank = 1;
        
        double high_end_val = max_val * 0.9;
        double step = (cutoff_rank > 1) ? (max_val - high_end_val) / (double)(cutoff_rank - 1) : 0.0;
        
        for (size_t i = 0; i < rank; ++i) {
            if (i < cutoff_rank) {
                s_values[i] = max_val - (double)i * step;
            } else {
                s_values[i] = min_val;
            }
        }
    }
    else if (type == "cluster1") {
        printf("Generating 'Cluster1' singular values (staircase)...\n");
        size_t cutoff1 = rank / 3;
        size_t cutoff2 = 2 * rank / 3;
        double mid_val = (max_val + min_val) / 2.0;

        for (size_t i = 0; i < rank; ++i) {
            if (i < cutoff1) {
                s_values[i] = max_val;
            } else if (i < cutoff2) {
                s_values[i] = mid_val;
            } else {
                s_values[i] = min_val;
            }
        }
    }
    else if (type == "arithmetic") {
        printf("Generating ARITHMETIC progression singular values...\n");
        if (rank == 1) {
            s_values[0] = max_val;
        } else {
            double step = (max_val - min_val) / (double)(rank - 1);
            for (size_t i = 0; i < rank; ++i) {
                s_values[i] = max_val - (double)i * step;
            }
        }
    }
    else if (type == "normal") {
        printf("Generating NORMAL (Gaussian-like) distributed singular values...\n");
        double mean = (double)rank / 2.0;
        double sigma = (double)rank / 6.0; 
        
        for (size_t i = 0; i < rank; ++i) {
            double x = (double)i;
            double gaussian_weight = exp(-0.5 * pow((x - mean) / sigma, 2.0));
            s_values[i] = min_val + (max_val - min_val) * gaussian_weight;
        }
    }

}


void startTimer()
{
    cudaEventCreate(&begin_evt);
    cudaEventRecord(begin_evt);
    cudaEventCreate(&end_evt);
}

float stopTimer()
{
    cudaEventRecord(end_evt);
    cudaEventSynchronize(end_evt);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, begin_evt, end_evt);
    cudaEventDestroy(begin_evt);
    cudaEventDestroy(end_evt);
    return milliseconds;
}