[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         // The CSR matrix data structure
                         StructuredBuffer<uint> row_ptr,
                         StructuredBuffer<uint> neighbors,
                         // Rest lengths of the neighbor edges
                         StructuredBuffer<float> neighbor_rest,
                         uniform uint num_particles) {
    if (tid >= num_particles) {
        return;
    }

    float3 pos_i = positions_in[tid];
    float w_i = inv_masses[tid];

    // Fixed particles should never move.
    if (w_i == 0.0) {
        positions_out[tid] = pos_i;
        return;
    }

    float3 total_corr = float3(0.0, 0.0, 0.0);
    uint count = 0;

    // Get the range of this particle's neighbors from the CSR arrays.
    uint start = row_ptr[tid];
    uint end = row_ptr[tid + 1];

    // Calculate the correction from each neighboring particle.
    for (uint k = start; k < end; k++) {
        uint j = neighbors[k];

        float w_j = inv_masses[j];
        float w_sum = w_i + w_j;

        if (w_sum <= 0.0) {
            continue;
        }

        float3 d = positions_in[j] - pos_i;
        float dist = length(d);

        if (dist > 1e-6) {
            float rest_length = neighbor_rest[k];

            float3 correction =
                (w_i / w_sum) *
                (dist - rest_length) *
                (d / dist);

            total_corr += correction;
            count++;
        }
    }

    // Average all neighbor corrections.
    if (count > 0) {
        pos_i += total_corr / float(count);
    }

    positions_out[tid] = pos_i;
}
