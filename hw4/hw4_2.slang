
[shader("compute")]
[numthreads(64, 1, 1)]
void predict(uint tid : SV_DispatchThreadID,
             RWStructuredBuffer<float3> positions,
             RWStructuredBuffer<float3> old_positions,
             RWStructuredBuffer<float3> velocities,
             StructuredBuffer<float> inv_masses,
             uniform uint num_particles,
             uniform float dt) {
    if (tid >= num_particles) {
        return;
    }
    // Save the position before this timestep.
    old_positions[tid] = positions[tid];
    // A particle with zero inverse mass is fixed.
    if (inv_masses[tid] == 0.0) {
        velocities[tid] = float3(0.0, 0.0, 0.0);
        return;
    }
    // Apply gravity to velocity.
    float3 gravity = float3(0.0, -9.8, 0.0);
    velocities[tid] += gravity * dt;
    // Predict the new position using the updated velocity.
    positions[tid] += velocities[tid] * dt;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         StructuredBuffer<float> rest_lengths,
                         uniform uint num_particles) {
    if (tid >= num_particles) {
        return;
    }
    float3 current_pos = positions_in[tid];
    // Fixed particles should never move.
    if (inv_masses[tid] == 0.0) {
        positions_out[tid] = current_pos;
        return;
    }
    float3 correction_sum = float3(0.0, 0.0, 0.0);
    uint neighbor_count = 0;
    // Check the particle to the left (previous particle).
    if (tid > 0) {
        uint j = tid - 1;
        float3 d = positions_in[j] - current_pos;
        float dist = length(d);
        float wi = inv_masses[tid];
        float wj = inv_masses[j];
        if (dist > 1e-6 && wi + wj > 0.0) {
            float rest_length = rest_lengths[j];

            float3 correction =
                (wi / (wi + wj)) * (dist - rest_length) * (d / dist);

            correction_sum += correction;
        }

        neighbor_count++;
    }

    // Check the particle to the right (next particle).
    if (tid + 1 < num_particles) {
        uint j = tid + 1;

        float3 d = positions_in[j] - current_pos;
        float dist = length(d);

        float wi = inv_masses[tid];
        float wj = inv_masses[j];

        if (dist > 1e-6 && wi + wj > 0.0) {
            float rest_length = rest_lengths[tid];

            float3 correction =
                (wi / (wi + wj)) * (dist - rest_length) * (d / dist);

            correction_sum += correction;
        }

        neighbor_count++;
    }
    // Average the corrections from the neighboring particles.
    if (neighbor_count > 0) {
        current_pos += correction_sum / float(neighbor_count);
    }
    positions_out[tid] = current_pos;
}
[shader("compute")]
[numthreads(64, 1, 1)]
void update_velocity(uint tid : SV_DispatchThreadID,
                     RWStructuredBuffer<float3> positions,
                     StructuredBuffer<float3> old_positions,
                     RWStructuredBuffer<float3> velocities,
                     StructuredBuffer<float> inv_masses,
                     uniform uint num_particles,
                     uniform float dt) {
    if (tid >= num_particles) {
        return;
    }
    // Fixed particles do not move.
    if (inv_masses[tid] == 0.0) {
        velocities[tid] = float3(0.0, 0.0, 0.0);
        return;
    }
    // Keep particles above the floor.
    if (positions[tid].y < -0.97) {
        positions[tid].y = -0.97;
    }
    // Reconstruct velocity from the corrected position.
    velocities[tid] = (positions[tid] - old_positions[tid]) / dt;
}

