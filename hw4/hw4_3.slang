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

void accumulate_constraint(
    int2 coord,
    int2 offset,
    float rest_length,
    int2 grid_size,
    float3 pos_i,
    float w_i,
    StructuredBuffer<float3> positions_in,
    StructuredBuffer<float> inv_masses,
    inout float3 total_corr,
    inout int count)
{
    int2 n_coord = coord + offset;
    if (n_coord.x < 0 || n_coord.x >= grid_size.x ||
        n_coord.y < 0 || n_coord.y >= grid_size.y) {
        return;
    }

    int j = n_coord.y * grid_size.x + n_coord.x;
    float w_j = inv_masses[j];
    float w_sum = w_i + w_j;
    if (w_sum <= 0.0) {
        return;
    }

    float3 d = positions_in[j] - pos_i;
    float dist = length(d);
    if (dist > 1e-6) {
        total_corr += (w_i / w_sum) * (dist - rest_length) * (d / dist);
    }
    count += 1;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         uniform int2 grid_size,
                         uniform float2 spacing) {
    if (tid >= grid_size.x * grid_size.y) {
        return;
    }
    // TODO: HW 4.3. Your code below.
    float3 pos_i = positions_in[tid];
    float w_i = inv_masses[tid];

    // Fixed particles should never move.
    if (w_i == 0.0) {
        positions_out[tid] = pos_i;
        return;
    }

    int2 coord = int2(int(tid) % grid_size.x, int(tid) / grid_size.x);
    float3 total_corr = float3(0.0, 0.0, 0.0);
    int count = 0;

    float diag_rest = length(spacing);

    // Structural constraints (immediate 4 neighbors)
    accumulate_constraint(coord, int2(-1,  0), spacing.x, grid_size, pos_i, w_i, positions_in, inv_masses, total_corr, count);
    accumulate_constraint(coord, int2( 1,  0), spacing.x, grid_size, pos_i, w_i, positions_in, inv_masses, total_corr, count);
    accumulate_constraint(coord, int2( 0, -1), spacing.y, grid_size, pos_i, w_i, positions_in, inv_masses, total_corr, count);
    accumulate_constraint(coord, int2( 0,  1), spacing.y, grid_size, pos_i, w_i, positions_in, inv_masses, total_corr, count);

    // Shearing constraints (4 diagonal neighbors)
    accumulate_constraint(coord, int2(-1, -1), diag_rest, grid_size, pos_i, w_i, positions_in, inv_masses, total_corr, count);
    accumulate_constraint(coord, int2( 1, -1), diag_rest, grid_size, pos_i, w_i, positions_in, inv_masses, total_corr, count);
    accumulate_constraint(coord, int2(-1,  1), diag_rest, grid_size, pos_i, w_i, positions_in, inv_masses, total_corr, count);
    accumulate_constraint(coord, int2( 1,  1), diag_rest, grid_size, pos_i, w_i, positions_in, inv_masses, total_corr, count);

    // Bending constraints (4 distance-2 neighbors)
    accumulate_constraint(coord, int2(-2,  0), 2.0 * spacing.x, grid_size, pos_i, w_i, positions_in, inv_masses, total_corr, count);
    accumulate_constraint(coord, int2( 2,  0), 2.0 * spacing.x, grid_size, pos_i, w_i, positions_in, inv_masses, total_corr, count);
    accumulate_constraint(coord, int2( 0, -2), 2.0 * spacing.y, grid_size, pos_i, w_i, positions_in, inv_masses, total_corr, count);
    accumulate_constraint(coord, int2( 0,  2), 2.0 * spacing.y, grid_size, pos_i, w_i, positions_in, inv_masses, total_corr, count);

    if (count > 0) {
        pos_i += total_corr / float(count);
    }

    positions_out[tid] = pos_i;
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
