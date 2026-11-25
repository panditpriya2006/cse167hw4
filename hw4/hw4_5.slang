
[shader("compute")]
[numthreads(64, 1, 1)]
void detect_collision(uint tid : SV_DispatchThreadID,
                      StructuredBuffer<float3> positions,
                      StructuredBuffer<float> inv_masses,
                      StructuredBuffer<float4> planes,
                      StructuredBuffer<float4> spheres,
                      RWStructuredBuffer<float4> contact_planes,
                      uniform uint num_particles,
                      uniform uint num_planes,
                      uniform uint num_spheres,
                      uniform float collision_distance,
                      uniform float contact_margin) {
    if (tid >= num_particles) {
        return;
    }

    float3 pos = positions[tid];

    // Start with no recorded collision.
    float min_dist = contact_margin + collision_distance;
    float4 best_contact = float4(0.0, 0.0, 0.0, 0.0);

    // Check all plane colliders.
    for (uint i = 0; i < num_planes; i++) {
        float4 plane = planes[i];

        float3 n = plane.xyz;
        float d = plane.w;

        float signed_dist = dot(n, pos) - d;

        if (signed_dist < min_dist) {
            min_dist = signed_dist;
            best_contact = plane;
        }
    }

    // Check all sphere colliders.
    for (uint i = 0; i < num_spheres; i++) {
        float4 sphere = spheres[i];

        float3 center = sphere.xyz;
        float radius = sphere.w;

        float3 direction = pos - center;
        float dist = length(direction);

        float signed_dist = dist - radius;

        if (signed_dist < min_dist) {
            min_dist = signed_dist;

            float3 n;
            if (dist > 1e-6) {
                n = direction / dist;
            } else {
                n = float3(0.0, 1.0, 0.0);
            }

            float d = dot(n, center) + radius;

            best_contact = float4(n, d);
        }
    }

    // Save the contact plane if a collision is nearby.
    if (min_dist < contact_margin + collision_distance) {
        best_contact.w += collision_distance;
        contact_planes[tid] = best_contact;
    } else {
        contact_planes[tid] = float4(0.0, 0.0, 0.0, 0.0);
    }
}


[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         StructuredBuffer<float3> old_positions,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         StructuredBuffer<uint> row_ptr,
                         StructuredBuffer<uint> neighbors,
                         StructuredBuffer<float> neighbor_rest,
                         StructuredBuffer<float4> contact_planes,
                         uniform uint num_particles,
                         uniform float static_friction,
                         uniform float kinetic_friction) {
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

    // Part 1: Distance constraints.
    float3 total_corr = float3(0.0, 0.0, 0.0);
    uint count = 0;

    uint start = row_ptr[tid];
    uint end = row_ptr[tid + 1];

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

    // Average the spring corrections.
    if (count > 0) {
        pos_i += total_corr / float(count);
    }

    // Part 2: Contact and friction.
    float4 contact = contact_planes[tid];
    float3 n = contact.xyz;

    // A zero normal means there is no contact.
    if (dot(n, n) > 0.5) {

        // Calculate penetration into the contact plane.
        float penetration = max(contact.w - dot(n, pos_i), 0.0);

        // Correct movement in the normal direction.
        pos_i += penetration * n;

        // Calculate movement along the tangent plane.
        float3 displacement = pos_i - old_positions[tid];

        float3 tangent =
            displacement - dot(displacement, n) * n;

        float tangent_len = length(tangent);

        // Apply friction if there is tangential movement.
        if (tangent_len > 1e-6) {

            if (tangent_len <= static_friction * penetration) {
                // Static friction completely stops tangential motion.
                pos_i -= tangent;
            } else {
                // Kinetic friction reduces tangential motion.
                float friction_amount =
                    min(tangent_len, kinetic_friction * penetration);

                pos_i -= (friction_amount / tangent_len) * tangent;
            }
        }
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

    // Reconstruct velocity using the corrected position.
    velocities[tid] =
        (positions[tid] - old_positions[tid]) / dt;
}

