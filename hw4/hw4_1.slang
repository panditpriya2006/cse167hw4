[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(uint tid : SV_DispatchThreadID,
                  RWStructuredBuffer<float3> positions,
                  RWStructuredBuffer<float3> velocities,
                  uniform uint num_particles,
                  // time step size
                  uniform float dt) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.1. Your code below.

    float3 pos = positions[tid];
    float3 vel = velocities[tid];

    // Symplectic Euler update
    float3 gravity = float3(0.0, -9.8, 0.0);
    vel += gravity * dt;
    pos += vel * dt;

    // Floor collision at y = -0.97
    if (pos.y < -0.97) {
        pos.y = -0.97;
        vel.y = -vel.y;
    }

    positions[tid] = pos;
    velocities[tid] = vel;
}
