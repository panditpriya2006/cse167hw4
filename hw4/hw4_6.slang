
struct BVHNode {
    float3 bbox_min;
    float3 bbox_max;

    // Which BVH node we should skip to
    // if this node is not intersecting.
    int next_index;

    uint triangle_begin;
    uint triangle_end;

    bool is_leaf;
};

float point_aabb_distance(float3 x, float3 bmin, float3 bmax) {
    return length(x - clamp(x, bmin, bmax));
}

struct ClosestPoint {
    float3 position;
    float3 bary;
};

// From https://iquilezles.org/articles/distfunctions/
ClosestPoint closest_point_triangle(
        float3 p, float3 a, float3 b, float3 c) {
    float3 ba = b - a, pa = p - a;
    float3 cb = c - b, pb = p - b;
    float3 ac = a - c, pc = p - c;
    float3 nor = cross(ba, ac);

    float t0 = clamp(dot(ba, pa) / dot(ba, ba), 0.0, 1.0);
    float t1 = clamp(dot(cb, pb) / dot(cb, cb), 0.0, 1.0);
    float t2 = clamp(dot(ac, pc) / dot(ac, ac), 0.0, 1.0);

    float3 q0 = a + ba * t0;
    float3 q1 = b + cb * t1;
    float3 q2 = c + ac * t2;

    if (sign(dot(cross(ba, nor), pa)) +
        sign(dot(cross(cb, nor), pb)) +
        sign(dot(cross(ac, nor), pc)) < 2.0) {

        float d0 = dot(p - q0, p - q0);
        float d1 = dot(p - q1, p - q1);
        float d2 = dot(p - q2, p - q2);

        if (d0 < d1 && d0 < d2)
            return {q0, float3(1.0 - t0, t0, 0.0)};
        if (d1 < d2)
            return {q1, float3(0.0, 1.0 - t1, t1)};
        return {q2, float3(t2, 0.0, 1.0 - t2)};
    }

    float inv_nor2 = 1.0 / dot(nor, nor);
    float3 q = p - nor * (dot(nor, pa) * inv_nor2);

    float wb = dot(cross(pa, ac), nor) * inv_nor2;
    float wc = dot(cross(pa, ba), nor) * inv_nor2;
    return {q, float3(1.0 - wb - wc, wb, wc)};
}


[shader("compute")]
[numthreads(64, 1, 1)]
void detect_collision(uint tid : SV_DispatchThreadID,
                      StructuredBuffer<float3> positions,
                      StructuredBuffer<float> inv_masses,
                      StructuredBuffer<float4> planes,
                      StructuredBuffer<float4> spheres,
                      StructuredBuffer<float3> mesh_vertices,
                      StructuredBuffer<float3> mesh_vertex_normals,
                      StructuredBuffer<uint3> mesh_triangles,
                      StructuredBuffer<BVHNode> bvh_nodes,
                      RWStructuredBuffer<float4> contact_planes,
                      uniform uint num_particles,
                      uniform uint num_planes,
                      uniform uint num_spheres,
                      uniform uint num_bvh_nodes,
                      uniform float collision_distance,
                      uniform float contact_margin) {

    if (tid >= num_particles) {
        return;
    }

    float3 pos = positions[tid];

    // Maximum distance at which we consider a contact.
    float contact_threshold = contact_margin + collision_distance;

    // Track the closest signed distance and its contact plane.
    float min_dist = contact_threshold;
    float4 best_contact = float4(0.0, 0.0, 0.0, 0.0);

    // --------------------------------------------------
    // Part 1: Check plane colliders.
    // --------------------------------------------------

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

    // --------------------------------------------------
    // Part 2: Check sphere colliders.
    // --------------------------------------------------

    for (uint i = 0; i < num_spheres; i++) {
        float4 sphere = spheres[i];

        float3 center = sphere.xyz;
        float radius = sphere.w;

        float3 direction = pos - center;
        float dist = length(direction);

        float signed_dist = dist - radius;

        if (signed_dist < min_dist) {
            min_dist = signed_dist;

            // Calculate the outward-facing sphere normal.
            float3 n;

            if (dist > 1e-6) {
                n = direction / dist;
            } else {
                n = float3(0.0, 1.0, 0.0);
            }

            // Tangent plane at the closest point on the sphere.
            float d = dot(n, center) + radius;

            best_contact = float4(n, d);
        }
    }

    // --------------------------------------------------
    // Part 3: Find the closest triangle using the BVH.
    // --------------------------------------------------

    // The minimum unsigned distance to a triangle.
    float min_mesh_dist = contact_threshold;

    // Store the closest point and its triangle's normal.
    float3 closest_position = float3(0.0, 0.0, 0.0);
    float3 closest_normal = float3(0.0, 0.0, 0.0);

    bool found_triangle = false;

    // Stackless BVH traversal starts at the root.
    uint node_index = 0;

    while (node_index < num_bvh_nodes) {

        BVHNode node = bvh_nodes[node_index];

        // Calculate the distance from the particle to this node's AABB.
        float bbox_dist = point_aabb_distance(
            pos,
            node.bbox_min,
            node.bbox_max
        );

        // If the bounding box is too far away, skip its subtree.
        if (bbox_dist > min_mesh_dist) {

            if (node.next_index < 0) {
                break;
            }

            node_index = uint(node.next_index);
            continue;
        }

        // If this is a leaf, check its triangles.
        if (node.is_leaf) {

            for (uint j = node.triangle_begin;
                 j < node.triangle_end;
                 j++) {

                uint3 tri = mesh_triangles[j];

                float3 p0 = mesh_vertices[tri.x];
                float3 p1 = mesh_vertices[tri.y];
                float3 p2 = mesh_vertices[tri.z];

                // Find the closest point on this triangle.
                ClosestPoint cp = closest_point_triangle(
                    pos, p0, p1, p2
                );

                float dist = length(pos - cp.position);

                // Keep the closest triangle found so far.
                if (dist < min_mesh_dist) {

                    min_mesh_dist = dist;
                    closest_position = cp.position;

                    // Calculate the geometric face normal.
                    float3 ng = normalize(
                        cross(p1 - p0, p2 - p0)
                    );

                    // Interpolate the vertex normals using barycentric
                    // coordinates from the closest point.
                    float3 n0 = mesh_vertex_normals[tri.x];
                    float3 n1 = mesh_vertex_normals[tri.y];
                    float3 n2 = mesh_vertex_normals[tri.z];

                    float3 ns = normalize(
                        cp.bary.x * n0 +
                        cp.bary.y * n1 +
                        cp.bary.z * n2
                    );

                    // Make the geometric normal point in the same
                    // direction as the interpolated vertex normal.
                    if (dot(ng, ns) < 0.0) {
                        ng = -ng;
                    }

                    closest_normal = ng;
                    found_triangle = true;
                }
            }
        }

        // Move to the next node in preorder traversal.
        node_index++;
    }

    // --------------------------------------------------
    // Part 4: Convert the closest triangle into a contact plane.
    // --------------------------------------------------

    if (found_triangle) {

        // Treat the closest triangle as an infinite plane.
        float signed_dist = dot(
            closest_normal,
            pos - closest_position
        );

        if (signed_dist < min_dist) {

            min_dist = signed_dist;

            float d = dot(
                closest_normal,
                closest_position
            );

            best_contact = float4(closest_normal, d);
        }
    }

    // --------------------------------------------------
    // Part 5: Save the contact plane.
    // --------------------------------------------------

    if (min_dist < contact_threshold) {

        // Shift the contact plane outward by collision_distance.
        best_contact.w += collision_distance;

        contact_planes[tid] = best_contact;

    } else {

        // No collision.
        contact_planes[tid] = float4(0.0, 0.0, 0.0, 0.0);
    }
}

