/*
 * Copyright (c) 2015-2016 The Khronos Group Inc.
 * Copyright (c) 2015-2016 Valve Corporation
 * Copyright (c) 2015-2016 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Relicensed from the WTFPL (http://www.wtfpl.net/faq/).
 */

#ifndef LINMATH_H
#define LINMATH_H

#include <math.h>
#include <iostream>
#include <iomanip>

// Converts degrees to radians.
#define degreesToRadians(angleDegrees) (angleDegrees * M_PI / 180.0)

// Converts radians to degrees.
#define radiansToDegrees(angleRadians) (angleRadians * 180.0 / M_PI)

typedef float vec3[3];
static inline void vec3_add(vec3 r, vec3 const a, vec3 const b)
{
    int i;
    for (i = 0; i < 3; ++i)
        r[i] = a[i] + b[i];
}
static inline void vec3_sub(vec3 r, vec3 const a, vec3 const b)
{
    int i;
    for (i = 0; i < 3; ++i)
        r[i] = a[i] - b[i];
}
static inline void vec3_scale(vec3 r, vec3 const v, float const s)
{
    int i;
    for (i = 0; i < 3; ++i)
        r[i] = v[i] * s;
}
static inline float vec3_mul_inner(vec3 const a, vec3 const b)
{
    float p = 0.f;
    int i;
    for (i = 0; i < 3; ++i)
        p += b[i] * a[i];
    return p;
}
static inline void vec3_mul_cross(vec3 r, vec3 const a, vec3 const b)
{
    r[0] = a[1] * b[2] - a[2] * b[1];
    r[1] = a[2] * b[0] - a[0] * b[2];
    r[2] = a[0] * b[1] - a[1] * b[0];
}
static inline float vec3_len(vec3 const v) { return sqrtf(vec3_mul_inner(v, v)); }
static inline void vec3_norm(vec3 r, vec3 const v)
{
    float k = 1.f / vec3_len(v);
    vec3_scale(r, v, k);
}
static inline void vec3_reflect(vec3 r, vec3 const v, vec3 const n)
{
    float p = 2.f * vec3_mul_inner(v, n);
    int i;
    for (i = 0; i < 3; ++i)
        r[i] = v[i] - p * n[i];
}

typedef float vec4[4];
static inline void vec4_add(vec4 r, vec4 const a, vec4 const b)
{
    int i;
    for (i = 0; i < 4; ++i)
        r[i] = a[i] + b[i];
}
static inline void vec4_sub(vec4 r, vec4 const a, vec4 const b)
{
    int i;
    for (i = 0; i < 4; ++i)
        r[i] = a[i] - b[i];
}
static inline void vec4_scale(vec4 r, vec4 v, float s)
{
    int i;
    for (i = 0; i < 4; ++i)
        r[i] = v[i] * s;
}
static inline float vec4_mul_inner(vec4 a, vec4 b)
{
    float p = 0.f;
    int i;
    for (i = 0; i < 4; ++i)
        p += b[i] * a[i];
    return p;
}
static inline void vec4_mul_cross(vec4 r, vec4 a, vec4 b)
{
    r[0] = a[1] * b[2] - a[2] * b[1];
    r[1] = a[2] * b[0] - a[0] * b[2];
    r[2] = a[0] * b[1] - a[1] * b[0];
    r[3] = 1.f;
}
static inline float vec4_len(vec4 v) { return sqrtf(vec4_mul_inner(v, v)); }
static inline void vec4_norm(vec4 r, vec4 v)
{
    float k = 1.f / vec4_len(v);
    vec4_scale(r, v, k);
}
static inline void vec4_reflect(vec4 r, vec4 v, vec4 n)
{
    float p = 2.f * vec4_mul_inner(v, n);
    int i;
    for (i = 0; i < 4; ++i)
        r[i] = v[i] - p * n[i];
}

typedef vec4 mat4x4[4];
static inline void mat4x4_identity(mat4x4 M)
{
    int i, j;
    for (i = 0; i < 4; ++i)
        for (j = 0; j < 4; ++j)
            M[i][j] = i == j ? 1.f : 0.f;
}
static inline void mat4x4_dup(mat4x4 M, mat4x4 N)
{
    int i, j;
    for (i = 0; i < 4; ++i)
        for (j = 0; j < 4; ++j)
            M[i][j] = N[i][j];
}
static inline void mat4x4_row(vec4 r, mat4x4 M, int i)
{
    int k;
    for (k = 0; k < 4; ++k)
        r[k] = M[k][i];
}
static inline void mat4x4_col(vec4 r, mat4x4 M, int i)
{
    int k;
    for (k = 0; k < 4; ++k)
        r[k] = M[i][k];
}
static inline void mat4x4_transpose(mat4x4 M, mat4x4 N)
{
    int i, j;
    for (j = 0; j < 4; ++j)
        for (i = 0; i < 4; ++i)
            M[i][j] = N[j][i];
}
static inline void mat4x4_add(mat4x4 M, mat4x4 a, mat4x4 b)
{
    int i;
    for (i = 0; i < 4; ++i)
        vec4_add(M[i], a[i], b[i]);
}
static inline void mat4x4_sub(mat4x4 M, mat4x4 a, mat4x4 b)
{
    int i;
    for (i = 0; i < 4; ++i)
        vec4_sub(M[i], a[i], b[i]);
}
static inline void mat4x4_scale(mat4x4 M, mat4x4 a, float k)
{
    int i;
    for (i = 0; i < 4; ++i)
        vec4_scale(M[i], a[i], k);
}
static inline void mat4x4_scale_aniso(mat4x4 M, mat4x4 a, float x, float y, float z)
{
    int i;
    vec4_scale(M[0], a[0], x);
    vec4_scale(M[1], a[1], y);
    vec4_scale(M[2], a[2], z);
    for (i = 0; i < 4; ++i)
    {
        M[3][i] = a[3][i];
    }
}
static inline void mat4x4_mul(mat4x4 M, mat4x4 a, mat4x4 b)
{
    int k, r, c;
    for (c = 0; c < 4; ++c)
        for (r = 0; r < 4; ++r)
        {
            M[c][r] = 0.f;
            for (k = 0; k < 4; ++k)
                M[c][r] += a[k][r] * b[c][k];
        }
}
static inline void mat4x4_mul_vec4(vec4 r, mat4x4 M, vec4 v)
{
    int i, j;
    for (j = 0; j < 4; ++j)
    {
        r[j] = 0.f;
        for (i = 0; i < 4; ++i)
            r[j] += M[i][j] * v[i];
    }
}
static inline void mat4x4_translate(mat4x4 T, float x, float y, float z)
{
    mat4x4_identity(T);
    T[3][0] = x;
    T[3][1] = y;
    T[3][2] = z;
}
static inline void mat4x4_translate_in_place(mat4x4 M, float x, float y, float z)
{
    vec4 t = {x, y, z, 0};
    vec4 r;
    int i;
    for (i = 0; i < 4; ++i)
    {
        mat4x4_row(r, M, i);
        M[3][i] += vec4_mul_inner(r, t);
    }
}
static inline void mat4x4_from_vec3_mul_outer(mat4x4 M, vec3 a, vec3 b)
{
    int i, j;
    for (i = 0; i < 4; ++i)
        for (j = 0; j < 4; ++j)
            M[i][j] = i < 3 && j < 3 ? a[i] * b[j] : 0.f;
}
static inline void mat4x4_rotate(mat4x4 R, mat4x4 M, float x, float y, float z, float angle)
{
    float s = sinf(angle);
    float c = cosf(angle);
    vec3 u = {x, y, z};

    if (vec3_len(u) > 1e-4)
    {
        vec3_norm(u, u);
        mat4x4 T;
        mat4x4_from_vec3_mul_outer(T, u, u);

        mat4x4 S = {{0, u[2], -u[1], 0}, {-u[2], 0, u[0], 0}, {u[1], -u[0], 0, 0}, {0, 0, 0, 0}};
        mat4x4_scale(S, S, s);

        mat4x4 C;
        mat4x4_identity(C);
        mat4x4_sub(C, C, T);

        mat4x4_scale(C, C, c);

        mat4x4_add(T, T, C);
        mat4x4_add(T, T, S);

        T[3][3] = 1.;
        mat4x4_mul(R, M, T);
    }
    else
    {
        mat4x4_dup(R, M);
    }
}
static inline void mat4x4_rotate_X(mat4x4 Q, mat4x4 M, float angle)
{
    float s = sinf(angle);
    float c = cosf(angle);
    mat4x4 R = {{1.f, 0.f, 0.f, 0.f}, {0.f, c, s, 0.f}, {0.f, -s, c, 0.f}, {0.f, 0.f, 0.f, 1.f}};
    mat4x4_mul(Q, M, R);
}
static inline void mat4x4_rotate_Y(mat4x4 Q, mat4x4 M, float angle)
{
    float s = sinf(angle);
    float c = cosf(angle);
    mat4x4 R = {{c, 0.f, s, 0.f}, {0.f, 1.f, 0.f, 0.f}, {-s, 0.f, c, 0.f}, {0.f, 0.f, 0.f, 1.f}};
    mat4x4_mul(Q, M, R);
}
static inline void mat4x4_rotate_Z(mat4x4 Q, mat4x4 M, float angle)
{
    float s = sinf(angle);
    float c = cosf(angle);
    mat4x4 R = {{c, s, 0.f, 0.f}, {-s, c, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}, {0.f, 0.f, 0.f, 1.f}};
    mat4x4_mul(Q, M, R);
}
static inline void mat4x4_invert(mat4x4 T, mat4x4 M)
{
    float s[6];
    float c[6];
    s[0] = M[0][0] * M[1][1] - M[1][0] * M[0][1];
    s[1] = M[0][0] * M[1][2] - M[1][0] * M[0][2];
    s[2] = M[0][0] * M[1][3] - M[1][0] * M[0][3];
    s[3] = M[0][1] * M[1][2] - M[1][1] * M[0][2];
    s[4] = M[0][1] * M[1][3] - M[1][1] * M[0][3];
    s[5] = M[0][2] * M[1][3] - M[1][2] * M[0][3];

    c[0] = M[2][0] * M[3][1] - M[3][0] * M[2][1];
    c[1] = M[2][0] * M[3][2] - M[3][0] * M[2][2];
    c[2] = M[2][0] * M[3][3] - M[3][0] * M[2][3];
    c[3] = M[2][1] * M[3][2] - M[3][1] * M[2][2];
    c[4] = M[2][1] * M[3][3] - M[3][1] * M[2][3];
    c[5] = M[2][2] * M[3][3] - M[3][2] * M[2][3];

    /* Assumes it is invertible */
    float idet = 1.0f / (s[0] * c[5] - s[1] * c[4] + s[2] * c[3] + s[3] * c[2] - s[4] * c[1] + s[5] * c[0]);

    T[0][0] = (M[1][1] * c[5] - M[1][2] * c[4] + M[1][3] * c[3]) * idet;
    T[0][1] = (-M[0][1] * c[5] + M[0][2] * c[4] - M[0][3] * c[3]) * idet;
    T[0][2] = (M[3][1] * s[5] - M[3][2] * s[4] + M[3][3] * s[3]) * idet;
    T[0][3] = (-M[2][1] * s[5] + M[2][2] * s[4] - M[2][3] * s[3]) * idet;

    T[1][0] = (-M[1][0] * c[5] + M[1][2] * c[2] - M[1][3] * c[1]) * idet;
    T[1][1] = (M[0][0] * c[5] - M[0][2] * c[2] + M[0][3] * c[1]) * idet;
    T[1][2] = (-M[3][0] * s[5] + M[3][2] * s[2] - M[3][3] * s[1]) * idet;
    T[1][3] = (M[2][0] * s[5] - M[2][2] * s[2] + M[2][3] * s[1]) * idet;

    T[2][0] = (M[1][0] * c[4] - M[1][1] * c[2] + M[1][3] * c[0]) * idet;
    T[2][1] = (-M[0][0] * c[4] + M[0][1] * c[2] - M[0][3] * c[0]) * idet;
    T[2][2] = (M[3][0] * s[4] - M[3][1] * s[2] + M[3][3] * s[0]) * idet;
    T[2][3] = (-M[2][0] * s[4] + M[2][1] * s[2] - M[2][3] * s[0]) * idet;

    T[3][0] = (-M[1][0] * c[3] + M[1][1] * c[1] - M[1][2] * c[0]) * idet;
    T[3][1] = (M[0][0] * c[3] - M[0][1] * c[1] + M[0][2] * c[0]) * idet;
    T[3][2] = (-M[3][0] * s[3] + M[3][1] * s[1] - M[3][2] * s[0]) * idet;
    T[3][3] = (M[2][0] * s[3] - M[2][1] * s[1] + M[2][2] * s[0]) * idet;
}
static inline void mat4x4_orthonormalize(mat4x4 R, mat4x4 M)
{
    mat4x4_dup(R, M);
    float s = 1.;
    vec3 h;

    vec3_norm(R[2], R[2]);

    s = vec3_mul_inner(R[1], R[2]);
    vec3_scale(h, R[2], s);
    vec3_sub(R[1], R[1], h);
    vec3_norm(R[2], R[2]);

    s = vec3_mul_inner(R[1], R[2]);
    vec3_scale(h, R[2], s);
    vec3_sub(R[1], R[1], h);
    vec3_norm(R[1], R[1]);

    s = vec3_mul_inner(R[0], R[1]);
    vec3_scale(h, R[1], s);
    vec3_sub(R[0], R[0], h);
    vec3_norm(R[0], R[0]);
}

static inline void mat4x4_frustum(mat4x4 M, float l, float r, float b, float t, float n, float f)
{
    M[0][0] = 2.f * n / (r - l);
    M[0][1] = M[0][2] = M[0][3] = 0.f;

    M[1][1] = 2.f * n / (t - b);
    M[1][0] = M[1][2] = M[1][3] = 0.f;

    M[2][0] = (r + l) / (r - l);
    M[2][1] = (t + b) / (t - b);
    M[2][2] = -(f + n) / (f - n);
    M[2][3] = -1.f;

    M[3][2] = -2.f * (f * n) / (f - n);
    M[3][0] = M[3][1] = M[3][3] = 0.f;
}
static inline void mat4x4_ortho(mat4x4 M, float l, float r, float b, float t, float n, float f)
{
    M[0][0] = 2.f / (r - l);
    M[0][1] = M[0][2] = M[0][3] = 0.f;

    M[1][1] = 2.f / (t - b);
    M[1][0] = M[1][2] = M[1][3] = 0.f;

    M[2][2] = -2.f / (f - n);
    M[2][0] = M[2][1] = M[2][3] = 0.f;

    M[3][0] = -(r + l) / (r - l);
    M[3][1] = -(t + b) / (t - b);
    M[3][2] = -(f + n) / (f - n);
    M[3][3] = 1.f;
}
static inline void mat4x4_perspective(mat4x4 m, float y_fov, float aspect, float n, float f)
{
    /* NOTE: Degrees are an unhandy unit to work with.
     * linmath.h uses radians for everything! */
    float const a = (float)(1.f / tan(y_fov / 2.f));

    m[0][0] = a / aspect;
    m[0][1] = 0.f;
    m[0][2] = 0.f;
    m[0][3] = 0.f;

    m[1][0] = 0.f;
    m[1][1] = a;
    m[1][2] = 0.f;
    m[1][3] = 0.f;

    m[2][0] = 0.f;
    m[2][1] = 0.f;
    m[2][2] = -((f + n) / (f - n));
    m[2][3] = -1.f;

    m[3][0] = 0.f;
    m[3][1] = 0.f;
    m[3][2] = -((2.f * f * n) / (f - n));
    m[3][3] = 0.f;
}
static inline void mat4x4_look_at(mat4x4 m, vec3 eye, vec3 center, vec3 up)
{
    /* Adapted from Android's OpenGL Matrix.java.                        */
    /* See the OpenGL GLUT documentation for gluLookAt for a description */
    /* of the algorithm. We implement it in a straightforward way:       */

    /* TODO: The negation of of can be spared by swapping the order of
     *       operands in the following cross products in the right way. */
    vec3 f;
    vec3_sub(f, center, eye);
    vec3_norm(f, f);

    vec3 s;
    vec3_mul_cross(s, f, up);
    vec3_norm(s, s);

    vec3 t;
    vec3_mul_cross(t, s, f);

    m[0][0] = s[0];
    m[0][1] = t[0];
    m[0][2] = -f[0];
    m[0][3] = 0.f;

    m[1][0] = s[1];
    m[1][1] = t[1];
    m[1][2] = -f[1];
    m[1][3] = 0.f;

    m[2][0] = s[2];
    m[2][1] = t[2];
    m[2][2] = -f[2];
    m[2][3] = 0.f;

    m[3][0] = 0.f;
    m[3][1] = 0.f;
    m[3][2] = 0.f;
    m[3][3] = 1.f;

    mat4x4_translate_in_place(m, -eye[0], -eye[1], -eye[2]);
}

typedef float quat[4];
static inline void quat_identity(quat q)
{
    q[0] = q[1] = q[2] = 0.f;
    q[3] = 1.f;
}
static inline void quat_add(quat r, quat a, quat b)
{
    int i;
    for (i = 0; i < 4; ++i)
        r[i] = a[i] + b[i];
}
static inline void quat_sub(quat r, quat a, quat b)
{
    int i;
    for (i = 0; i < 4; ++i)
        r[i] = a[i] - b[i];
}
static inline void quat_mul(quat r, quat p, quat q)
{
    vec3 w;
    vec3_mul_cross(r, p, q);
    vec3_scale(w, p, q[3]);
    vec3_add(r, r, w);
    vec3_scale(w, q, p[3]);
    vec3_add(r, r, w);
    r[3] = p[3] * q[3] - vec3_mul_inner(p, q);
}
static inline void quat_scale(quat r, quat v, float s)
{
    int i;
    for (i = 0; i < 4; ++i)
        r[i] = v[i] * s;
}
static inline float quat_inner_product(quat a, quat b)
{
    float p = 0.f;
    int i;
    for (i = 0; i < 4; ++i)
        p += b[i] * a[i];
    return p;
}
static inline void quat_conj(quat r, quat q)
{
    int i;
    for (i = 0; i < 3; ++i)
        r[i] = -q[i];
    r[3] = q[3];
}
#define quat_norm vec4_norm
static inline void quat_mul_vec3(vec3 r, quat q, vec3 v)
{
    quat v_ = {v[0], v[1], v[2], 0.f};

    quat_conj(r, q);
    quat_norm(r, r);
    quat_mul(r, v_, r);
    quat_mul(r, q, r);
}
static inline void mat4x4_from_quat(mat4x4 M, quat q)
{
    float a = q[3];
    float b = q[0];
    float c = q[1];
    float d = q[2];
    float a2 = a * a;
    float b2 = b * b;
    float c2 = c * c;
    float d2 = d * d;

    M[0][0] = a2 + b2 - c2 - d2;
    M[0][1] = 2.f * (b * c + a * d);
    M[0][2] = 2.f * (b * d - a * c);
    M[0][3] = 0.f;

    M[1][0] = 2 * (b * c - a * d);
    M[1][1] = a2 - b2 + c2 - d2;
    M[1][2] = 2.f * (c * d + a * b);
    M[1][3] = 0.f;

    M[2][0] = 2.f * (b * d + a * c);
    M[2][1] = 2.f * (c * d - a * b);
    M[2][2] = a2 - b2 - c2 + d2;
    M[2][3] = 0.f;

    M[3][0] = M[3][1] = M[3][2] = 0.f;
    M[3][3] = 1.f;
}

static inline void mat4x4o_mul_quat(mat4x4 R, mat4x4 M, quat q)
{
    /*  XXX: The way this is written only works for othogonal matrices. */
    /* TODO: Take care of non-orthogonal case. */
    quat_mul_vec3(R[0], q, M[0]);
    quat_mul_vec3(R[1], q, M[1]);
    quat_mul_vec3(R[2], q, M[2]);

    R[3][0] = R[3][1] = R[3][2] = 0.f;
    R[3][3] = 1.f;
}
static inline void quat_from_mat4x4(quat q, mat4x4 M)
{
    float r = 0.f;
    r = sqrtf(1.f + M[0][0] + M[1][1] + M[2][2]);

    if (r < 1e-6)
    {
        q[0] = 1.f;
        q[1] = q[2] = q[3] = 0.f;
        return;
    }

    q[3] = r / 2.f;
    q[2] = -(M[1][0] - M[0][1]) / (2.f * r);
    q[1] = -(M[0][2] - M[2][0]) / (2.f * r);
    q[0] = -(M[2][1] - M[1][2]) / (2.f * r);
}

static inline void print_quat(std::string name, quat e, bool semicolon_separated = false, bool no_endline = false)
{

    std::cout << std::fixed << std::setprecision(9);
    if (semicolon_separated)
    {
        std::cout << "\033[32m" <<  e[3] << "\033[0m" << ";" << e[0] << ";" << e[1] << ";" << e[2];
        if (no_endline)
            std::cout << ";";
        else
            std::cout << std::endl;
    }

    else
        std::cout << std::setw(20) << name << " = (" << std::setw(8) << e[3] << " | " << std::setw(8) << e[0] << ", " << std::setw(8) << e[1] << ", " << std::setw(8) << e[2] << ")" << std::endl;
    std::cout << std::defaultfloat << std::setprecision(6);
}

static inline void print_vec4(std::string name, vec4 e, bool semicolon_separated = false, bool no_endline = false)
{

    std::cout << std::fixed << std::setprecision(5);
    if (semicolon_separated)
    {
        std::cout << e[0] << ";" << e[1] << ";" << e[2] << ";" << e[3];
        if (no_endline)
            std::cout << ";";
        else
            std::cout << std::endl;
    }

    else
    {
        std::cout << std::setw(26) << "┌" << std::setw(39) << "┐" << std::endl;
        std::cout << std::setw(20) << name << " = │" << std::setw(8) << e[0] << " " << std::setw(8) << e[1] << " " << std::setw(8) << e[2] << " " << std::setw(8) << e[3] << " │" << std::endl;
        std::cout << std::setw(26) << "└" << std::setw(39) << "┘" << std::endl;
    }

    std::cout << std::defaultfloat << std::setprecision(6);
}

static inline void print_mat4x4(std::string name, mat4x4 m)
{
    float det = m[0][0] * m[1][1] * m[2][2] + m[1][0] * m[2][1] * m[0][2] +
                m[2][0] * m[0][1] * m[1][2] - m[2][0] * m[1][1] * m[0][2] -
                m[1][0] * m[0][1] * m[2][2] - m[0][0] * m[2][1] * m[1][2];
    std::cout << std::fixed << std::setprecision(5);
    std::cout << std::setw(26) << "┌" << std::setw(39) << "┐" << std::endl;
    std::cout << std::setw(26) << "│" << std::setw(8) << m[0][0] << " " << std::setw(8) << m[0][1] << " " << std::setw(8) << m[0][2] << " " << std::setw(8) << m[0][3] << " │" << std::endl;
    std::cout << std::setw(20) << name << " = │" << std::setw(8) << m[1][0] << " " << std::setw(8) << m[1][1] << " " << std::setw(8) << m[1][2] << " " << std::setw(8) << m[1][3] << " │" << std::endl;
    std::cout << std::setw(26) << "│" << std::setw(8) << m[2][0] << " " << std::setw(8) << m[2][1] << " " << std::setw(8) << m[2][2] << " " << std::setw(8) << m[2][3] << " │" << std::endl;
    std::cout << std::setw(16) << "det =" << std::setw(6) << det << "│" << std::setw(8) << m[3][0] << " " << std::setw(8) << m[3][1] << " " << std::setw(8) << m[3][2] << " " << std::setw(8) << m[3][3] << " │" << std::endl;
    std::cout << std::setw(26) << "└" << std::setw(39) << "┘" << std::endl;
    std::cout << std::defaultfloat << std::setprecision(6);
}

typedef float vec17[17];

static inline void vec17_dup(vec17 r, vec17 const a)
{
    int i;
    for (i = 0; i < 17; ++i)
        r[i] = a[i];
}

static inline void vec17_add(vec17 r, vec17 const a, vec17 const b)
{
    int i;
    for (i = 0; i < 17; ++i)
        r[i] = a[i] + b[i];
}

static inline void vec17_subtract(vec17 r, vec17 const a, vec17 const b)
{
    int i;
    for (i = 0; i < 17; ++i)
        r[i] = a[i] - b[i];
}

static inline float vec17_mul_inner(vec17 a, vec17 b)
{
    float p = 0.f;
    int i;
    for (i = 0; i < 17; ++i)
        p += b[i] * a[i];
    return p;
}

static inline void vec17_scale(vec17 r, vec17 v, float s)
{
    int i;
    for (i = 0; i < 17; ++i)
        r[i] = v[i] * s;
}

typedef vec17 mat17x17[17];

static inline void mat17x17_transpose(mat17x17 M, mat17x17 N)
{
    int i, j;
    for (j = 0; j < 17; ++j)
        for (i = 0; i < 17; ++i)
            M[i][j] = N[j][i];
}

static inline void mat17x17_mul(mat17x17 M, mat17x17 a, mat17x17 b)
{
    int k, r, c;
    for (c = 0; c < 17; ++c)
        for (r = 0; r < 17; ++r)
        {
            M[c][r] = 0.f;
            for (k = 0; k < 17; ++k)
                M[c][r] += a[k][r] * b[c][k];
        }
}

static inline void mat17x17_dup(mat17x17 M, mat17x17 N)
{
    int i, j;
    for (i = 0; i < 17; ++i)
        for (j = 0; j < 17; ++j)
            M[i][j] = N[i][j];
}

static inline void mat17x17_identity(mat17x17 M)
{
    int i, j;
    for (i = 0; i < 17; ++i)
        for (j = 0; j < 17; ++j)
            M[i][j] = i == j ? 1.f : 0.f;
}

static inline void mat17x17_scale(mat17x17 M, mat17x17 a, float k)
{
    int i;
    for (i = 0; i < 17; ++i)
        vec17_scale(M[i], a[i], k);
}

static inline void mat17x17_add(mat17x17 M, mat17x17 a, mat17x17 b)
{
    int i;
    for (i = 0; i < 17; ++i)
        vec17_add(M[i], a[i], b[i]);
}

static inline void mat17x17_subtract(mat17x17 M, mat17x17 a, mat17x17 b)
{
    int i;
    for (i = 0; i < 17; ++i)
        vec17_subtract(M[i], a[i], b[i]);
}

static inline void mat17x17_mul_vec17(vec17 r, mat17x17 M, vec17 v)
{
    int i, j;
    for (j = 0; j < 17; ++j)
    {
        r[j] = 0.f;
        for (i = 0; i < 17; ++i)
            r[j] += M[i][j] * v[i];
    }
}

typedef vec4 mat4x17[17];
typedef vec17 mat17x4[4];

static inline void mat4x17_transpose(mat17x4 M, mat4x17 N)
{
    int i, j;
    for (j = 0; j < 17; ++j)
        for (i = 0; i < 4; ++i)
            M[i][j] = N[j][i];
}

static inline void mat17x17_mul_mat17x4(mat17x4 M, mat17x17 a, mat17x4 b)
{
    int k, r, c;
    for (c = 0; c < 4; ++c)
        for (r = 0; r < 17; ++r)
        {
            M[c][r] = 0.f;
            for (k = 0; k < 17; ++k)
                M[c][r] += a[k][r] * b[c][k];
        }
}

static inline void mat4x17_mul_mat17x17(mat4x17 M, mat4x17 a, mat17x17 b)
{
    int k, r, c;
    for (c = 0; c < 17; ++c)
        for (r = 0; r < 4; ++r)
        {
            M[c][r] = 0.f;
            for (k = 0; k < 17; ++k)
                M[c][r] += a[k][r] * b[c][k];
        }
}

static inline void mat4x17_mul_mat17x4(mat4x4 M, mat4x17 a, mat17x4 b)
{
    int k, r, c;
    for (c = 0; c < 4; ++c)
        for (r = 0; r < 4; ++r)
        {
            M[c][r] = 0.f;
            for (k = 0; k < 17; ++k)
                M[c][r] += a[k][r] * b[c][k];
        }
}

static inline void mat4x17_mul_vec17(vec4 r, mat4x17 M, vec17 v)
{
    int i, j;
    for (j = 0; j < 4; ++j)
    {
        r[j] = 0.f;
        for (i = 0; i < 17; ++i)
            r[j] += M[i][j] * v[i];
    }
}

static inline void mat17x4_mul_mat4x4(mat17x4 M, mat17x4 a, mat4x4 b)
{
    int k, r, c;
    for (c = 0; c < 4; ++c)
        for (r = 0; r < 17; ++r)
        {
            M[c][r] = 0.f;
            for (k = 0; k < 4; ++k)
                M[c][r] += a[k][r] * b[c][k];
        }
}

static inline void mat17x4_mul_mat4x17(mat17x17 M, mat17x4 a, mat4x17 b)
{
    int k, r, c;
    for (c = 0; c < 17; ++c)
        for (r = 0; r < 17; ++r)
        {
            M[c][r] = 0.f;
            for (k = 0; k < 4; ++k)
                M[c][r] += a[k][r] * b[c][k];
        }
}

static inline void mat17x4_mul_vec4(vec17 r, mat17x4 M, vec4 v)
{
    int i, j;
    for (j = 0; j < 17; ++j)
    {
        r[j] = 0.f;
        for (i = 0; i < 4; ++i)
            r[j] += M[i][j] * v[i];
    }
}

static inline void print_vec17(std::string name, vec17 e, bool semicolon_separated = false, bool no_endline = false)
{

    std::cout << std::fixed << std::setprecision(3);
    if (semicolon_separated)
    {
        for (int i = 0; i < 17; i++)
        {
            if ((i == 0)||(i==3)||(i==6)) std::cout << "\e[1m";
            std::cout << e[i];
            if ((i == 0)||(i==3)||(i==6)) std::cout << "\e[0m";
            if (i < 16)
                std::cout << ";";
        }
        if (no_endline)
            std::cout << ";";
        else
            std::cout << std::endl;
    }

    else
    {
        std::cout << std::setw(26) << "┌" << std::setw(139) << "┐" << std::endl;
        std::cout << std::setw(20) << name << " = │";
        for (int i = 0; i < 17; i++) {
            if ((i == 0)||(i==3)||(i==6)) std::cout << "\e[1m";
            if (i == 9) std::cout << "\033[32m";
            if ((i == 10)||(i==11)||(i==12)) std::cout << "\e[1m";
            std::cout << std::setw(7) << e[i] << " ";
            if (i == 9) std::cout << "\033[0m";
            if ((i == 10)||(i==11)||(i==12)) std::cout << "\e[0m";
            if ((i == 0)||(i==3)||(i==6)) std::cout << "\e[0m";
        }
            
        std::cout << "│" << std::endl;
        std::cout << std::setw(26) << "└" << std::setw(139) << "┘" << std::endl;
    }

    std::cout << std::defaultfloat << std::setprecision(6);
}

static inline void print_mat17x17(std::string name, mat17x17 m)
{
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::setw(26) << "┌" << std::setw(139) << "┐" << std::endl;
    for (int i = 0; i < 17; i++)
    {
        if (i == 8)
            std::cout << std::setw(20) << name << " = │";
        else
            std::cout << std::setw(26) << "│";
        for (int j = 0; j < 17; j++)
        {
            std::cout << std::setw(7) << m[j][i] << " ";
        }
        std::cout << "│" << std::endl;
    }
    std::cout << std::setw(26) << "└" << std::setw(139) << "┘" << std::endl;
    std::cout << std::defaultfloat << std::setprecision(6);
}

static inline void print_mat4x17(std::string name, mat4x17 m)
{
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::setw(26) << "┌" << std::setw(139) << "┐" << std::endl;
    for (int i = 0; i < 4; i++)
    {
        if (i == 1)
            std::cout << std::setw(20) << name << " = │";
        else
            std::cout << std::setw(26) << "│";
        for (int j = 0; j < 17; j++)
        {
            std::cout << std::setw(7) << m[j][i] << " ";
        }
        std::cout << "│" << std::endl;
    }
    std::cout << std::setw(26) << "└" << std::setw(139) << "┘" << std::endl;
    std::cout << std::defaultfloat << std::setprecision(6);
}

static inline void print_mat17x4(std::string name, mat17x4 m)
{
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::setw(26) << "┌" << std::setw(35) << "┐" << std::endl;
    for (int i = 0; i < 17; i++)
    {
        if (i == 8)
            std::cout << std::setw(20) << name << " = │";
        else
            std::cout << std::setw(26) << "│";
        for (int j = 0; j < 4; j++)
        {
            std::cout << std::setw(7) << m[j][i] << " ";
        }
        std::cout << "│" << std::endl;
    }
    std::cout << std::setw(26) << "└" << std::setw(35) << "┘" << std::endl;
    std::cout << std::defaultfloat << std::setprecision(6);
}

#endif
